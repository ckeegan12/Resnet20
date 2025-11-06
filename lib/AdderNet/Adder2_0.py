
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

class adder2_0(Function):
    """
    AdderNet 2.0 forward operation with FBR (Fusion Bias Removal)
    """
    @staticmethod
    def forward(ctx, W_col, X_col):
        # W_col: (out_channels, in_channels*k*k) - already quantized integers
        # X_col: (in_channels*k*k, locations*batch) - activations
        ctx.save_for_backward(W_col, X_col)
        
        # Core AdderNet operation: -Σ|W - X|
        output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
class adder2d2_0(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, 
                 stride=1, padding=0, bias=False):
        super(adder2d2_0, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        
        # Weight parameter: (out_channels, in_channels, k, k)
        # During training: contains float weights
        # After FBR preprocessing: contains quantized integer weights
        self.adder = torch.nn.Parameter(
            nn.init.normal_(
                torch.randn(output_channel, input_channel, kernel_size, kernel_size)
            )
        )
        
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        n_x, d_x, h_x, w_x = x.size()
        n_filters = self.output_channel
        
        # Calculate output dimensions
        h_out = (h_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (w_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # Unfold input into columns
        # X_col shape: (batch, in_channels*k*k, h_out*w_out)
        X_col = torch.nn.functional.unfold(
            x.view(1, -1, h_x, w_x), 
            self.kernel_size, 
            dilation=1, 
            padding=self.padding, 
            stride=self.stride
        ).view(n_x, -1, h_out * w_out)
        
        # Reshape: (in_channels*k*k, h_out*w_out*batch)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        
        # Reshape weights: (out_channels, in_channels*k*k)
        W_col = self.adder.view(n_filters, -1)
        
        # Apply adder operation (NO quantization logic here!)
        out = adder2_0.apply(W_col, X_col)
        
        # Reshape output: (out_channels, h_out, w_out, batch) -> (batch, out_channels, h_out, w_out)
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        
        # Add bias if needed
        if self.bias:
            out += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return out