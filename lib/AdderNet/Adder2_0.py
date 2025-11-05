import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

class adder2_0(Function):
    @staticmethod
    def forward(ctx, W_col, X_col, bits, delta):
        # W_col: (out_channels, in_channels*k*k), X_col: (in_channels*k*k, locations*batch)
        ctx.save_for_backward(W_col, X_col)
        ctx.bits = bits
        
        q_bits = bits
        max_val = 2.5
        
        # Quantization bounds: [-2^(q-1), 2^(q-1)-1]
        q_min = -(2**(q_bits-1))
        q_max = 2**(q_bits-1) - 1

        ctx.delta = max_val / q_max
        
        # During training: quantize weights on-the-fly
        # During inference: weights are already quantized integers from FBR preprocessing
        if ctx.needs_input_grad[0]:  # Training mode
            # Quantize: W_q = round(W / δ)
            W_q = torch.round(W_col / delta)
            W_clip = torch.clamp(W_q, q_min, q_max)
        else:  # Inference mode (eval)
            # Weights are already quantized integers from post_proc_act_quant.py
            W_clip = W_col
        
        # Compute output: -Σ|X - W_clip|
        # This is the core AdderNet operation (L1 distance)
        output = -(W_clip.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        
        # Gradient w.r.t. weights
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        
        # Gradient w.r.t. input
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col, None, None
    
class adder2d2_0(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bits, 
                 max_val=2.5, stride=1, padding=0, bias=False):
        super(adder2d2_0, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.bits = bits
        self.delta = max_val / (2**(bits-1) - 1)
        
        # Weight parameter: (out_channels, in_channels, k, k)
        # During training: contains float weights
        # After FBR preprocessing: contains quantized integer weights
        self.adder = torch.nn.Parameter(
            nn.init.normal_(
                torch.randn(output_channel, input_channel, kernel_size, kernel_size)
            )
        )

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (w_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        n_filters = self.output_channel

        # Unfold input into columns for efficient computation
        # X_unfolded: (batch, in_channels*k*k, locations)
        X_unfolded = nn.functional.unfold(
            x, 
            kernel_size=(self.kernel_size, self.kernel_size), 
            dilation=1, 
            padding=self.padding, 
            stride=self.stride
        )
        
        # Reshape for computation
        # X_col: (in_channels*k*k, locations*batch)
        X_col = X_unfolded.permute(1, 2, 0).contiguous().view(X_unfolded.size(1), -1)
        
        # W_col: (out_channels, in_channels*k*k)
        W_col = self.adder.view(n_filters, -1)

        # Apply adder operation: -Σ|X - W_clip|
        out = adder2_0.apply(W_col, X_col, self.bits, self.delta)
        
        # Reshape output: (batch, out_channels, h_out, w_out)
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        
        return out