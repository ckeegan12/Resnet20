import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

class adder2_0(Function):
    @staticmethod
    def forward(ctx, W_col, X_col, bits): 
        ctx.save_for_backward(W_col, X_col)
        
        q_bits = bits
        
        alpha = W_col.abs().max()
        scale = alpha / (2**(q_bits-1) - 1)
        scale = scale.clamp(min=1e-6)
        
        q_min = -(2**(q_bits-1))
        q_max = 2**(q_bits-1) - 1
        
        W_q = torch.round(W_col / scale)
        W_clip_q = torch.clamp(W_q, q_min, q_max)
        
        W_clip = W_clip_q * scale
        
        # W_bias = -|W - W_clip|
        W_bias = -(W_col - W_clip).abs()
        sum_W_bias_per_filter = W_bias.sum(dim=1)
        
        # New_SAD = -sum|X - W_clip|
        New_SAD = -(W_clip.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        
        # output = New_SAD + sum W_bias
        output = New_SAD + sum_W_bias_per_filter.unsqueeze(1)
        
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col, None 
    
class adder2d2_0(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, bits, stride=1, padding=0, bias = False):
        super(adder2d2_0, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.bits = bits
        self.adder = torch.nn.Parameter(
            nn.init.normal_(
                torch.randn(output_channel, input_channel, kernel_size, kernel_size)
            )
        )

    def forward(self, x):
        n_x, d_x, h_x, w_x = x.size()
        h_out = (h_x - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (w_x - self.kernel_size + 2 * self.padding) / self.stride + 1
        h_out, w_out = int(h_out), int(w_out)

        n_filters = self.output_channel

        X_unfolded = nn.functional.unfold(
            x, 
            kernel_size=(self.kernel_size, self.kernel_size), 
            dilation=1, 
            padding=self.padding, 
            stride=self.stride
        )
        
        X_col = X_unfolded.permute(1, 2, 0)
        X_col = X_col.contiguous()
        X_col = X_col.view(X_col.size(0), -1)

        W_col = self.adder.view(n_filters, -1)

        out = adder2_0.apply(W_col, X_col, self.bits)
        
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        
        return out