import torch.nn as nn
import torch

class BatchNorm2dWithAdderBias(nn.BatchNorm2d):
    """
    BatchNorm2d that implements the equation:
    Y = α * (-Σ|X - W_clip|) / √(σ² + ε) + (β - α * (μ - ΣW_bias) / √(σ² + ε))
    
    Note: W_bias from adder is negative: W_bias = -(W_q - W_clip).abs()
    So: μ - W_bias = μ - (W_q - W_clip).abs()
    """
    def __init__(self, num_features, bits, max_val=2.5, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm2dWithAdderBias, self).__init__(num_features, eps, momentum, affine)
        self.bits = bits
        self.delta = max_val / (2**(bits-1) - 1)
        self.adder_weight_bias = None
    
    def set_adder_weight_bias(self, weight_bias):
        if weight_bias is not None:
            self.adder_weight_bias = weight_bias.detach()
        else:
            self.adder_weight_bias = None
    
    def forward(self, x):
        if self.training:
            # During training, use standard batch norm
            return super().forward(x)
        else:
            # Inference mode - implement the AdderNet 2.0 FBR
            # Y = α * x / √(σ² + ε) + (β - α * (μ - ΣW_bias) / √(σ² + ε))
            # where x is already -Σ|X - W_clip| from the adder layer
            
            # Compute standard deviation
            running_std = torch.sqrt(self.running_var + self.eps)
            
            if self.affine:
                # First term: α * x / √(σ² + ε)
                out = self.weight.view(1, -1, 1, 1) * x / running_std.view(1, -1, 1, 1)
                
                # Second term: β/δ - α * (μ - ΣW_bias) / √(σ² + ε)
                # Note: W_bias is negative, so μ - W_bias = μ + |W_bias|
                if self.adder_weight_bias is not None:
                    # Subtract the negative W_bias (which adds the absolute value)
                    adjusted_mean = self.running_mean - self.adder_weight_bias.to(self.running_mean.device)
                else:
                    adjusted_mean = self.running_mean
                
                # Quantize bias by delta (from post_proc_act_quant.py)
                bias_quantized = self.bias / self.delta
                
                # Complete bias term: β/δ - α * (μ - ΣW_bias) / √(σ² + ε)
                bias_term = bias_quantized - self.weight * adjusted_mean / running_std
                
                # Add bias term
                out = out + bias_term.view(1, -1, 1, 1)
            else:
                # Without affine transformation, just normalize
                out = x / running_std.view(1, -1, 1, 1)
            
            return out