import torch.nn as nn

class BatchNorm2dWithAdderBias(nn.BatchNorm2d):
    """
    BatchNorm2d that adds weight bias from adder operation to running_mean
    and quantizes the bias term by delta
    """
    def __init__(self, num_features, bits, max_val=2.5, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm2dWithAdderBias, self).__init__(num_features, eps, momentum, affine)
        self.bits = bits
        self.delta = max_val / (2**(bits-1) - 1)
        self.adder_weight_bias = None
    
    def set_adder_weight_bias(self, weight_bias):
        self.adder_weight_bias = weight_bias
    
    def forward(self, x):
        if self.training:
            # During training, use standard batch norm
            return super().forward(x)
        else:
            # running_mean_modified = running_mean + weight_bias
            # bias_quantized = bias / delta
            
            if self.adder_weight_bias is not None:
                # Add weight bias to running mean
                running_mean_modified = self.running_mean + self.adder_weight_bias
            else:
                running_mean_modified = self.running_mean
            
            # Normalize: (x - (running_mean + weight_bias)) / sqrt(running_var + eps)
            out = (x - running_mean_modified.view(1, -1, 1, 1)) / (self.running_var.view(1, -1, 1, 1) + self.eps).sqrt()
            
            if self.affine:
                # Apply weight (scaling in batchnorm gamma)
                out = out * self.weight.view(1, -1, 1, 1)
                # Apply bias (bias term in batchnorm beta)
                if self.delta is not None:
                    bias_quantized = self.bias / self.delta
                else:
                    bias_quantized = self.bias
                out = out + bias_quantized.view(1, -1, 1, 1)
            
            return out