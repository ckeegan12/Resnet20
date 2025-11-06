import torch.nn as nn
from Adder2_0 import adder2d2_0

class ResidualBlock2_0(nn.Module):
    """
    Residual Block for AdderNet 2.0 with Fusion Bias Removal (FBR)
    
    Uses standard BatchNorm2d since the FBR preprocessing (post_proc_act_quant.py) 
    adjusts the BatchNorm parameters offline. No dynamic weight bias adjustment needed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, downsample=None):
        super(ResidualBlock2_0, self).__init__()
        
        self.adder1 = adder2d2_0(in_channels, out_channels, kernel_size, 
                                 stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.adder2 = adder2d2_0(out_channels, out_channels, kernel_size, 
                                 stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        
        # First adder + BN + ReLU
        out = self.adder1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second adder + BN
        out = self.adder2(out)
        out = self.bn2(out)
        
        # Downsample residual if needed
        if self.downsample is not None:
            downsample_adder, downsample_bn = self.downsample
            residual = downsample_adder(x)
            residual = downsample_bn(residual)
        
        # Add residual and apply ReLU
        out += residual
        out = self.relu(out)
        
        return out