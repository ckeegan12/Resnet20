import torch.nn as nn
from block2_0 import ResidualBlock2_0
from Adder2_0 import adder2d2_0

class Layer2_0(nn.Module):
    """
    Layer composition for AdderNet 2.0 with Fusion Bias Removal (FBR)
    
    Creates a sequence of residual blocks with adder operations.
    Uses standard BatchNorm2d since FBR preprocessing handles bias fusion offline.
    """
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(Layer2_0, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        downsample = None
        stride = 1

        # Create downsample path if channel dimensions change
        if in_channels != out_channels:
            self.downsample_adder = adder2d2_0(in_channels, out_channels, kernel_size=1, 
                                                stride=2, padding=0, bias=False)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
            downsample = (self.downsample_adder, self.downsample_bn)
            stride = 2
        else:
            stride = 1

        self.blocks = nn.ModuleList()

        # First block (may have stride=2 for downsampling)
        self.blocks.append(ResidualBlock2_0(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            stride=stride, 
                                            downsample=downsample, 
                                            ))

        # Remaining blocks (stride=1)
        for _ in range(num_blocks - 1):
            self.blocks.append(ResidualBlock2_0(in_channels=out_channels, 
                                                out_channels=out_channels, 
                                                padding=1, 
                                                ))
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out