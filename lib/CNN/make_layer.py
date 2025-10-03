import torch.nn as nn
from block import ResidualBlock

class Layer(nn.Module):
  """
  Layer make-up for the model using the Residual blocks
  """
  def __init__(self, in_channels, out_channels, num_blocks=3):
    super(Layer, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    downsample = None
    stride = 1

    if in_channels != out_channels:
      self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
      self.downsample_bn = nn.BatchNorm2d(out_channels)
      downsample = (self.downsample_conv, self.downsample_bn)
      stride = 2
    else:
      stride = 1

    self.blocks = nn.ModuleList()
    self.blocks.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, 
                                    stride=stride, downsample=downsample))

    for _ in range(num_blocks - 1):
      self.blocks.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, padding=1))
  
  def forward(self, x):
    out = x
    for block in self.blocks:
      out = block(out)
    return out      