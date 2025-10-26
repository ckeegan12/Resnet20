import torch.nn as nn
from block2_0 import ResidualBlock2_0
from Adder2_0 import adder2d2_0

class Layer2_0(nn.Module):
  """
  Layer make-up for the AdderNet model using the Residual blocks with adder operations only
  """
  def __init__(self, in_channels, out_channels, bits, max_delta=2.5, num_blocks=3):
      super(Layer2_0, self).__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels

      downsample = None
      stride = 1

      if in_channels != out_channels:
          self.downsample_adder = adder2d2_0(in_channels, out_channels, kernel_size=1, bits=bits, max_delta=max_delta, stride=2, padding=0, bias=False)
          self.downsample_bn = nn.BatchNorm2d(out_channels)
          downsample = (self.downsample_adder, self.downsample_bn)
          stride = 2
      else:
          stride = 1

      self.blocks = nn.ModuleList()

      self.blocks.append(ResidualBlock2_0(in_channels=in_channels, out_channels=out_channels, 
                                      stride=stride, downsample=downsample, bits=bits, max_delta=max_delta))

      for _ in range(num_blocks - 1):
          self.blocks.append(ResidualBlock2_0(in_channels=out_channels, out_channels=out_channels, padding=1, bits=bits, max_delta=max_delta))
  
  def forward(self, x):
      out = x
      for block in self.blocks:
          out = block(out)
      return out