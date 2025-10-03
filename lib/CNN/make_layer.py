import torch.nn as nn
from block import ResidualBlock

class Layer(nn.Module):
  """
  Layer make-up for the model using the Residual blocks
  """
  def __intit__(self, in_channels, out_channels, num_blocks=3):
    super(Layer,self).__intit__()
    self.in_channels = in_channels
    self.out_channels = out_channels
  
    self.blocks = nn.ModuleList()
    self.blocks.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels))

    for nums in range(num_blocks):
      self.blocks.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, padding=1))
  
  def foward(self, x):
    out = x
    for block in self.blocks:
      out = block(out)
    return out      