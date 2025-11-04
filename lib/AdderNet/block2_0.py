import torch.nn as nn
from batchnorm2_0 import BatchNorm2dWithAdderBias
from Adder2_0 import adder2d2_0

class ResidualBlock2_0(nn.Module):
  """
  Residual Blocks for AdderNet2.0 for a 3x3 kernel, stride of 1, and no downsampling
  """
  def __init__(self, in_channels, out_channels, bits, max_delta=2.5, kernel_size=3, stride=1, padding=1, downsample=None):
    super(ResidualBlock2_0, self).__init__()
    self.adder1 = adder2d2_0(in_channels, out_channels, kernel_size, bits, max_delta, stride, padding, bias=False)
    self.bn1 = BatchNorm2dWithAdderBias(out_channels, delta=max_delta)
    self.relu = nn.ReLU()
    self.adder2 = adder2d2_0(out_channels, out_channels, kernel_size, bits, max_delta, stride=1, padding=padding, bias=False)
    self.bn2 = BatchNorm2dWithAdderBias(out_channels, delta=max_delta)
    self.downsample = downsample

  def forward(self, x):
    residual = x
    out = self.adder1(x)
    out = self.bn1(out)
    self.bn1.set_adder_weight_bias(self.adder1.get_weight_bias())
    out = self.relu(out)
    out = self.adder2(out)
    self.bn2.set_adder_weight_bias(self.adder2.get_weight_bias())
    out = self.bn2(out)
    if self.downsample is not None:
      downsample_adder, downsample_bn = self.downsample
      residual = downsample_adder(x)
      if isinstance(downsample_bn, BatchNorm2dWithAdderBias): 
                downsample_bn.set_adder_weight_bias(downsample_adder.get_weight_bias()) 
      residual = downsample_bn(residual)
    out += residual
    out = self.relu(out)
    return out