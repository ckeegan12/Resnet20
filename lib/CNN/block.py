import torch.nn as nn

class ResidualBlock(nn.Module):
  """
  Residual Blocks for Resnet20 for a 3x3 kernel, stride of 1, and no downsampling
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, downsample=None):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out