import torch.nn as nn
import torch
from torch.nn import functional as F
from Adder import adder2d
from Layer import Layer

class AdderNet(nn.Module):
    def __init__(self, num_classes=10, load_weights = None):
        super(AdderNet, self).__init__()
        # Initial convolution layer for AdderNet
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Residual layers
        self.layer1 = Layer(16, 16, num_blocks=3)
        self.layer2 = Layer(16, 32, num_blocks=3)
        self.layer3 = Layer(32, 64, num_blocks=3)
        
        # Fully connected layers
        self.fc = nn.Conv2d(64, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        if load_weights is not None:
            self.load_manual_weights(load_weights)
        
        self.activations = {}

    def load_manual_weights(self, weights_dict):
        with torch.no_grad():
          for name, param in self.named_parameters():
            if name in weights_dict:
                weight_value = weights_dict[name]
                if weight_value.shape == param.shape:
                    param.copy_(weight_value.to(param.device))
        
          for name, buffer in self.named_buffers():
              if name in weights_dict:
                  buffer_value = weights_dict[name]
                  if buffer_value.shape == buffer.shape:
                      buffer.copy_(buffer_value.to(buffer.device))

    def forward(self, x, save_activations = False):
        if save_activations:
            self.activations['input_activation'] = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if save_activations:
            self.activations['prelayer_activation'] = out.clone()
        out = self.layer1(out)
        if save_activations:
            self.activations['layer1_activation'] = out.clone()
        out = self.layer2(out)
        if save_activations:
            self.activations['layer2_activation'] = out.clone()
        out = self.layer3(out)
        if save_activations:
            self.activations['layer3_activation'] = out.clone()
        out = F.avg_pool2d(out, 8)
        out = self.fc(out)
        out = self.bn2(out)
        out = out.view(out.size(0), -1)
        return out

    def classification(self, x):
        out = self.forward(x)
        return F.softmax(out, dim=1)