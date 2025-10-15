import torch.nn as nn
import torch
from torch.nn import functional as F
from Adder import adder2d
from Layer import Layer

class AdderNet(nn.Module):
    def __init__(self, num_classes=10):
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
        self.fc = nn.Linear(64, num_classes)

    def load_manual_weights(self, weights_dict):

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights_dict:
                    param.copy_(weights_dict[name])
            
            for name, buffer in self.named_buffers():
                if name in weights_dict:
                    buffer.copy_(weights_dict[name])

    def forward(self, x, save_activations = False):
        if save_activations:
            self.activations['input_activation'] = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if save_activations:
            self.activations['prelayer_activation'] = out
        out = self.layer1(out)
        if save_activations:
            self.activations['layer1_activation'] = out
        out = self.layer2(out)
        if save_activations:
            self.activations['layer2_activation'] = out
        out = self.layer3(out)
        if save_activations:
            self.activations['layer3_activation'] = out
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out

    def classification(self, x):
        out = self.forward(x)
        return F.softmax(out, dim=1)