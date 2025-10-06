import torch.nn as nn
from torch.nn import functional as F
from Adder import adder2d
from Layer import Layer

class AdderNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AdderNet, self).__init__()
        # Initial convolution layer
        self.adder1 = adder2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Residual layers
        self.layer1 = Layer(16, 16, num_blocks=3)
        self.layer2 = Layer(16, 32, num_blocks=3)
        self.layer3 = Layer(32, 64, num_blocks=3)
        
        # Fully connected layers
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.adder1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out

    def classification(self, x):
        out = self.forward(x)
        return F.softmax(out, dim=1)