from torch import nn
from torch.nn import functional as F
from make_layer import Layer


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Residual layers
        self.layer1 = Layer(16, 3)
        self.layer2 = Layer(32, 3)
        self.layer3 = Layer(64, 3)
        
        # Fully connected layers
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1.forward(out)
        out = self.layer2.forward(out)
        out = self.layer3.forward(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out

    def classification(self, x):
        out = self.forward(x)
        return F.softmax(out, dim=1)