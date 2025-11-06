import torch.nn as nn
import torch
from torch.nn import functional as F
from Layer import Layer
from layer2_0 import Layer2_0

class AdderNet(nn.Module):
    """
    Standard AdderNet model (non-quantized version)
    """
    def __init__(self, num_classes=10, load_weights=None):
        super(AdderNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Residual layers
        self.layer1 = Layer(16, 16, num_blocks=3)
        self.layer2 = Layer(16, 32, num_blocks=3)
        self.layer3 = Layer(32, 64, num_blocks=3)
        
        # Fully connected layer
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

    def forward(self, x, save_activations=False):
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
    

class AdderNet2_0(nn.Module):
    """
    AdderNet 2.0 with Fusion Bias Removal (FBR)
    
    FBR Preprocessing:
    - Adder weights → quantized integers: W_clip ∈ [-2^(q-1), 2^(q-1)-1]
    - BatchNorm running_mean → adjusted: μ' = round(μ/δ) + Σ|W_q - W_clip|
    - BatchNorm bias → quantized: β' = β/δ
    - Final FC weights → scaled: W_fc' = W_fc * δ
    
    During inference:
    - Adder: outputs -Σ|X - W_clip| (integer operations)
    - BatchNorm: Y = γ * (X - μ') / √(σ² + ε) + β'
    - The weight bias is implicitly handled via the adjusted running_mean
    """
    def __init__(self, num_classes=10, load_weights=None):
        super(AdderNet2_0, self).__init__()

        # Initial convolution layer (standard conv, not quantized)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Quantized adder layers with FBR
        self.layer1 = Layer2_0(16, 16, num_blocks=3)
        self.layer2 = Layer2_0(16, 32, num_blocks=3)
        self.layer3 = Layer2_0(32, 64, num_blocks=3)
        
        # Fully connected layer
        self.fc = nn.Conv2d(64, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        if load_weights is not None:
            self.load_manual_weights(load_weights)
        
        self.activations = {}

    def load_manual_weights(self, weights_dict):
        """
        Expected preprocessing:
        1. Adder weights: quantized integers W_clip
        2. BN running_mean: adjusted with weight bias (μ' = round(μ/δ) + bias_sum)
        3. BN bias: quantized by delta (β' = β/δ)
        4. BN weight: quantized by delta (γ' = γ/δ) for bn1 only
        5. FC weights: scaled by delta (W_fc' = W_fc * δ)
        """
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

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations['input_activation_2.0'] = x.clone()
        
        # Initial conv + BN + ReLU
        # Note: bn1.weight and bn1.bias are divided by delta
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if save_activations:
            self.activations['prelayer_activation_2.0'] = out.clone()
        
        # Quantized adder layers with FBR
        out = self.layer1(out)
        if save_activations:
            self.activations['layer1_activation_2.0'] = out.clone()
        
        out = self.layer2(out)
        if save_activations:
            self.activations['layer2_activation_2.0'] = out.clone()
        
        out = self.layer3(out)
        if save_activations:
            self.activations['layer3_activation_2.0'] = out.clone()
        
        # Global average pooling
        out = F.avg_pool2d(out, 8)
        
        # Final FC layer (weights scaled by delta in preprocessing)
        out = self.fc(out)
        out = self.bn2(out)
        out = out.view(out.size(0), -1)
        
        return out

    def classification(self, x):
        out = self.forward(x)
        return F.softmax(out, dim=1)