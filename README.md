# AdderNet 2.0: Optimal FPGA Acceleration with AOQ and FBR

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 🎯 Overview

This repository implements **AdderNet 2.0**, a multiplication-free neural network architecture that achieves **top-1 accuracy comparable to CNNs** while using only addition operations. Implementation features:

- **High Accuracy**: AdderNet2.0 **competitive top-1 accuracy** on CIFAR-10
- **Efficient Operations**: Replaces costly MAC operations with efficient SAD (Sum of Absolute Differences)
- **Novel Quantization**: Activation-Oriented Quantization (AOQ) enables INT6 precision
- **Memory Optimization**: Fused Bias Removal (FBR) reduces on-chip memory requirements with a theoretical memory reduction of 81.25% (≈5.33× smaller)

## 📊 Key Features

AdderNet 2.0 achieves **Accuracy of 90.83% on CIFAR-10** using ResNet-20 architecture while using efficient INT6 quantization.

## 🧮 Mathematical Foundation

### 1. Standard Convolution (CNN)

In traditional CNNs, the convolution operation performs multiplication and accumulation (MAC):

$$Y(m, n, c_o) = \sum_{i=1}^{k} \sum_{j=1}^{k} \sum_{l=1}^{c_i} X(m+i, n+j, l) \times W(i, j, l, c_o)$$

where:
- $W \in \mathbb{R}^{k \times k \times c_i \times c_o}$ is the 4-D weight tensor
- $X \in \mathbb{R}^{h \times w \times c_i}$ is the input feature map
- $Y$ is the output feature map
- $k$ is the kernel size, $c_i$ and $c_o$ are input and output channels

### 2. AdderNet Operation

AdderNet replaces multiplication with $\ell_1$-norm distance metric (Sum of Absolute Differences):

$$Y(m, n, c_o) = -\sum_{i=1}^{k} \sum_{j=1}^{k} \sum_{l=1}^{c_i} |X(m+i, n+j, l) - W(i, j, l, c_o)|$$

This formulation:
- Eliminates costly multiplications
- Uses efficient absolute difference operations
- Maintains competitive accuracy with CNNs

### 3. Activation-Oriented Quantization (AOQ)

Traditional weight quantization fails for AdderNet due to the coupled nature of weights and activations in SAD operations. AOQ addresses this through:

#### Step 1: Activation Quantization
For $q$-bit quantization, BN outputs are quantized:

$$\text{BN}(\cdot) \in [-2^{q-1}, 2^{q-1} - 1]$$

This leads to $(q-1)$-bit non-negative ReLU outputs:

$$\text{ReLU}(\cdot) \in [0, 2^{q-1} - 1]$$

#### Step 2: Weight Clipping
Weights are clipped to the same range as activations:

$$W_{clip}(w) = \begin{cases} 
-2^{q-1} & \text{if } w < -2^{q-1} \\
w & \text{if } w \in [-2^{q-1}, 2^{q-1} - 1] \\
2^{q-1} - 1 & \text{if } w > 2^{q-1} - 1
\end{cases}$$

#### Step 3: Weight Bias Compensation
The difference between original and clipped weights is computed as:

$$W_{bias} = -|W - W_{clip}|$$

The SAD operation is then mathematically equivalent:

$$-\sum |X - W| \equiv -\sum |X - W_{clip}| + \sum W_{bias}$$

where $\sum W_{bias}$ is a constant pre-computed after training with **zero hardware overhead**.

### 4. Batch Normalization

Batch normalization normalizes the SAD features:

$$Y = \alpha \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta = \frac{\alpha}{\sqrt{\sigma^2 + \epsilon}} X + \left(\beta - \alpha \frac{\mu}{\sqrt{\sigma^2 + \epsilon}}\right)$$

where:
- $\mu$ and $\sigma$ are the running mean and variance
- $\alpha$ and $\beta$ are learnable scale and shift parameters
- $\epsilon$ is a stability constant

### 5. Fused Bias Removal (FBR)

FBR optimizes memory usage through two-stage optimization:

#### Stage 1: Bias Fusion
The weight bias is fused into the BN bias:

$$\beta' = \beta - \alpha \frac{\mu - \sum W_{bias}}{\sqrt{\sigma^2 + \epsilon}}$$

#### Stage 2: Dynamic Bias Removal
During SAD computation, the fused bias is removed on-the-fly:

$$\text{FB} = \mu - \sum W_{bias}$$

$$\text{Output} = -\sum |X - W_{clip}| - \frac{\text{FB}}{K_R \times K_C}$$

This reduces intermediate feature memory bitwidth significantly (for ResNet-20).

## 🏗️ Architecture

### Standard CNN ResNet-20
```
Input (3×32×32)
    ↓
Conv2d (3→16, 3×3) → BN → ReLU
    ↓
Layer1: 3× ResBlock (16→16)
    ↓
Layer2: 3× ResBlock (16→32, stride=2)
    ↓
Layer3: 3× ResBlock (32→64, stride=2)
    ↓
AvgPool (8×8) → FC (64→10)
```

### AdderNet 2.0 ResNet-20
```
Input (3×32×32)
    ↓
Conv2d (3→16, 3×3) → BN → ReLU
    ↓
Layer1: 3× AdderBlock (16→16, INT6)
    ↓
Layer2: 3× AdderBlock (16→32, INT6, stride=2)
    ↓
Layer3: 3× AdderBlock (32→64, INT6, stride=2)
    ↓
AvgPool (8×8) → FC (64→10)
```

Each AdderBlock uses:
- **Adder2d** layers instead of Conv2d
- **AOQ-quantized weights** (W_clip)
- **FBR-optimized** BatchNorm with adjusted running_mean

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ckeegan12/Resnet20_Addernet.git
cd Resnet20_Addernet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy
```

## 🚀 Quick Start

### Training CNN Model
```python
from lib.CNN.model import ResNet20
from lib.CNN.Train import model_training
import torch

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet20(num_classes=10).to(device)

# Train model
trainer = model_training(model, train_loader, epochs=200)
train_loss = trainer.forward(device)
```

### Training AdderNet Model
```python
from lib.AdderNet.model import AdderNet
from lib.AdderNet.Train import model_training
import torch

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdderNet(num_classes=10).to(device)

# Train model
trainer = model_training(model, lr=0.05, train_loader=train_loader, epochs=200)
train_loss = trainer.forward(device)
```

### Quantization and Inference
```python
from lib.AdderNet.quantization_encoder import Quant

# Quantize weights (INT6)
quantized_weights, scale = Quant.symmetric_quantization(weight_tensor, bits=6)
```
**Note** Inference and training were ran on google colab using the T4 GPU to speed up execution time
## 📁 Repository Structure

```
.
├── lib/
│   ├── CNN/
│   │   ├── model.py              # ResNet-20 CNN implementation
│   │   ├── block.py              # CNN residual blocks
│   │   ├── make_layer.py         # Layer composition
│   │   ├── Train.py              # CNN training loop
│   │   ├── quantization_encoder.py
│   │   └── quantization_decoding.py
│   └── AdderNet/
│       ├── model.py              # AdderNet & AdderNet 2.0 models
│       ├── Adder.py              # Original adder operation
│       ├── Adder2_0.py           # AdderNet 2.0 with FBR
│       ├── Block.py              # AdderNet residual blocks
│       ├── block2_0.py           # AdderNet 2.0 residual blocks
│       ├── Layer.py              # Layer composition
│       ├── layer2_0.py           # AdderNet 2.0 layer composition
│       ├── Train.py              # AdderNet training loop
│       ├── quantization_encoder.py
│       └── quantization_decoding.py
└── README.md
```

## 🔬 Key Features

### 1. Activation-Oriented Quantization (AOQ)
- Novel quantization strategy that preserves model accuracy
- Enables low-bit quantization (INT6 and INT8) and unsigned INT5 due to ReLU activations zeroing all negative variables
- Maintains coupled weight-activation relationship in SAD operations

### 2. Fused Bias Removal (FBR)
- Reduces on-chip memory bitwidth requirements
- Fuses weight bias into BatchNorm parameters
- Dynamic bias removal during SAD computation

## 📈 Hardware Efficiency

AdderNet 2.0 demonstrates significant improvements in hardware resource utilization and energy efficiency compared to traditional CNN and baseline AdderNet implementations on FPGA platforms.

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2024addernet2,
  title={AdderNet 2.0: Optimal FPGA Acceleration of AdderNet with Activation-Oriented Quantization and Fused Bias Removal based Memory Optimization},
  author={Zhang, Yunxiang and Al Kailani, Omar and Zhao, Wenfeng},
  booktitle={Design Automation Conference (DAC)},
  year={2024}
}
```

📚 References

Original AdderNet
Chen, H., Wang, Y., Xu, C., Shi, B., Xu, C., Tian, Q., & Xu, C. (2020). AdderNet: Do We Really Need Multiplications in Deep Learning? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Paper | Code

AdderNet 2.0
Zhang, Y., Al Kailani, O., & Zhao, W. (2024). AdderNet 2.0: Optimal FPGA Acceleration of AdderNet with Activation-Oriented Quantization and Fused Bias Removal based Memory Optimization. In Proceedings of the Design Automation Conference (DAC).

Non-Negative AdderNet
Zhang, Y., Ahmed, S., Almalky, A. M. A., Rakin, A. S., & Zhao, W. Non-Negative AdderNet: Algorithm-Hardware Co-design for Lightweight Defense of Adversarial Bit-Flip Attacks.

**Note**: This implementation focuses on ResNet-20 for CIFAR-10. ResNet-50 for ImageNet is also supported with similar architecture modifications.
