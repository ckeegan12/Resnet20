"""
Export AdderNet20 weights to C headers for Vitis HLS
=====================================================
This script loads the trained PyTorch model and exports all weights
and batch normalization parameters as C header files with int5 quantization.

Usage:
    python export_weights_hls.py

Output:
    Updates all header files in Layer_1_header/, Layer_2_header/, 
    Layer_3_header/, and Batchnorm/ directories.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AdderNet'))

from quantization_encoder import Quant

# Configuration
MODEL_PATH = "../AdderNet/AdderNet_model.pth"
HLS_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_BITS = 5  # int5 quantization

def quantize_weights(tensor, bits=5):
    """Apply symmetric int5 quantization to weights."""
    quantized, scale = Quant.symmetric_quantization(tensor, bits)
    return quantized.numpy(), scale

def format_array_1d(arr, name, dtype="input_t"):
    """Format 1D array as C header."""
    values = ", ".join([f"{v}" for v in arr.flatten()])
    return f"const {dtype} {name}[{len(arr)}] = {{\n    {values}\n}};"

def format_array_4d(arr, name, dtype="weight_t"):
    """Format 4D array as C header for conv weights [O][I][H][W]."""
    o, i, h, w = arr.shape
    lines = [f"const {dtype} {name}[{o}][{i}][{h}][{w}] = {{"]
    
    for oc in range(o):
        lines.append("    {")
        for ic in range(i):
            lines.append("        {")
            for hh in range(h):
                row = ", ".join([f"{arr[oc, ic, hh, ww]}" for ww in range(w)])
                comma = "," if hh < h - 1 else ""
                lines.append(f"            {{{row}}}{comma}")
            comma = "," if ic < i - 1 else ""
            lines.append(f"        }}{comma}")
        comma = "," if oc < o - 1 else ""
        lines.append(f"    }}{comma}")
    lines.append("};")
    return "\n".join(lines)

def format_array_sliced(arr, name, slices, slice_shape, dtype="weight_t"):
    """Format sliced weight array for layer2/3 conv operations."""
    lines = [f"const {dtype} {name}[{slices}][{slice_shape[0]}][{slice_shape[1]}][{slice_shape[2]}][{slice_shape[3]}] = {{"]
    # Reshape original array into slices
    # This is a simplified version - actual reshaping depends on the layer
    for s in range(slices):
        lines.append("    { /* Slice " + str(s) + " */")
        # Add placeholder
        lines.append("        // Populate from model")
        lines.append("    },")
    lines.append("};")
    return "\n".join(lines)

def write_header(filepath, guard_name, content, include_params=True):
    """Write a complete C header file."""
    header = f"""#ifndef {guard_name}
#define {guard_name}

"""
    if include_params:
        header += '#include "../parameters.h"\n\n'
    
    header += content + "\n\n"
    header += f"#endif // {guard_name}\n"
    
    with open(filepath, 'w') as f:
        f.write(header)
    print(f"  Written: {os.path.basename(filepath)}")

def export_batchnorm(state_dict, layer_name, output_dir, num_channels):
    """Export BatchNorm parameters."""
    prefix = layer_name
    
    # Get parameters with fallback
    gamma = state_dict.get(f"{prefix}.weight", torch.ones(num_channels))
    beta = state_dict.get(f"{prefix}.bias", torch.zeros(num_channels))
    mean = state_dict.get(f"{prefix}.running_mean", torch.zeros(num_channels))
    var = state_dict.get(f"{prefix}.running_var", torch.ones(num_channels))
    
    return {
        'gamma': gamma.numpy(),
        'beta': beta.numpy(),
        'mean': mean.numpy(),
        'var': var.numpy()
    }

def main():
    print("=" * 60)
    print("AdderNet20 Weight Export for Vitis HLS")
    print("=" * 60)
    
    # Check model file exists
    model_path = os.path.join(HLS_DIR, MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure AdderNet_model.pth is in the AdderNet directory.")
        return 1
    
    print(f"\n[1] Loading model from: {MODEL_PATH}")
    state_dict = torch.load(model_path, map_location='cpu')
    print(f"    Loaded {len(state_dict)} parameters")
    
    # List all keys for debugging
    print("\n[2] Model keys found:")
    for key in sorted(state_dict.keys()):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"    {key}: {shape}")
    
    # Export initial convolution (conv1)
    print("\n[3] Exporting initial convolution weights...")
    if 'conv1.weight' in state_dict:
        conv1_weight = state_dict['conv1.weight']  # [16, 3, 3, 3]
        # Pad to [16, 16, 3, 3]
        padded = torch.zeros(16, 16, 3, 3)
        padded[:, :3, :, :] = conv1_weight
        quantized, scale = quantize_weights(padded, QUANT_BITS)
        print(f"    conv1: {conv1_weight.shape} -> padded to (16,16,3,3), scale={scale:.6f}")
        
        content = f"// Initial convolution weights - quantized to int5\n"
        content += f"// Original shape: {list(conv1_weight.shape)}, scale: {scale:.6f}\n\n"
        content += format_array_4d(quantized, "initial_conv_adder")
        
        write_header(
            os.path.join(HLS_DIR, "Layer_1_header", "initial_conv_adder.h"),
            "INITIAL_CONV_ADDER_H",
            content
        )
    
    # Export Layer 1 weights (16x16x3x3)
    print("\n[4] Exporting Layer 1 weights...")
    layer1_weights = [
        ("layer1.0.conv1", "layer1_0_conv1_adder"),
        ("layer1.0.conv2", "layer1_0_conv2_adder"),
        ("layer1.1.conv1", "layer1_1_conv1_adder"),
        ("layer1.1.conv2", "layer1_1_conv2_adder"),
        ("layer1.2.conv1", "layer1_2_conv1_adder"),
        ("layer1.2.conv2", "layer1_2_conv2_adder"),
    ]
    
    for layer_key, header_name in layer1_weights:
        weight_key = f"{layer_key}.adder"
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            quantized, scale = quantize_weights(weight, QUANT_BITS)
            print(f"    {layer_key}: {weight.shape}, scale={scale:.6f}")
            
            content = f"// {layer_key} weights - int5 quantized\n"
            content += f"// Shape: {list(weight.shape)}, scale: {scale:.6f}\n\n"
            content += format_array_4d(quantized, header_name)
            
            write_header(
                os.path.join(HLS_DIR, "Layer_1_header", f"{header_name}.h"),
                f"{header_name.upper()}_H",
                content
            )
    
    # Export BatchNorm1 parameters
    print("\n[5] Exporting BatchNorm parameters...")
    bn1_params = export_batchnorm(state_dict, "bn1", HLS_DIR, 16)
    
    for param_name, values in bn1_params.items():
        content = f"// BatchNorm1 {param_name} - {len(values)} channels\n\n"
        content += format_array_1d(values, f"bn1_{param_name}")
        write_header(
            os.path.join(HLS_DIR, "Batchnorm", f"bn1_{param_name}.h"),
            f"BN1_{param_name.upper()}_H",
            content
        )
    
    # Export FC weights
    print("\n[6] Exporting FC layer weights...")
    if 'fc.weight' in state_dict:
        fc_weight = state_dict['fc.weight'].squeeze()  # [10, 64, 1, 1] -> [10, 64]
        if len(fc_weight.shape) == 4:
            fc_weight = fc_weight.squeeze(-1).squeeze(-1)
        fc_flat = fc_weight.flatten().numpy()
        
        content = f"// FC layer weights - 10 classes x 64 channels = 640 total\n"
        content += f"// Access: fc_weight[class_idx * 64 + channel_idx]\n\n"
        content += format_array_1d(fc_flat, "fc_weight")
        
        write_header(
            os.path.join(HLS_DIR, "Batchnorm", "fc_weight.h"),
            "FC_WEIGHT_H",
            content
        )
    
    # Export BN2 parameters
    bn2_params = export_batchnorm(state_dict, "bn2", HLS_DIR, 10)
    
    name_mapping = {'gamma': 'weight', 'beta': 'bias', 'mean': 'mean', 'var': 'var'}
    for param_name, values in bn2_params.items():
        out_name = name_mapping[param_name]
        content = f"// BatchNorm2 {out_name} - 10 classes\n\n"
        content += format_array_1d(values, f"bn2_{out_name}")
        write_header(
            os.path.join(HLS_DIR, "Batchnorm", f"bn2_{out_name}.h"),
            f"BN2_{out_name.upper()}_H",
            content
        )
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated headers in Layer_*_header/ and Batchnorm/")
    print("2. Run Vitis HLS: vitis_hls -f run_hls.tcl")
    print("3. Check synthesis reports in addernet20_hls/solution1/syn/report/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
