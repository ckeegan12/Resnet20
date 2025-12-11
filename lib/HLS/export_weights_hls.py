"""
Export AdderNet 2.0 weights to C headers for Vitis HLS
Exports as UNSIGNED int5 (range 0-31) for ap_uint<5>
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AdderNet'))
from quantization_encoder import Quant

MODEL_PATH = "../AdderNet/AdderNet_model.pth"
HLS_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_BITS = 5

def quantize_weights_unsigned(tensor, bits=5):
    """Apply symmetric int5 quantization, then convert to unsigned by adding offset."""
    quantized, scale = Quant.symmetric_quantization(tensor, bits)
    # Signed int5 range: [-16, 15], offset by 16 to get unsigned [0, 31]
    unsigned_quantized = quantized.numpy() + 16
    # Clamp to valid range just in case
    unsigned_quantized = np.clip(unsigned_quantized, 0, 31).astype(np.int32)
    return unsigned_quantized, scale

def format_array_1d(arr, name, dtype="input_t"):
    values = ", ".join([f"{v}" for v in arr.flatten()])
    return f"const {dtype} {name}[{len(arr)}] = {{\n    {values}\n}};"

def format_array_4d(arr, name, dtype="weight_t"):
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

def write_header(filepath, guard_name, content):
    header = f"#ifndef {guard_name}\n#define {guard_name}\n\n"
    header += '#include "../parameters.h"\n\n'
    header += content + "\n\n"
    header += f"#endif // {guard_name}\n"
    with open(filepath, 'w') as f:
        f.write(header)
    print(f"  Written: {os.path.basename(filepath)}")

def get_key(state_dict, key):
    """Try both with and without 'module.' prefix."""
    if key in state_dict:
        return state_dict[key]
    if f"module.{key}" in state_dict:
        return state_dict[f"module.{key}"]
    return None

def main():
    print("=" * 60)
    print("AdderNet 2.0 Weight Export - UNSIGNED INT5")
    print("Output range: [0, 31] for ap_uint<5>")
    print("=" * 60)
    
    model_path = os.path.join(HLS_DIR, MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    print(f"\n[1] Loading model from: {MODEL_PATH}")
    state_dict = torch.load(model_path, map_location='cpu')
    print(f"    Loaded {len(state_dict)} parameters")
    
    # Export initial convolution (conv1)
    print("\n[2] Exporting initial convolution weights (UNSIGNED)...")
    conv1_weight = get_key(state_dict, "conv1.weight")
    if conv1_weight is not None:
        padded = torch.zeros(16, 16, 3, 3)
        padded[:, :3, :, :] = conv1_weight
        quantized, scale = quantize_weights_unsigned(padded, QUANT_BITS)
        print(f"    conv1: {conv1_weight.shape} -> unsigned [0,31], scale={scale:.6f}")
        print(f"    Sample values: min={quantized.min()}, max={quantized.max()}")
        
        content = f"// Initial conv weights - UNSIGNED int5 [0,31], scale: {scale:.6f}\n"
        content += f"// Offset: add 16 to signed values to get unsigned\n\n"
        content += format_array_4d(quantized, "initial_conv_adder")
        write_header(
            os.path.join(HLS_DIR, "Layer_1_header", "initial_conv_adder.h"),
            "INITIAL_CONV_ADDER_H", content
        )
    else:
        print("    WARNING: conv1.weight not found!")

    # Export Layer 1 weights
    print("\n[3] Exporting Layer 1 weights (UNSIGNED)...")
    layer1_weights = [
        ("layer1.0.conv1", "layer1_0_conv1_adder"),
        ("layer1.0.conv2", "layer1_0_conv2_adder"),
        ("layer1.1.conv1", "layer1_1_conv1_adder"),
        ("layer1.1.conv2", "layer1_1_conv2_adder"),
        ("layer1.2.conv1", "layer1_2_conv1_adder"),
        ("layer1.2.conv2", "layer1_2_conv2_adder"),
    ]
    
    for layer_key, header_name in layer1_weights:
        weight = get_key(state_dict, f"{layer_key}.adder")
        if weight is not None:
            quantized, scale = quantize_weights_unsigned(weight, QUANT_BITS)
            print(f"    {layer_key}: {weight.shape}, range=[{quantized.min()},{quantized.max()}]")
            
            content = f"// {layer_key} - UNSIGNED int5 [0,31], scale: {scale:.6f}\n\n"
            content += format_array_4d(quantized, header_name)
            write_header(
                os.path.join(HLS_DIR, "Layer_1_header", f"{header_name}.h"),
                f"{header_name.upper()}_H", content
            )
        else:
            print(f"    WARNING: {layer_key}.adder not found!")

    # Export Layer 2 weights
    print("\n[4] Exporting Layer 2 weights (UNSIGNED)...")
    layer2_weights = [
        ("layer2.0.downsample.0", "layer2_0_downsample_0_adder"),
        ("layer2.0.conv1", "layer2_0_conv1_adder"),
        ("layer2.0.conv2", "layer2_0_conv2_adder"),
        ("layer2.1.conv1", "layer2_1_conv1_adder"),
        ("layer2.1.conv2", "layer2_1_conv2_adder"),
        ("layer2.2.conv1", "layer2_2_conv1_adder"),
        ("layer2.2.conv2", "layer2_2_conv2_adder"),
    ]
    
    for layer_key, header_name in layer2_weights:
        weight = get_key(state_dict, f"{layer_key}.adder")
        if weight is not None:
            quantized, scale = quantize_weights_unsigned(weight, QUANT_BITS)
            print(f"    {layer_key}: {weight.shape}, range=[{quantized.min()},{quantized.max()}]")
            
            content = f"// {layer_key} - UNSIGNED int5 [0,31], scale: {scale:.6f}\n\n"
            content += format_array_4d(quantized, header_name)
            write_header(
                os.path.join(HLS_DIR, "Layer_2_header", f"{header_name}.h"),
                f"{header_name.upper()}_H", content
            )
        else:
            print(f"    WARNING: {layer_key}.adder not found!")

    # Export Layer 3 weights
    print("\n[5] Exporting Layer 3 weights (UNSIGNED)...")
    layer3_weights = [
        ("layer3.0.downsample.0", "layer3_0_downsample_0_adder"),
        ("layer3.0.conv1", "layer3_0_conv1_adder"),
        ("layer3.0.conv2", "layer3_0_conv2_adder"),
        ("layer3.1.conv1", "layer3_1_conv1_adder"),
        ("layer3.1.conv2", "layer3_1_conv2_adder"),
        ("layer3.2.conv1", "layer3_2_conv1_adder"),
        ("layer3.2.conv2", "layer3_2_conv2_adder"),
    ]
    
    for layer_key, header_name in layer3_weights:
        weight = get_key(state_dict, f"{layer_key}.adder")
        if weight is not None:
            quantized, scale = quantize_weights_unsigned(weight, QUANT_BITS)
            print(f"    {layer_key}: {weight.shape}, range=[{quantized.min()},{quantized.max()}]")
            
            content = f"// {layer_key} - UNSIGNED int5 [0,31], scale: {scale:.6f}\n\n"
            content += format_array_4d(quantized, header_name)
            write_header(
                os.path.join(HLS_DIR, "Layer_3_header", f"{header_name}.h"),
                f"{header_name.upper()}_H", content
            )
        else:
            print(f"    WARNING: {layer_key}.adder not found!")

    # Export BatchNorm1 parameters (these stay as input_t, not unsigned)
    print("\n[6] Exporting BatchNorm parameters...")
    for param, out_name in [("weight", "gamma"), ("bias", "beta"), ("running_mean", "mean"), ("running_var", "var")]:
        val = get_key(state_dict, f"bn1.{param}")
        if val is not None:
            content = f"// BatchNorm1 {out_name} - 16 channels\n\n"
            content += format_array_1d(val.numpy(), f"bn1_{out_name}")
            write_header(
                os.path.join(HLS_DIR, "Batchnorm", f"bn1_{out_name}.h"),
                f"BN1_{out_name.upper()}_H", content
            )
            print(f"    bn1.{param}: {val.shape}")

    # Export FC weights
    print("\n[7] Exporting FC layer weights...")
    fc_weight = get_key(state_dict, "fc.weight")
    if fc_weight is not None:
        fc_squeezed = fc_weight.squeeze()
        fc_flat = fc_squeezed.flatten().numpy()
        print(f"    fc.weight: {fc_weight.shape} -> flattened {len(fc_flat)}")
        content = f"// FC weights - 10 classes x 64 channels = 640\n\n"
        content += format_array_1d(fc_flat, "fc_weight")
        write_header(
            os.path.join(HLS_DIR, "Batchnorm", "fc_weight.h"),
            "FC_WEIGHT_H", content
        )

    # Export BN2 parameters
    for param, out_name in [("weight", "weight"), ("bias", "bias"), ("running_mean", "mean"), ("running_var", "var")]:
        val = get_key(state_dict, f"bn2.{param}")
        if val is not None:
            content = f"// BatchNorm2 {out_name} - 10 classes\n\n"
            content += format_array_1d(val.numpy(), f"bn2_{out_name}")
            write_header(
                os.path.join(HLS_DIR, "Batchnorm", f"bn2_{out_name}.h"),
                f"BN2_{out_name.upper()}_H", content
            )
            print(f"    bn2.{param}: {val.shape}")

    print("\n" + "=" * 60)
    print("Export complete! All weights are UNSIGNED [0,31]")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
