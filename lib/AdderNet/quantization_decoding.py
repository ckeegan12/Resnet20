import numpy as np
import torch

class Quant_decode:
  def dequant(scale: float, quant_tensor, original_tensor):
    """
    input: scale = scaling facotr for quantization, quant_tensor = quantized array in int8, original_tensor = fp32 original tensor as array
    output: fp32_tensor = scaled up array from int 8 to fp32, max_error = quantization error
    """
    original_shape = quant_tensor.shape

    quant_array = quant_tensor.detach().cpu().numpy().flatten().astype(np.float32)
    original_array = original_tensor.detach().cpu().numpy().flatten().astype(np.float32)

    scaled_array = quant_array * scale
    fp32_tensor = torch.from_numpy(scaled_array.reshape(original_shape))
    quant_error = scaled_array - original_array
    max_error = np.max(np.abs(quant_error))

    return fp32_tensor, max_error