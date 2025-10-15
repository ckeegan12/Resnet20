import numpy as np

class Quant_decode:
  def dequant(scale: float, quant_tensor, original_tensor):
    """
    input: scale = scaling facotr for quantization, quant_tensor = quantized array in int8, original_tensor = fp32 original tensor as array
    output: fp32_tensor = scaled up array from int 8 to fp32, max_error = quantization error
    """
    
    fp32_tensor = quant_tensor * scale

    fp32_array = fp32_tensor.cpu().numpy().flatten()
    original_array = original_tensor.cpu().numpy().flatten()

    quant_error = fp32_array - original_array
    max_error = np.max(np.abs(quant_error))

    return fp32_tensor, max_error