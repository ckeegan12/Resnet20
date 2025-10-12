import numpy as np
import torch

class Quant:
  @staticmethod
  # Clamping function
  def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
      """
      input: params_q = unquantizied parameter, lower_bound = -(2^(n-1) - 1), upper_bound = 2^(n-1)
      output: clamped parameter
      """
      params_q[params_q < lower_bound] = lower_bound
      params_q[params_q > upper_bound] = upper_bound
      return params_q

  @staticmethod
  # symmetric int quantization
  def symmetric_quantization(weight_tensor, bits: int):
      """
      input: params = tensor, bits = bit size
      output: quantized = int8, scale = scaling factor for quantization
      """
      # Reshape tensor
      original_shape = weight_tensor.shape
      params = weight_tensor.cpu().numpy().flatten()

      # Calculate the scale
      alpha = np.max(np.abs(params))
      scale = alpha / (2**(bits-1)-1)
      lower_bound = -2**(bits-1)
      upper_bound = 2**(bits-1)-1
      # Quantize the parameters
      quantized = Quant.clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
      quantized_tensor = quantized.reshape(original_shape)
      quantized_tensor = torch.from_numpy(quantized_tensor)

      return quantized_tensor, scale
