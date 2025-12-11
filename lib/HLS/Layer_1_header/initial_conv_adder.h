#ifndef INITIAL_CONV_ADDER_H
#define INITIAL_CONV_ADDER_H

#include "../parameters.h"

// Initial convolution weights for AdderNet20
// Dimensions: [out_channels][in_channels][kernel_h][kernel_w]
// Shape: 16 x 16 x 3 x 3 (padded input channels from 3 to 16)
// Note: First 3 input channels contain actual weights, rest are zeros

const weight_t initial_conv_adder[16][16][3][3] = {
    // Placeholder - populate with export_weights_hls.py
    // Each value should be int5: range [-16, 15] mapped to ap_uint<5>
};

#endif // INITIAL_CONV_ADDER_H
