#ifndef LAYER2_0_CONV2_ADDER_H
#define LAYER2_0_CONV2_ADDER_H

#include "../parameters.h"

// Layer 2 Block 0 Conv2 weights
// 3x3 convolution with 2 slices, 16 output channels each
// Input: 32ch, Output: 32ch
// Dimensions: [2 slices][16 out channels][32 in channels][3][3]

const weight_t layer2_0_conv2_adder[2][16][32][3][3] = {
    // Placeholder - populate with export_weights_hls.py
    // Slice 0: outputs 0-15
    {{{{0}}}},
    // Slice 1: outputs 16-31
    {{{{0}}}}
};

#endif // LAYER2_0_CONV2_ADDER_H