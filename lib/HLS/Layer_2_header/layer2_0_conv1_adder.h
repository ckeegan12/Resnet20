#ifndef LAYER2_0_CONV1_ADDER_H
#define LAYER2_0_CONV1_ADDER_H

#include "../parameters.h"

// Layer 2 Block 0 Conv1 weights
// Used for 3x3 convolution with 2 slices
// Dimensions: [2 slices][16 out channels][16 in channels][3][3]
// Total: 2 * 16 = 32 output channels

const weight_t layer2_0_conv1_adder[2][16][16][3][3] = {
    // Placeholder - populate with export_weights_hls.py
    // Slice 0: outputs 0-15
    {{{{0}}}},
    // Slice 1: outputs 16-31
    {{{{0}}}}
};

#endif // LAYER2_0_CONV1_ADDER_H