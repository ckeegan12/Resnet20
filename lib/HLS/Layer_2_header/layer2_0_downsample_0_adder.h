#ifndef LAYER2_0_DOWN_SAMPLE_0_ADDER_H
#define LAYER2_0_DOWN_SAMPLE_0_ADDER_H

#include "../parameters.h"

// Layer 2 downsample 1x1 convolution weights
// Used for stride-2 downsampling from 16ch to 32ch
// Dimensions: [4 slices][8 output channels per slice][16 input channels]
// Total: 4 * 8 = 32 output channels

const weight_t layer2_0_downsample_0_adder[4][8][16] = {
    // Placeholder - populate with export_weights_hls.py
    // Slice 0: outputs 0-7
    {{0}},
    // Slice 1: outputs 8-15
    {{0}},
    // Slice 2: outputs 16-23
    {{0}},
    // Slice 3: outputs 24-31
    {{0}}
};

#endif // LAYER2_0_DOWN_SAMPLE_0_ADDER_H