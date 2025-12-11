#ifndef LAYER3_0_DOWNSAMPLE_0_ADDER_H
#define LAYER3_0_DOWNSAMPLE_0_ADDER_H

#include "../parameters.h"

// Layer 3 downsample 1x1 convolution weights
// Used for stride-2 downsampling from 32ch to 64ch
// Dimensions: [16 slices][4 output channels per slice][32 input channels]
// Total: 16 * 4 = 64 output channels

const weight_t layer3_0_downsample_0_adder[16][4][32] = {
    // Placeholder - populate with export_weights_hls.py
    {{0}}, {{0}}, {{0}}, {{0}},
    {{0}}, {{0}}, {{0}}, {{0}},
    {{0}}, {{0}}, {{0}}, {{0}},
    {{0}}, {{0}}, {{0}}, {{0}}
};

#endif // LAYER3_0_DOWNSAMPLE_0_ADDER_H
