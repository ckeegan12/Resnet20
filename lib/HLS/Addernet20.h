#ifndef ADDERNET20_H
#define ADDERNET20_H

#include "parameters.h"

// ----------------- Main Kernel Function -----------------
extern "C" void ADDERNET20_2_0(
    const input_t* IFM_flat,   // Input: flattened 32x32x3 image
    const weight_t* weights,   // (Not used - weights in headers)
    result_t* OFM              // Output: 10 class scores
);

// ----------------- Reshape Helper Functions -----------------
// Layer 2 reshape functions
void lay2_reshape(input_t IFM1[16][H2][W2], input_t OFM[CH2][H2][W2], int slice);
void lay2_reshape_down(input_t IFM1[CH1][H1][W1], input_t OFM[CH2][H2][W2], int slice);

// Layer 3 reshape functions
void lay3_reshape(input_t IFM1[16][H3][W3], input_t OFM[CH3][H3][W3], int slice);
void lay3_reshape_down(input_t IFM1[CH1][H2][W2], input_t OFM[CH3][H3][W3], int slice);

#endif // ADDERNET20_H