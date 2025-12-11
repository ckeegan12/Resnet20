#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <ap_fixed.h>

// ----------------- Data Types -----------------
typedef ap_fixed<16,8> input_t;   // 16-bit fixed point, 8 integer bits
typedef ap_uint<5> weight_t;   // 5-bit unsigned for weights
typedef ap_fixed<32,16> result_t; // 32-bit accumulator

// ----------------- Network Dimensions -----------------
// Input image: 32x32x3 (CIFAR-10)
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_CH 3

// Layer 1: 16 channels, 32x32 (no downsampling)
#define CH1 16
#define H1 32
#define W1 32

// Layer 2: 32 channels, 16x16 (stride-2 downsample)
#define CH2 32
#define H2 16
#define W2 16

// Layer 3: 64 channels, 8x8 (stride-2 downsample)
#define CH3 64
#define H3 8
#define W3 8

// Output classes (CIFAR-10)
#define NUM_CLASSES 10

// ----------------- Helper Constants -----------------
#define EPSILON 1e-5

#endif // PARAMETERS_H