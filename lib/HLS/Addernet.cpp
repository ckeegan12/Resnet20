// ADDERNET20_2_0.h - Header file
#ifndef ADDERNET20_2_0_H
#define ADDERNET20_2_0_H

#include "parameters.h"

void ADDERNET20_2_0(input_t* IFM, input_t* weight, result_t* OFM);

#endif

// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "ap_fixed.h"
#include "ap_int.h"

// Fixed-point types for AdderNet 2.0 with FBR
typedef ap_fixed<16, 8> input_t;     // change depending on input and activation bit
typedef ap_fixed<32, 16> result_t;   // Intermediate results
typedef ap_int<6> weight_t;          // Quantized weights: 6-bit signed integer for AdderNet2.0

#endif

// ADDERNET20_2_0.cpp - Main implementation
#include <iostream>
#include "parameters.h"
#include "ADDERNET20_2_0.h"

using namespace std;

// Weight includes for AdderNet 2.0 (quantized integer weights)
#include "Layer_1_header/layer1_0_conv1_adder.h"
#include "Layer_1_header/layer1_0_conv2_adder.h"
#include "Layer_1_header/layer1_1_conv1_adder.h"
#include "Layer_1_header/layer1_1_conv2_adder.h"
#include "Layer_1_header/layer1_2_conv1_adder.h"
#include "Layer_1_header/layer1_2_conv2_adder.h"

#include "weights/layer2_0_downsample_0_adder.h"
#include "weights/layer2_0_conv1_adder.h"
#include "weights/layer2_0_conv2_adder.h"
#include "weights/layer2_1_conv1_adder.h"
#include "weights/layer2_1_conv2_adder.h"
#include "weights/layer2_2_conv1_adder.h"
#include "weights/layer2_2_conv2_adder.h"

#include "weights/layer3_0_downsample_0_adder.h"
#include "weights/layer3_0_conv1_adder.h"
#include "weights/layer3_0_conv2_adder.h"
#include "weights/layer3_1_conv1_adder.h"
#include "weights/layer3_1_conv2_adder.h"
#include "weights/layer3_2_conv1_adder.h"
#include "weights/layer3_2_conv2_adder.h"

// BatchNorm parameters (adjusted with FBR)
#include "weights/bn1_gamma.h"
#include "weights/bn1_beta.h"
#include "weights/bn1_mean.h"  // Already adjusted: μ' = round(μ/δ) + Σ|W_bias|
#include "weights/bn1_var.h"

#include "weights/fc_weight.h"
#include "weights/bn2_weight.h"
#include "weights/bn2_bias.h"

// Global buffers
input_t ADD0[16][32][32];
input_t p1[16][32][32];
input_t p2[16][32][32];
input_t padded[16][34][34];

input_t ADD1[32][16][16];
input_t OFM_out_1[32][16][16];
input_t OFM_out_2[32][16][16];
input_t padded_1[32][18][18];
input_t part_ofm[16][16][16];

input_t ADD2[64][8][8];
input_t OFM_out_3[64][8][8];
input_t OFM_out_4[64][8][8];
input_t padded_2[64][10][10];
input_t part_ofm2[16][8][8];

weight_t WBUF3x3[16][16][3][3];
weight_t WBUF3x3_2[16][32][3][3];
weight_t WBUF3x3_3[16][64][3][3];
weight_t WBUF1x1[8][16];
weight_t WBUF1x1_1[4][32];

input_t AV[64];
input_t FC1[10];

// ========== Core AdderNet 2.0 Operation: -Σ|W - X| ==========
input_t ADDER_OP(weight_t W, input_t X) {
    #pragma HLS INLINE
    result_t diff = W - X;
    result_t abs_diff = (diff >= 0) ? diff : -diff;
    return -abs_diff;  
}

// Optimized ADDER operation for 4 parallel computations
input_t ADDER_4(weight_t W0, input_t X0, weight_t W1, input_t X1,
                weight_t W2, input_t X2, weight_t W3, input_t X3) {
    #pragma HLS INLINE off
    result_t diff0 = W0 - X0;
    result_t diff1 = W1 - X1;
    result_t diff2 = W2 - X2;
    result_t diff3 = W3 - X3;
    
    result_t abs0 = (diff0 >= 0) ? diff0 : -diff0;
    result_t abs1 = (diff1 >= 0) ? diff1 : -diff1;
    result_t abs2 = (diff2 >= 0) ? diff2 : -diff2;
    result_t abs3 = (diff3 >= 0) ? diff3 : -diff3;
    
    result_t sum = abs0 + abs1 + abs2 + abs3;
    return -sum;
}

// ========== Reshape and Padding Functions ==========
void lay2_reshape(input_t IFM1[16][16][16], input_t OFM[32][16][16], int c) {
    for(int j=0; j<16; j++) {
        for(int k=0; k<16; k++) {
            for (int i=0; i<16; i++) {
                OFM[i+c*16][j][k] = IFM1[i][j][k];
            }
        }
    }
}

void lay2_reshape_down(input_t IFM1[16][32][32], input_t OFM[32][16][16], int c) {
    for(int j=0; j<16; j++) {
        for(int k=0; k<16; k++) {
            for (int i=0; i<16; i++) {
                OFM[i+c*16][j][k] = IFM1[i][j*2][k*2];
            }
        }
    }
}

void lay3_reshape(input_t IFM1[16][8][8], input_t OFM[64][8][8], int c) {
    for(int j=0; j<8; j++) {
        for(int k=0; k<8; k++) {
            for (int i=0; i<16; i++) {
                OFM[i+c*16][j][k] = IFM1[i][j][k];
            }
        }
    }
}

void lay3_reshape_down(input_t IFM1[32][16][16], input_t OFM[64][8][8], int c) {
    for(int j=0; j<8; j++) {
        for(int k=0; k<8; k++) {
            for (int i=0; i<32; i++) {
                OFM[i+c*32][j][k] = IFM1[i][j*2][k*2];
            }
        }
    }
}

void padding_0(input_t D[16][32][32], input_t IFM[16][34][34]) {
    for(int j=0; j<32; j++) {
        for(int k=0; k<32; k++) {
            for(int i=0; i<16; i++) {
                IFM[i][j+1][k+1] = D[i][j][k];
            }
        }
    }
}

void padding_1(input_t D[32][16][16], input_t IFM[32][18][18]) {
    for(int j=0; j<16; j++) {
        for(int k=0; k<16; k++) {
            for(int i=0; i<32; i++) {
                IFM[i][j+1][k+1] = D[i][j][k];
            }
        }
    }
}

void padding_2(input_t D[64][8][8], input_t IFM[64][10][10]) {
    for(int j=0; j<8; j++) {
        for(int k=0; k<8; k++) {
            for(int i=0; i<64; i++) {
                IFM[i][j+1][k+1] = D[i][j][k];
            }
        }
    }
}

void IFM_reshape(input_t* IFM, input_t OFM[16][32][32]) {
    for(int i=0; i<3; i++) {
        for(int j=0; j<32; j++) {
            for(int k=0; k<32; k++) {
                OFM[i][j][k] = IFM[i*1024+j*32+k];
            }
        }
    }
}

void OFM_reshape(input_t IFM[16][32][32], input_t *OFM) {
    for(int i=0; i<16; i++) {
        for(int j=0; j<32; j++) {
            for(int k=0; k<32; k++) {
                OFM[i*32*32+j*32+k] = IFM[i][j][k];
            }
        }
    }
}

// ========== Weight Reshape Functions ==========
void wreshape_0(weight_t* w, weight_t WBUF[16][16][3][3]) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<3; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[i*3*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_01(weight_t* w, weight_t WBUF[16][16][3][3]) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<16; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[i*16*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_20(weight_t* w, weight_t WBUF[16][16][3][3], int c) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<16; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[2304*c+i*16*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_21(weight_t* w, weight_t WBUF[16][32][3][3], int c) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<32; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[4608*c+i*32*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_30(weight_t* w, weight_t WBUF[16][32][3][3], int c) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<32; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[4608*c+i*32*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_31(weight_t* w, weight_t WBUF[16][64][3][3], int c) {
    for(int k=0; k<3; k++) {
        for(int l=0; l<3; l++) {
            for(int j=0; j<64; j++) {
                for(int i=0; i<16; i++) {
                    WBUF[i][j][k][l] = w[9216*c+i*64*3*3+j*3*3+k*3+l];
                }
            }
        }
    }
}

void wreshape_downsample1(weight_t* w, weight_t WBUF0[8][16], int c) {
    for(int j=0; j<16; j++) {
        for(int i=0; i<8; i++) {
            WBUF0[i][j] = w[c*128+i*16+j];
        }
    }
}

void wreshape_downsample2(weight_t* w, weight_t WBUF0[4][32], int c) {
    for(int j=0; j<32; j++) {
        for(int i=0; i<4; i++) {
            WBUF0[i][j] = w[c*128+i*32+j];
        }
    }
}

// ========== AdderNet Processing Elements ==========
// PE0: 16x16 channels, AdderNet operation with FBR
void PE0_16_16_adder(input_t IFM[16][34][34], weight_t WBUF[16][16][3][3], 
                     input_t OFM[16][32][32]) {
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < 32; o_row++) {
                for (int o_col = 0; o_col < 32; o_col++) {
                    #pragma HLS pipeline
                    for (int o_ch = 0; o_ch < 16; o_ch++) {
                        #pragma HLS unroll
                        for (int i_ch = 0; i_ch < 16; i_ch += 4) {
                            #pragma HLS unroll
                            // AdderNet 2.0 operation: -Σ|W - X|
                            OFM[o_ch][o_row][o_col] += ADDER_4(
                                WBUF[o_ch][i_ch][ii][jj], IFM[i_ch][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+1][ii][jj], IFM[i_ch+1][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+2][ii][jj], IFM[i_ch+2][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+3][ii][jj], IFM[i_ch+3][o_row+ii][o_col+jj]
                            );
                        }
                    }
                }
            }
        }
    }
}

// PE1: 16x32 channels, AdderNet operation
void PE1_16_32_adder(input_t IFM[32][18][18], weight_t WBUF[16][32][3][3], 
                     input_t OFM[16][16][16]) {
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < 16; o_row++) {
                for (int o_col = 0; o_col < 16; o_col++) {
                    #pragma HLS pipeline
                    for (int o_ch = 0; o_ch < 16; o_ch++) {
                        #pragma HLS unroll
                        for (int i_ch = 0; i_ch < 32; i_ch += 4) {
                            #pragma HLS unroll
                            OFM[o_ch][o_row][o_col] += ADDER_4(
                                WBUF[o_ch][i_ch][ii][jj], IFM[i_ch][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+1][ii][jj], IFM[i_ch+1][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+2][ii][jj], IFM[i_ch+2][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+3][ii][jj], IFM[i_ch+3][o_row+ii][o_col+jj]
                            );
                        }
                    }
                }
            }
        }
    }
}

// PE2: 16x64 channels, AdderNet operation
void PE2_16_64_adder(input_t IFM[64][10][10], weight_t WBUF[16][64][3][3], 
                     input_t OFM[16][8][8]) {
    #pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=1
    #pragma HLS ARRAY_PARTITION variable=WBUF complete dim=2
    #pragma HLS ARRAY_PARTITION variable=OFM complete dim=1

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < 8; o_row++) {
                for (int o_col = 0; o_col < 8; o_col++) {
                    #pragma HLS pipeline
                    for (int o_ch = 0; o_ch < 16; o_ch++) {
                        #pragma HLS unroll
                        for (int i_ch = 0; i_ch < 64; i_ch += 4) {
                            #pragma HLS unroll
                            OFM[o_ch][o_row][o_col] += ADDER_4(
                                WBUF[o_ch][i_ch][ii][jj], IFM[i_ch][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+1][ii][jj], IFM[i_ch+1][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+2][ii][jj], IFM[i_ch+2][o_row+ii][o_col+jj],
                                WBUF[o_ch][i_ch+3][ii][jj], IFM[i_ch+3][o_row+ii][o_col+jj]
                            );
                        }
                    }
                }
            }
        }
    }
}

// ========== 1x1 Convolution for Downsampling ==========
void conv1x1_adder(input_t IFM[16][32][32], weight_t WEIGHT[8][16], 
                   result_t OFM[8][16][16]) {
    for(int k=0; k<16; k++) {
        for(int l=0; l<16; l++) {
            for(int i=0; i<8; i++) {
                for(int j=0; j<16; j++) {
                    // AdderNet operation for 1x1 conv
                    OFM[i][k][l] += ADDER_OP(WEIGHT[i][j], IFM[j][k*2][l*2]);
                }
            }
        }
    }
}

void conv1x1_1_adder(input_t IFM[32][16][16], weight_t WEIGHT[4][32], 
                     result_t OFM[4][8][8]) {
    for(int k=0; k<8; k++) {
        for(int l=0; l<8; l++) {
            for(int i=0; i<4; i++) {
                for(int j=0; j<32; j++) {
                    OFM[i][k][l] += ADDER_OP(WEIGHT[i][j], IFM[j][k*2][l*2]);
                }
            }
        }
    }
}

// ========== Batch Normalization with FBR ==========
// Y = γ * (X - μ') / √(σ² + ε) + β'
void bn1(input_t IFM[16][32][32], result_t OFM[16][32][32], 
         input_t* gamma, input_t* beta, input_t* mean, input_t* var) {
    #pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
    
    const input_t eps = 1e-5;
    
    for (int i=0; i<32; i++) {
        for (int j=0; j<32; j++) {
            #pragma HLS pipeline
            for (int c=0; c<16; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = (IFM[c][i][j] - mean[c]) * std_inv;
                result_t scaled = normalized * gamma[c] + beta[c];
                
                // ReLU activation
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}

void bn2(input_t IFM[32][16][16], result_t OFM[32][16][16], 
         input_t* gamma, input_t* beta, input_t* mean, input_t* var) {
    #pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
    
    const input_t eps = 1e-5;
    
    for (int i=0; i<16; i++) {
        for (int j=0; j<16; j++) {
            #pragma HLS pipeline
            for (int c=0; c<32; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = (IFM[c][i][j] - mean[c]) * std_inv;
                result_t scaled = normalized * gamma[c] + beta[c];
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}

void bn3(input_t IFM[64][8][8], result_t OFM[64][8][8], 
         input_t* gamma, input_t* beta, input_t* mean, input_t* var) {
    #pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
    
    const input_t eps = 1e-5;
    
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            #pragma HLS pipeline
            for (int c=0; c<64; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = (IFM[c][i][j] - mean[c]) * std_inv;
                result_t scaled = normalized * gamma[c] + beta[c];
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}

// ========== Residual Addition ==========
void ADD_0(input_t IFM1[16][32][32], input_t IFM2[16][32][32], 
           result_t OFM[16][32][32]) {
    for (int j=0; j<32; j++) {
        for (int c=0; c<32; c++) {
            for (int i=0; i<16; i++) {
                OFM[i][j][c] = IFM1[i][j][c] + IFM2[i][j][c];
            }
        }
    }
}

void ADD_1(input_t IFM1[32][16][16], input_t IFM2[32][16][16], 
           result_t OFM[32][16][16]) {
    for (int i=0; i<32; i++) {
        for (int j=0; j<16; j++) {
            for (int c=0; c<16; c++) {
                OFM[i][j][c] = IFM1[i][j][c] + IFM2[i][j][c];
            }
        }
    }
}

void ADD_2(input_t IFM1[64][8][8], input_t IFM2[64][8][8], 
           result_t OFM[64][8][8]) {
    for (int i=0; i<64; i++) {
        for (int j=0; j<8; j++) {
            for (int c=0; c<8; c++) {
                OFM[i][j][c] = IFM1[i][j][c] + IFM2[i][j][c];
            }
        }
    }
}

// ========== Main AdderNet 2.0 Model ==========
void ADDERNET20_2_0(input_t* IFM, input_t* weight, result_t* OFM) {
    
    #pragma HLS ALLOCATION function instances=padding_0 limit=1
    #pragma HLS ALLOCATION function instances=padding_1 limit=1
    #pragma HLS ALLOCATION function instances=padding_2 limit=1
    #pragma HLS ALLOCATION function instances=wreshape_01 limit=1
    #pragma HLS ALLOCATION function instances=wreshape_20 limit=1
    #pragma HLS ALLOCATION function instances=wreshape_21 limit=1
    #pragma HLS ALLOCATION function instances=wreshape_30 limit=1
    #pragma HLS ALLOCATION function instances=wreshape_31 limit=1
    #pragma HLS ALLOCATION function instances=PE0_16_16_adder limit=1
    #pragma HLS ALLOCATION function instances=PE1_16_32_adder limit=1
    #pragma HLS ALLOCATION function instances=PE2_16_64_adder limit=1
    #pragma HLS ALLOCATION function instances=conv1x1_adder limit=1
    #pragma HLS ALLOCATION function instances=conv1x1_1_adder limit=1
    #pragma HLS ALLOCATION function instances=ADD_0 limit=1
    #pragma HLS ALLOCATION function instances=ADD_1 limit=1
    #pragma HLS ALLOCATION function instances=ADD_2 limit=1
    #pragma HLS ALLOCATION function instances=bn1 limit=1
    #pragma HLS ALLOCATION function instances=bn2 limit=1
    #pragma HLS ALLOCATION function instances=bn3 limit=1

    // Initial convolution 
    IFM_reshape(IFM, ADD0);
    padding_0(ADD0, padded);
    wreshape_0(weight, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 1: Block 0 ==========
    padding_0(ADD0, padded);
    wreshape_01(layer1_0_conv1_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_0(ADD0, padded);
    wreshape_01(layer1_0_conv2_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 1: Block 1 ==========
    padding_0(ADD0, padded);
    wreshape_01(layer1_1_conv1_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_0(ADD0, padded);
    wreshape_01(layer1_1_conv2_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 1: Block 2 ==========
    padding_0(ADD0, padded);
    wreshape_01(layer1_2_conv1_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_0(ADD0, padded);
    wreshape_01(layer1_2_conv2_adder, WBUF3x3);
    PE0_16_16_adder(padded, WBUF3x3, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 2: Downsample Path/Block 3 ==========
    for(int c=0; c<4; c++) {
        wreshape_downsample1(layer2_0_downsample_0_adder, WBUF1x1, c);
        conv1x1_adder(ADD0, WBUF1x1, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    bn2(OFM_out_1, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 2: Block 0 ==========
    padding_0(ADD0, padded);
    for(int c=0; c<2; c++) {
        wreshape_20(layer2_0_conv1_adder, WBUF3x3, c);
        PE0_16_16_adder(padded, WBUF3x3, p1);
        lay2_reshape_down(p1, OFM_out_1, c);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_1(OFM_out_2, padded_1);
    for(int c=0; c<2; c++) {
        wreshape_21(layer2_0_conv2_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 2: Block 1 ==========
    padding_1(ADD1, padded_1);
    for(int c=0; c<2; c++) {
        wreshape_21(layer2_1_conv1_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_1(OFM_out_2, padded_1);
    for(int c=0; c<2; c++) {
        wreshape_21(layer2_1_conv2_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 2: Block 2 ==========
    padding_1(ADD1, padded_1);
    for(int c=0; c<2; c++) {
        wreshape_21(layer2_2_conv1_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_1(OFM_out_2, padded_1);
    for(int c=0; c<2; c++) {
        wreshape_21(layer2_2_conv2_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay2_reshape(part_ofm, OFM_out_1, c);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 3: Downsample Path/Block3 ==========
    for(int c=0; c<16; c++) {
        wreshape_downsample2(layer3_0_downsample_0_adder, WBUF1x1_1, c);
        conv1x1_1_adder(ADD1, WBUF1x1_1, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_4, c);
    }
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 3: Block 0 ==========
    padding_1(ADD1, padded_1);
    for(int c=0; c<4; c++) {
        wreshape_30(layer3_0_conv1_adder, WBUF3x3_2, c);
        PE1_16_32_adder(padded_1, WBUF3x3_2, part_ofm);
        lay3_reshape_down(part_ofm, OFM_out_3, c);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_2(OFM_out_4, padded_2);
    for(int c=0; c<4; c++) {
        wreshape_31(layer3_0_conv2_adder, WBUF3x3_3, c);
        PE2_16_64_adder(padded_2, WBUF3x3_3, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_3, c);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 3: Block 1 ==========
    padding_2(ADD2, padded_2);
    for(int c=0; c<4; c++) {
        wreshape_31(layer3_1_conv1_adder, WBUF3x3_3, c);
        PE2_16_64_adder(padded_2, WBUF3x3_3, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_3, c);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_2(OFM_out_4, padded_2);
    for(int c=0; c<4; c++) {
        wreshape_31(layer3_1_conv2_adder, WBUF3x3_3, c);
        PE2_16_64_adder(padded_2, WBUF3x3_3, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_3, c);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Layer 3: Block 2 ==========
    padding_2(ADD2, padded_2);
    for(int c=0; c<4; c++) {
        wreshape_31(layer3_2_conv1_adder, WBUF3x3_3, c);
        PE2_16_64_adder(padded_2, WBUF3x3_3, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_3, c);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    padding_2(OFM_out_4, padded_2);
    for(int c=0; c<4; c++) {
        wreshape_31(layer3_2_conv2_adder, WBUF3x3_3, c);
        PE2_16_64_adder(padded_2, WBUF3x3_3, part_ofm2);
        lay3_reshape(part_ofm2, OFM_out_3, c);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);
    
    // ========== Global Average Pooling ==========
    for(int c=0; c<64; c++) {
        result_t sum = 0;
        for(int i=0; i<8; i++) {
            for(int j=0; j<8; j++) {
                sum += ADD2[c][i][j];
            }
        }
        AV[c] = sum / 64.0;
    }
    
    // ========== Fully Connected Layer ==========
    for(int i=0; i<10; i++) {
        result_t sum = 0;
        for(int j=0; j<64; j++) {
            sum += fc_weight[i*64+j] * AV[j];
        }
        FC1[i] = sum;
    }
    
    // ========== Final Batch Norm  ==========
    const input_t eps = 1e-5;
    for(int i=0; i<10; i++) {
        input_t std_inv = 1.0 / hls::sqrt(bn2_var[i] + eps);
        result_t normalized = (FC1[i] - bn2_mean[i]) * std_inv;
        OFM[i] = normalized * bn2_weight[i] + bn2_bias[i];
    }
}