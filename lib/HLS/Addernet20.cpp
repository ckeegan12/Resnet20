// Addernet20.cpp - HLS implementation tuned for KV260/PYNQ
// Expects per-layer weights and batchnorm arrays to be provided as headers in lib/HLS/weights/...
#include "Addernet20.h"
#include "parameters.h"
#include <hls_math.h>

// ----------------- Weight & BN headers (you must provide these) -----------------
// +-+-+-+-+----------------- Layer 1 -----------------+-+-+-+-+-+-+-+
#include "Layer_1_header/initial_conv_adder.h"
#include "Layer_1_header/layer1_0_conv1_adder.h"
#include "Layer_1_header/layer1_0_conv2_adder.h"
#include "Layer_1_header/layer1_1_conv1_adder.h"
#include "Layer_1_header/layer1_1_conv2_adder.h"
#include "Layer_1_header/layer1_2_conv1_adder.h"
#include "Layer_1_header/layer1_2_conv2_adder.h"


// +-+-+-+-+----------------- Layer 2 -----------------+-+-+-+-+-+-+-+
#include "Layer_2_header/layer2_0_downsample_0_adder.h"
#include "Layer_2_header/layer2_0_conv1_adder.h"
#include "Layer_2_header/layer2_0_conv2_adder.h"
#include "Layer_2_header/layer2_1_conv1_adder.h"
#include "Layer_2_header/layer2_1_conv2_adder.h"
#include "Layer_2_header/layer2_2_conv1_adder.h"
#include "Layer_2_header/layer2_2_conv2_adder.h"

// +-+-+-+-+----------------- Layer 3 -----------------+-+-+-+-+-+-+-+
#include "Layer_3_header/layer3_0_downsample_0_adder.h"
#include "Layer_3_header/layer3_0_conv1_adder.h"
#include "Layer_3_header/layer3_0_conv2_adder.h"
#include "Layer_3_header/layer3_1_conv1_adder.h"
#include "Layer_3_header/layer3_1_conv2_adder.h"
#include "Layer_3_header/layer3_2_conv1_adder.h"
#include "Layer_3_header/layer3_2_conv2_adder.h"

// +-+-+-+-+----------------- BatchNorm -----------------+-+-+-+-+-+-+-+
#include "Batchnorm/bn1_gamma.h"
#include "Batchnorm/bn1_beta.h"
#include "Batchnorm/bn1_mean.h"
#include "Batchnorm/bn1_var.h"

// ----------------- Fully Connected + BN2 -----------------
#include "Batchnorm/fc_weight.h"
#include "Batchnorm/bn2_weight.h"
#include "Batchnorm/bn2_bias.h"
#include "Batchnorm/bn2_mean.h"
#include "Batchnorm/bn2_var.h"

// ----------------- On-chip buffers (static to keep placement stable) -----------------
static input_t ADD0[CH1][H1][W1];
static input_t p1[CH1][H1][W1];
static input_t p2[CH1][H1][W1];
static input_t padded0[CH1][H1+2][W1+2];

static input_t ADD1[CH2][H2][W2];
static input_t OFM_out_1[CH2][H2][W2];
static input_t OFM_out_2[CH2][H2][W2];
static input_t padded1[CH2][H2+2][W2+2];
static result_t part_ofm1[8][H2][W2]; // 1x1 conv small result

static input_t ADD2[CH3][H3][W3];
static input_t OFM_out_3[CH3][H3][W3];
static input_t OFM_out_4[CH3][H3][W3];
static input_t padded2[CH3][H3+2][W3+2];
static result_t part_ofm2[4][H3][W3];

static weight_t WBUF3x3_16x16[CH1][CH1][3][3];
static weight_t WBUF3x3_16x32[CH1][CH2][3][3];
static weight_t WBUF3x3_16x64[CH1][CH3][3][3];
static weight_t WBUF1x1_8x16[8][CH1];
static weight_t WBUF1x1_4x32[4][CH2];

static input_t AV[CH3];
static result_t FC1[NUM_CLASSES];

// ----------------- Core primitives -----------------
inline result_t ADDER_OP_scalar(weight_t W, input_t X) {
#pragma HLS INLINE
    result_t diff = (result_t)W - (result_t)X;
    result_t absd = (diff >= 0) ? diff : -diff;
    return -absd;
}

inline result_t ADDER_4(weight_t W0, input_t X0, weight_t W1, input_t X1,
                       weight_t W2, input_t X2, weight_t W3, input_t X3) {
#pragma HLS INLINE
    result_t s0 = (result_t)W0 - (result_t)X0; if (s0 < 0) s0 = -s0;
    result_t s1 = (result_t)W1 - (result_t)X1; if (s1 < 0) s1 = -s1;
    result_t s2 = (result_t)W2 - (result_t)X2; if (s2 < 0) s2 = -s2;
    result_t s3 = (result_t)W3 - (result_t)X3; if (s3 < 0) s3 = -s3;
    return -(s0 + s1 + s2 + s3);
}

// ----------------- reshape & padding helpers -----------------
static void IFM_reshape(const input_t* IFM, input_t OFM[CH1][H1][W1]) {
    // map first 3 channels from flattened IFM into channels 0..2; rest zero
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
#pragma HLS PIPELINE
                if (c < 3) {
                    OFM[c][i][j] = IFM[c*H1*W1 + i*W1 + j];
                } else {
                    OFM[c][i][j] = 0;
                }
            }
        }
    }
}

static void padding0(input_t D[CH1][H1][W1], input_t IFM[CH1][H1+2][W1+2]) {
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
#pragma HLS PIPELINE
                IFM[c][i+1][j+1] = D[c][i][j];
            }
        }
    }
}
static void padding1(input_t D[CH2][H2][W2], input_t IFM[CH2][H2+2][W2+2]) {
    for (int c = 0; c < CH2; c++) {
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < W2; j++) {
#pragma HLS PIPELINE
                IFM[c][i+1][j+1] = D[c][i][j];
            }
        }
    }
}
static void padding2(input_t D[CH3][H3][W3], input_t IFM[CH3][H3+2][W3+2]) {
    for (int c = 0; c < CH3; c++) {
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < W3; j++) {
#pragma HLS PIPELINE
                IFM[c][i+1][j+1] = D[c][i][j];
            }
        }
    }
}

// ----------------- Layer reshape helpers -----------------
// Layer 2: Copy 16-channel slice into 32-channel output at offset (slice * 16)
void lay2_reshape(input_t IFM1[16][H2][W2], input_t OFM[CH2][H2][W2], int slice) {
#pragma HLS INLINE off
    int base_ch = slice * 16;
    for (int c = 0; c < 16; c++) {
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < W2; j++) {
#pragma HLS PIPELINE
                OFM[base_ch + c][i][j] = IFM1[c][i][j];
            }
        }
    }
}

// Layer 2 downsample: Downsample from H1xW1 to H2xW2 with stride-2, place at slice offset
void lay2_reshape_down(input_t IFM1[CH1][H1][W1], input_t OFM[CH2][H2][W2], int slice) {
#pragma HLS INLINE off
    int base_ch = slice * CH1;
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H2; i++) {
            for (int j = 0; j < W2; j++) {
#pragma HLS PIPELINE
                OFM[base_ch + c][i][j] = IFM1[c][i * 2][j * 2];
            }
        }
    }
}

// Layer 3: Copy 16-channel slice into 64-channel output at offset (slice * 4)
void lay3_reshape(input_t IFM1[16][H3][W3], input_t OFM[CH3][H3][W3], int slice) {
#pragma HLS INLINE off
    int base_ch = slice * 4;
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < W3; j++) {
#pragma HLS PIPELINE
                OFM[base_ch + c][i][j] = IFM1[c][i][j];
            }
        }
    }
}

// Layer 3 downsample: Downsample from H2xW2 to H3xW3 with stride-2, place at slice offset
void lay3_reshape_down(input_t IFM1[CH1][H2][W2], input_t OFM[CH3][H3][W3], int slice) {
#pragma HLS INLINE off
    int base_ch = slice * CH1;
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H3; i++) {
            for (int j = 0; j < W3; j++) {
#pragma HLS PIPELINE
                OFM[base_ch + c][i][j] = IFM1[c][i * 2][j * 2];
            }
        }
    }
}


// ----------------- PEs (use shaped header arrays directly) -----------------
static void PE0_16_16_adder(input_t IFM[CH1][H1+2][W1+2], const weight_t WBUF[CH1][CH1][3][3], input_t OFM[CH1][H1][W1]) {
#pragma HLS INLINE off
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < H1; o_row++) {
                for (int o_col = 0; o_col < W1; o_col++) {
#pragma HLS PIPELINE II=1
                    for (int o_ch = 0; o_ch < CH1; o_ch++) {
#pragma HLS UNROLL factor=1
                        for (int i_ch = 0; i_ch < CH1; i_ch += 4) {
#pragma HLS UNROLL factor=1
                            OFM[o_ch][o_row][o_col] += (input_t)ADDER_4(
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

static void PE1_16_32_adder(input_t IFM[CH2][H2+2][W2+2], const weight_t WBUF[CH1][CH2][3][3], input_t OFM[CH1][H2][W2]) {
#pragma HLS INLINE off
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < H2; o_row++) {
                for (int o_col = 0; o_col < W2; o_col++) {
#pragma HLS PIPELINE II=1
                    for (int o_ch = 0; o_ch < CH1; o_ch++) {
#pragma HLS UNROLL factor=1
                        for (int i_ch = 0; i_ch < CH2; i_ch += 4) {
#pragma HLS UNROLL factor=1
                            OFM[o_ch][o_row][o_col] += (input_t)ADDER_4(
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

static void PE2_16_64_adder(input_t IFM[CH3][H3+2][W3+2], const weight_t WBUF[CH1][CH3][3][3], input_t OFM[CH1][H3][W3]) {
#pragma HLS INLINE off
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o_row = 0; o_row < H3; o_row++) {
                for (int o_col = 0; o_col < W3; o_col++) {
#pragma HLS PIPELINE II=1
                    for (int o_ch = 0; o_ch < CH1; o_ch++) {
#pragma HLS UNROLL factor=1
                        for (int i_ch = 0; i_ch < CH3; i_ch += 4) {
#pragma HLS UNROLL factor=1
                            OFM[o_ch][o_row][o_col] += (input_t)ADDER_4(
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

// 1x1 conv used for downsample path (weights passed as shaped slices from headers)
static void conv1x1_adder(input_t IFM[CH1][H1][W1], const weight_t WEIGHT[8][CH1], result_t OFM[8][H2][W2]) {
    for (int k = 0; k < H2; k++) {
        for (int l = 0; l < W2; l++) {
#pragma HLS PIPELINE II=1
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < CH1; j++) {
                    OFM[i][k][l] += ADDER_OP_scalar(WEIGHT[i][j], IFM[j][k*2][l*2]);
                }
            }
        }
    }
}
static void conv1x1_1_adder(input_t IFM[CH2][H2][W2], const weight_t WEIGHT[4][CH2], result_t OFM[4][H3][W3]) {
    for (int k = 0; k < H3; k++) {
        for (int l = 0; l < W3; l++) {
#pragma HLS PIPELINE II=1
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < CH2; j++) {
                    OFM[i][k][l] += ADDER_OP_scalar(WEIGHT[i][j], IFM[j][k*2][l*2]);
                }
            }
        }
    }
}

// BatchNorm + ReLU
static void bn1(input_t IFM[CH1][H1][W1], result_t OFM[CH1][H1][W1], const input_t* gamma, const input_t* beta, const input_t* mean, const input_t* var) {
    const input_t eps = (input_t)1e-5;
    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < W1; j++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < CH1; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = ((result_t)IFM[c][i][j] - (result_t)mean[c]) * (result_t)std_inv;
                result_t scaled = normalized * (result_t)gamma[c] + (result_t)beta[c];
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}
static void bn2(input_t IFM[CH2][H2][W2], result_t OFM[CH2][H2][W2], const input_t* gamma, const input_t* beta, const input_t* mean, const input_t* var) {
    const input_t eps = (input_t)1e-5;
    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < W2; j++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < CH2; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = ((result_t)IFM[c][i][j] - (result_t)mean[c]) * (result_t)std_inv;
                result_t scaled = normalized * (result_t)gamma[c] + (result_t)beta[c];
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}
static void bn3(input_t IFM[CH3][H3][W3], result_t OFM[CH3][H3][W3], const input_t* gamma, const input_t* beta, const input_t* mean, const input_t* var) {
    const input_t eps = (input_t)1e-5;
    for (int i = 0; i < H3; i++) {
        for (int j = 0; j < W3; j++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < CH3; c++) {
                input_t std_inv = 1.0 / hls::sqrt(var[c] + eps);
                result_t normalized = ((result_t)IFM[c][i][j] - (result_t)mean[c]) * (result_t)std_inv;
                result_t scaled = normalized * (result_t)gamma[c] + (result_t)beta[c];
                OFM[c][i][j] = (scaled > 0) ? scaled : 0;
            }
        }
    }
}

// Residual add
static void ADD_0(input_t IFM1[CH1][H1][W1], input_t IFM2[CH1][H1][W1], result_t OFM[CH1][H1][W1]) {
    for (int j = 0; j < H1; j++) {
        for (int k = 0; k < W1; k++) {
#pragma HLS PIPELINE II=1
            for (int i = 0; i < CH1; i++) {
                OFM[i][j][k] = (result_t)IFM1[i][j][k] + (result_t)IFM2[i][j][k];
            }
        }
    }
}
static void ADD_1(input_t IFM1[CH2][H2][W2], input_t IFM2[CH2][H2][W2], result_t OFM[CH2][H2][W2]) {
    for (int i = 0; i < CH2; i++) {
        for (int j = 0; j < H2; j++) {
            for (int k = 0; k < W2; k++) {
#pragma HLS PIPELINE II=1
                OFM[i][j][k] = (result_t)IFM1[i][j][k] + (result_t)IFM2[i][j][k];
            }
        }
    }
}
static void ADD_2(input_t IFM1[CH3][H3][W3], input_t IFM2[CH3][H3][W3], result_t OFM[CH3][H3][W3]) {
    for (int i = 0; i < CH3; i++) {
        for (int j = 0; j < H3; j++) {
            for (int k = 0; k < W3; k++) {
#pragma HLS PIPELINE II=1
                OFM[i][j][k] = (result_t)IFM1[i][j][k] + (result_t)IFM2[i][j][k];
            }
        }
    }
}

// ----------------- Top-level kernel -----------------
extern "C" void ADDERNET20_2_0(const input_t* IFM_flat, const weight_t* weights, result_t* OFM) {
#pragma HLS INTERFACE m_axi port=IFM_flat offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=weights  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=OFM      offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=IFM_flat bundle=control
#pragma HLS INTERFACE s_axilite port=weights    bundle=control
#pragma HLS INTERFACE s_axilite port=OFM        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

#pragma HLS DATAFLOW

    // 1) Reshape input flattened buffer into on-chip CH1xH1xW1 buffer
    IFM_reshape(IFM_flat, ADD0);

    // 2) Initial convolution: use shaped header array initial_conv_adder
    // Clear accumulators for p1
    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    // initial_conv_adder must be weight_t[CH1][CH1][3][3]
    PE0_16_16_adder(padded0, initial_conv_adder, p1);

    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 1: Block 0 ==========
    // Use the header arrays directly (already shaped)
    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_0_conv1_adder, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_0_conv2_adder, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 1: Block 1 ==========
    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_1_conv1_adder, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_1_conv2_adder, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 1: Block 2 ==========
    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_2_conv1_adder, p1);
    bn1(p1, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    for (int c = 0; c < CH1; c++)
        for (int i = 0; i < H1; i++)
            for (int j = 0; j < W1; j++)
                p1[c][i][j] = 0;

    padding0(ADD0, padded0);
    PE0_16_16_adder(padded0, layer1_2_conv2_adder, p1);
    ADD_0(p1, ADD0, p2);
    bn1(p2, ADD0, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 2: Downsample Path ==========
    // layer2_0_downsample_0_adder must be declared as weight_t[4][8][16]
    for (int slice = 0; slice < 4; slice++) {
        // clear small intermediate
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    part_ofm1[i][j][k] = 0;

        conv1x1_adder(ADD0, layer2_0_downsample_0_adder[slice], part_ofm1);
        // lay2_reshape expects input_t IFM1[16][16][16]; we need to copy/convert part_ofm1 result into appropriate OFM_out_1 slice
        // Here we assume a helper lay2_reshape exists; reuse original lay2_reshape signature:
        // lay2_reshape((input_t (*)[H2][W2])part_ofm1, OFM_out_1, slice);
        // Since part_ofm1 is result_t, cast to input_t pointer to match lay2_reshape's parameter type:
        lay2_reshape((input_t(*)[H2][W2])part_ofm1, OFM_out_1, slice);
    }
    bn2(OFM_out_1, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 2: Block 0 ==========
    padding0(ADD0, padded0);
    for (int slice = 0; slice < 2; slice++) {
        // Clear p1
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H1; j++)
                for (int k = 0; k < W1; k++)
                    p1[i][j][k] = 0;

        // layer2_0_conv1_adder expected as weight_t[2][CH1][CH1][3][3]
        PE0_16_16_adder(padded0, layer2_0_conv1_adder[slice], p1);
        lay2_reshape_down(p1, OFM_out_1, slice);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding1(OFM_out_2, padded1);
    for (int slice = 0; slice < 2; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        // layer2_0_conv2_adder expected as weight_t[2][CH1][CH2][3][3]
        PE1_16_32_adder(padded1, layer2_0_conv2_adder[slice], p1);
        lay2_reshape((input_t(*)[H2][W2])p1, OFM_out_1, slice);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 2: Block 1 ==========
    padding1(ADD1, padded1);
    for (int slice = 0; slice < 2; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        PE1_16_32_adder(padded1, layer2_1_conv1_adder[slice], p1);
        lay2_reshape((input_t(*)[H2][W2])p1, OFM_out_1, slice);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding1(OFM_out_2, padded1);
    for (int slice = 0; slice < 2; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        PE1_16_32_adder(padded1, layer2_1_conv2_adder[slice], p1);
        lay2_reshape((input_t(*)[H2][W2])p1, OFM_out_1, slice);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 2: Block 2 ==========
    padding1(ADD1, padded1);
    for (int slice = 0; slice < 2; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        PE1_16_32_adder(padded1, layer2_2_conv1_adder[slice], p1);
        lay2_reshape((input_t(*)[H2][W2])p1, OFM_out_1, slice);
    }
    bn2(OFM_out_1, OFM_out_2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding1(OFM_out_2, padded1);
    for (int slice = 0; slice < 2; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        PE1_16_32_adder(padded1, layer2_2_conv2_adder[slice], p1);
        lay2_reshape((input_t(*)[H2][W2])p1, OFM_out_1, slice);
    }
    ADD_1(OFM_out_1, ADD1, OFM_out_2);
    bn2(OFM_out_2, ADD1, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 3: Downsample Path ==========
    // layer3_0_downsample_0_adder must be weight_t[16][4][32]
    for (int slice = 0; slice < 16; slice++) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    part_ofm2[i][j][k] = 0;

        conv1x1_1_adder(ADD1, layer3_0_downsample_0_adder[slice], part_ofm2);
        // lay3_reshape expects input_t IFM1[16][8][8]; cast part_ofm2 result to input_t pointer
        lay3_reshape((input_t(*)[H3][W3])part_ofm2, OFM_out_4, slice);
    }
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 3: Block 0 ==========
    padding1(ADD1, padded1);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < W2; k++)
                    p1[i][j][k] = 0;

        PE1_16_32_adder(padded1, layer3_0_conv1_adder[slice], p1);
        lay3_reshape_down(p1, OFM_out_3, slice);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding2(OFM_out_4, padded2);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    p1[i][j][k] = 0;

        PE2_16_64_adder(padded2, layer3_0_conv2_adder[slice], p1);
        lay3_reshape((input_t(*)[H3][W3])p1, OFM_out_3, slice);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 3: Block 1 ==========
    padding2(ADD2, padded2);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    p1[i][j][k] = 0;

        PE2_16_64_adder(padded2, layer3_1_conv1_adder[slice], p1);
        lay3_reshape((input_t(*)[H3][W3])p1, OFM_out_3, slice);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding2(OFM_out_4, padded2);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    p1[i][j][k] = 0;

        PE2_16_64_adder(padded2, layer3_1_conv2_adder[slice], p1);
        lay3_reshape((input_t(*)[H3][W3])p1, OFM_out_3, slice);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    // ========== Layer 3: Block 2 ==========
    padding2(ADD2, padded2);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    p1[i][j][k] = 0;

        PE2_16_64_adder(padded2, layer3_2_conv1_adder[slice], p1);
        lay3_reshape((input_t(*)[H3][W3])p1, OFM_out_3, slice);
    }
    bn3(OFM_out_3, OFM_out_4, bn1_gamma, bn1_beta, bn1_mean, bn1_var);

    padding2(OFM_out_4, padded2);
    for (int slice = 0; slice < 4; slice++) {
        for (int i = 0; i < CH1; i++)
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < W3; k++)
                    p1[i][j][k] = 0;

        PE2_16_64_adder(padded2, layer3_2_conv2_adder[slice], p1);
        lay3_reshape((input_t(*)[H3][W3])p1, OFM_out_3, slice);
    }
    ADD_2(OFM_out_3, ADD2, OFM_out_4);
    bn3(OFM_out_4, ADD2, bn1_gamma, bn1_beta, bn1_mean, bn1_var);


    // ----------------- Global Average Pooling -----------------
    for (int c = 0; c < CH3; c++) {
#pragma HLS PIPELINE
        result_t sum = 0;
        for (int i = 0; i < H3; i++)
            for (int j = 0; j < W3; j++)
                sum += (result_t)ADD2[c][i][j];
        AV[c] = (input_t)(sum / (H3 * W3));
    }

    // ----------------- Fully connected -----------------
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS PIPELINE
        result_t sum = 0;
        for (int j = 0; j < CH3; j++) {
            sum += (result_t)fc_weight[i*CH3 + j] * (result_t)AV[j];
        }
        FC1[i] = sum;
    }

    // Final BN/scale/bias (bn2)
    const input_t eps = (input_t)1e-5;
    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS PIPELINE
        input_t std_inv = 1.0 / hls::sqrt(bn2_var[i] + eps);
        result_t normalized = (FC1[i] - bn2_mean[i]) * std_inv;
        OFM[i] = normalized * bn2_weight[i] + bn2_bias[i];
    }

}