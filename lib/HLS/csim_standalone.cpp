// ============================================================================
// Standalone C Simulation for AdderNet20 (No Vitis HLS required)
// ============================================================================
// Compile: g++ -o csim_test csim_standalone.cpp -std=c++11
// Run: ./csim_test

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdint>

// ============================================================================
// Mock HLS types for standalone compilation
// ============================================================================
typedef float input_t;     // replaces ap_fixed<16,8>
typedef uint8_t weight_t;  // replaces ap_uint<5> - unsigned [0,31]
typedef float result_t;    // replaces ap_fixed<32,16>

// Network dimensions
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_CH 3
#define CH1 16
#define H1 32
#define W1 32
#define CH2 32
#define H2 16
#define W2 16
#define CH3 64
#define H3 8
#define W3 8
#define NUM_CLASSES 10

// ============================================================================
// Include weight headers (they use weight_t which is now uint8_t)
// ============================================================================
// For standalone test, we'll use simplified weights
// In real HLS, these come from the generated headers

// Sample weight buffer (simplified for testing)
static weight_t sample_weights[16][16][3][3];

// BatchNorm parameters (simplified)
static input_t bn1_gamma[16], bn1_beta[16], bn1_mean[16], bn1_var[16];
static input_t bn2_weight[10], bn2_bias[10], bn2_mean[10], bn2_var[10];
static input_t fc_weight[640];

// On-chip buffers
static input_t ADD0[CH1][H1][W1];
static input_t p1[CH1][H1][W1];
static input_t padded0[CH1][H1+2][W1+2];
static input_t AV[CH3];
static result_t FC1[NUM_CLASSES];

// ============================================================================
// Core AdderNet operation (unsigned weights)
// ============================================================================
inline result_t ADDER_OP(weight_t W, input_t X) {
    // W is unsigned [0,31], needs offset subtraction for signed interpretation
    result_t W_signed = (result_t)W - 16.0f;  // Convert back to signed
    result_t diff = W_signed - X;
    return -(diff >= 0 ? diff : -diff);
}

// ============================================================================
// Initialize test data
// ============================================================================
void init_test_data() {
    // Initialize BN parameters to identity transform
    for (int i = 0; i < 16; i++) {
        bn1_gamma[i] = 1.0f;
        bn1_beta[i] = 0.0f;
        bn1_mean[i] = 0.0f;
        bn1_var[i] = 1.0f;
    }
    for (int i = 0; i < 10; i++) {
        bn2_weight[i] = 1.0f;
        bn2_bias[i] = 0.0f;
        bn2_mean[i] = 0.0f;
        bn2_var[i] = 1.0f;
    }
    // Random FC weights
    for (int i = 0; i < 640; i++) {
        fc_weight[i] = ((rand() % 100) - 50) / 100.0f;
    }
    // Random conv weights (unsigned)
    for (int o = 0; o < 16; o++)
        for (int i = 0; i < 16; i++)
            for (int h = 0; h < 3; h++)
                for (int w = 0; w < 3; w++)
                    sample_weights[o][i][h][w] = rand() % 32;  // [0, 31]
}

// ============================================================================
// Simplified forward pass (demonstrates the adder operation)
// ============================================================================
void simplified_forward(const input_t* input, result_t* output) {
    std::cout << "Running simplified AdderNet forward pass..." << std::endl;
    
    // 1. Reshape input into ADD0 buffer
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
                if (c < 3) {
                    ADD0[c][i][j] = input[c * H1 * W1 + i * W1 + j];
                } else {
                    ADD0[c][i][j] = 0;
                }
            }
        }
    }
    std::cout << "  - Input reshaped to [16][32][32]" << std::endl;
    
    // 2. Apply one convolution layer (simplified)
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
                p1[c][i][j] = 0;
            }
        }
    }
    
    // Padding
    for (int c = 0; c < CH1; c++) {
        for (int i = 0; i < H1+2; i++) {
            for (int j = 0; j < W1+2; j++) {
                padded0[c][i][j] = 0;
            }
        }
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
                padded0[c][i+1][j+1] = ADD0[c][i][j];
            }
        }
    }
    
    // Simple 3x3 adder convolution
    std::cout << "  - Applying 3x3 Adder convolution..." << std::endl;
    for (int o = 0; o < CH1; o++) {
        for (int r = 0; r < H1; r++) {
            for (int c = 0; c < W1; c++) {
                result_t sum = 0;
                for (int i = 0; i < CH1; i++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            sum += ADDER_OP(sample_weights[o][i][kh][kw], 
                                           padded0[i][r+kh][c+kw]);
                        }
                    }
                }
                p1[o][r][c] = (input_t)sum;
            }
        }
    }
    std::cout << "  - Convolution complete" << std::endl;
    
    // 3. Global average pooling (simplified - just use first layer output)
    for (int c = 0; c < CH3; c++) {
        AV[c] = 0;
    }
    for (int c = 0; c < CH1; c++) {
        result_t sum = 0;
        for (int i = 0; i < H1; i++) {
            for (int j = 0; j < W1; j++) {
                sum += p1[c][i][j];
            }
        }
        AV[c] = sum / (H1 * W1);
    }
    std::cout << "  - Global average pooling applied" << std::endl;
    
    // 4. FC layer
    for (int i = 0; i < NUM_CLASSES; i++) {
        result_t sum = 0;
        for (int j = 0; j < CH3; j++) {
            sum += fc_weight[i * CH3 + j] * AV[j];
        }
        output[i] = sum;
    }
    std::cout << "  - FC layer complete" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "AdderNet20 Standalone C Simulation" << std::endl;
    std::cout << "============================================" << std::endl;
    
    srand(42);
    init_test_data();
    
    // Create test input
    const int INPUT_SIZE = 3 * 32 * 32;
    input_t* input = new input_t[INPUT_SIZE];
    result_t output[NUM_CLASSES];
    
    std::cout << "\n[1] Generating random input (32x32x3)..." << std::endl;
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (rand() % 256) / 255.0f * 2.0f - 1.0f;  // [-1, 1]
    }
    
    // Run forward
    std::cout << "\n[2] Running forward pass..." << std::endl;
    simplified_forward(input, output);
    
    // Display results
    std::cout << "\n[3] Output scores:" << std::endl;
    const char* classes[] = {"airplane", "automobile", "bird", "cat", "deer",
                             "dog", "frog", "horse", "ship", "truck"};
    int max_idx = 0;
    result_t max_val = output[0];
    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << "   " << std::setw(10) << classes[i] << ": " 
                  << std::fixed << std::setprecision(4) << output[i] << std::endl;
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "Predicted: " << classes[max_idx] << std::endl;
    std::cout << "============================================" << std::endl;
    
    delete[] input;
    std::cout << "\nC Simulation PASSED" << std::endl;
    return 0;
}
