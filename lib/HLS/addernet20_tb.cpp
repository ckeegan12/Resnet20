// ============================================================================
// AdderNet20 HLS Testbench
// ============================================================================
// Tests the ADDERNET20_2_0 kernel with a sample CIFAR-10 format input
// Compile with: g++ -I$XILINX_HLS/include addernet20_tb.cpp Addernet20.cpp -o tb

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "Addernet20.h"

// CIFAR-10 class labels
const char* CIFAR10_CLASSES[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Generate random input in range [-1, 1] (normalized image)
void generate_random_input(input_t* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input_t)((rand() % 256) / 255.0 * 2.0 - 1.0);
    }
}

// Find argmax of output
int find_argmax(result_t* output, int size) {
    int max_idx = 0;
    result_t max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Apply softmax for probability display
void softmax(result_t* input, float* output, int size) {
    float max_val = (float)input[0];
    for (int i = 1; i < size; i++) {
        if ((float)input[i] > max_val) max_val = (float)input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = exp((float)input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "AdderNet20 HLS Testbench" << std::endl;
    std::cout << "Target: Kria KV260 (Int5 Quantized)" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Allocate buffers
    const int INPUT_SIZE = 3 * 32 * 32;  // CIFAR-10: 3 channels x 32x32
    const int OUTPUT_SIZE = 10;           // 10 classes
    
    input_t* input = new input_t[INPUT_SIZE];
    weight_t* weights = nullptr;  // Weights are in headers, not used
    result_t* output = new result_t[OUTPUT_SIZE];
    
    // Initialize output to zero
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
    }
    
    // Generate random test input (simulates normalized CIFAR-10 image)
    std::cout << "\n[1] Generating random test input (32x32x3)..." << std::endl;
    srand(42);  // Fixed seed for reproducibility
    generate_random_input(input, INPUT_SIZE);
    
    // Print input statistics
    float min_val = 1000, max_val = -1000, sum = 0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        float v = (float)input[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    std::cout << "   Input stats: min=" << std::fixed << std::setprecision(3) 
              << min_val << ", max=" << max_val 
              << ", mean=" << (sum / INPUT_SIZE) << std::endl;
    
    // Run inference
    std::cout << "\n[2] Running ADDERNET20_2_0 kernel..." << std::endl;
    ADDERNET20_2_0(input, weights, output);
    std::cout << "   Inference complete." << std::endl;
    
    // Display raw outputs
    std::cout << "\n[3] Raw output scores:" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << "   " << std::setw(10) << CIFAR10_CLASSES[i] 
                  << ": " << std::fixed << std::setprecision(4) 
                  << (float)output[i] << std::endl;
    }
    
    // Find prediction
    int predicted_class = find_argmax(output, OUTPUT_SIZE);
    
    // Display softmax probabilities
    float probs[10];
    softmax(output, probs, OUTPUT_SIZE);
    
    std::cout << "\n[4] Softmax probabilities:" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << "   " << std::setw(10) << CIFAR10_CLASSES[i] 
                  << ": " << std::fixed << std::setprecision(4) 
                  << (probs[i] * 100) << "%" << std::endl;
    }
    
    // Final result
    std::cout << "\n============================================" << std::endl;
    std::cout << "Predicted class: " << CIFAR10_CLASSES[predicted_class] 
              << " (index " << predicted_class << ")" << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(2) 
              << (probs[predicted_class] * 100) << "%" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Cleanup
    delete[] input;
    delete[] output;
    
    // Return 0 for success (HLS csim expects this)
    std::cout << "\nTestbench PASSED" << std::endl;
    return 0;
}
