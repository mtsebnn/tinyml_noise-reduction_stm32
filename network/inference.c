#include "nn_data.h"

// data
const nn_data_t data = {
    .name = "data", .w1 = w1,
    .b1 = b1, .w2 = w2,
    .b2 = B2, .requant_multiplier = REQUANT_MULT,
    .shift_value = SHIFT_VALUE
};

// neural network fixed-point arithmetic
int32_t predict_nn(const int8_t *input_window, const nn_data_t *config) {

    // layer 1 (16 inputs, 8 outputs)
    int32_t layer1_out[HIDDEN_NEURONS] = {0};
    for(int neuron = 0; neuron < HIDDEN_NEURONS; neuron++) {
        int32_t neuron_sum = config->b1[neuron]; // start with bias
        for(int i = 0; i < WINDOW_SIZE; i++) {
            neuron_sum += ((int32_t)input_window[i] * (int32_t)config->w1[neuron][i]);
        }
        // ReLU activation function (max(0, x))
        if (neuron_sum < 0) {
            neuron_sum = 0;
        }
        // requantization 32-bit -> 8-bit
        int64_t scaled_out = (int64_t)neuron_sum * (int64_t)config->requant_multiplier; // 64-bit multiplication to prevent overflow
        int32_t shifted_out = (int32_t)((scaled_out + (1 << (config->shift_value - 1))) >> config->shift_value); // rounded bit-shift
        // interval 0, 127 (negative values were already eliminated by ReLU)
        if (shifted_out > 127) {
            layer1_out[neuron] = 127;
        } else {
            layer1_out[neuron] = shifted_out;
        }
    }

    // layer 2 (8 inputs, 1 output)
    int32_t layer2_out = config->b2;
    for(int neuron = 0; neuron < HIDDEN_NEURONS; neuron++) {
        layer2_out += (layer1_out[neuron] * (int32_t)config->w2[neuron]);
    }
    return layer2_out;
}