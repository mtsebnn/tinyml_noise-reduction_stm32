#include "nn_data.h"

// scenarios
const nn_scenario_t scenario_whitenoise = {
    .name = "only whitenoise", .w1 = w1_only_whitenoise,
    .b1 = b1_only_whitenoise, .w2 = w2_only_whitenoise,
    .b2 = B2_ONLY_WHITENOISE, .requant_multiplier = REQUANT_MULT_ONLY_WHITENOISE,
    .shift_value = SHIFT_VALUE_ONLY_WHITENOISE
};

const nn_scenario_t scenario_echo = {
    .name = "only echo", .w1 = w1_only_echo,
    .b1 = b1_only_echo, .w2 = w2_only_echo,
    .b2 = B2_ONLY_ECHO, .requant_multiplier = REQUANT_MULT_ONLY_ECHO,
    .shift_value = SHIFT_VALUE_ONLY_ECHO
};

const nn_scenario_t scenario_quantization = {
    .name = "only quantization", .w1 = w1_only_quantization,
    .b1 = b1_only_quantization, .w2 = w2_only_quantization,
    .b2 = B2_ONLY_QUANTIZATION, .requant_multiplier = REQUANT_MULT_ONLY_QUANTIZATION,
    .shift_value = SHIFT_VALUE_ONLY_QUANTIZATION
};

const nn_scenario_t scenario_combined = {
    .name = "combined", .w1 = w1_combined,
    .b1 = b1_combined, .w2 = w2_combined,
    .b2 = B2_COMBINED, .requant_multiplier = REQUANT_MULT_COMBINED,
    .shift_value = SHIFT_VALUE_COMBINED
};

// neural network fixed-point arithmetic
int32_t predict(const int8_t *input_window, const nn_scenario_t *config) {

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