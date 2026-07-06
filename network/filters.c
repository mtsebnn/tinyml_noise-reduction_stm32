#include "filters.h"

// kalman
#define Q16_SHIFT 16 // 1 << 16 = 65536
#define Q 3277 // (0.05 * 65536) process noise covariance (how much the true signal is expected to change between two measurements (small value -> signal changes slowly))
#define R 3277 // (0.05 * 65536) measurement noise covariance (expected amount of noise in the sensor measurement)
static int32_t x_pred = 0; // current prediction
static int32_t P = 65536; // (1 shifted by 16 bits) uncertainty of the current prediction (small value -> high trust in prediction) 

// wiener coeff. (calculated in network.py)
const int32_t wiener_weights[16] = {
    -2401, -1597, -702, 128, 1098, 1904, 2527, 2951, 
    3052, 2878, 2433, 1712, 829, -226, -975, -1943
};

// IIR filter
static int32_t iir_state = 0; // internal state of the filter

void reset_filters(void) {
    P = 65536;
    x_pred = 0;
    iir_state = 0;
}

// Kalman algorithm
__attribute__((noinline)) int32_t predict_kalman(int8_t current_sample) {
    P = P + Q; // update uncertainty because of time went by since the last prediction

    // float K = P / (P + R) -> in fixed: (P << 16) / (P + R) => kalman gain (determines how much new measurement influences prediction)
    int64_t p_shift = (int64_t)P << Q16_SHIFT;
    int32_t K = (int32_t)(p_shift / (P + R));

    // x_pred = x_pred + K * ((float)current_sample - x_pred) => update prediction (main noise reduction step)
    int32_t sample_fixed = (int32_t)current_sample << Q16_SHIFT;
    int32_t diff = sample_fixed - x_pred;
    int64_t mult = ((int64_t)K * diff) >> Q16_SHIFT;
    x_pred = x_pred + (int32_t)mult;

    // P = (1.0f - K) * P -> in fixed: ((65536 - K) * P) >> 16 => reduce uncertainty after including the new measurement
    int32_t diff2 = 65536 - K;
    P = (int32_t)(((int64_t)diff2 * P) >> 16);

    return x_pred >> 16;
}

// Wiener algorithm (FIR-filter), predict future value based on past values
__attribute__((noinline)) int32_t predict_wiener(const int8_t *input_window) {
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += (int32_t)input_window[i] * wiener_weights[i]; // formula Wiener prediction (time domain)
    }
    return sum >> 14; // re-shift
}

// IIR (low-pass) filter (EMA (Exponential Moving Average))
__attribute__((noinline)) int32_t predict_iir(int8_t current_sample) {
    int32_t sample_fixed = (int32_t)current_sample << 8;
    // IIR/EMA filter formula -> converted to fixed-point -> i.e. 0.625 * y = y * 5 / 8 = (5 * y) >> 3
    // using 62.5% signal history and 37.5% new measurement
    iir_state = ((5 * sample_fixed) + (3 * iir_state)) >> 3; 
    return iir_state >> 8;
}