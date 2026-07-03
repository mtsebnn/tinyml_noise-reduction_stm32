#ifndef FILTERS_H_
#define FILTERS_H_

#include <stdint.h>

void reset_filters(void);
__attribute__((noinline)) int32_t predict_kalman(int8_t current_sample);
__attribute__((noinline)) int32_t predict_iir(int8_t current_sample);
__attribute__((noinline)) int32_t predict_wiener(const int8_t *input_window);

#endif