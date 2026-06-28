
#ifndef NN_DATA_H_
#define NN_DATA_H_

#include <stdint.h> // declares sets of integer types with specific widths e.g. int8_t

#define WINDOW_SIZE 16
#define HIDDEN_NEURONS 8

typedef struct {
    const char *name;
    // w1 is a pointer to an array of WINDOW_SIZE int8_t elements
    // C array decay: In most expressions array "decays" into a pointer to its first element (i.e. 2D-array becomes a pointer to its first row)
    // i.e. w1_only_whitenoise has type int8_t [HIDDEN_NEURONS][WINDOW_SIZE] -> decays to -> int8_t (*) [WINDOW_SIZE]
    // i.e. we are going to use: scenario_whitenoise.w1 = w1_only_whitenoise = &w1_only_whitenoise[0] 
    // (means that w1 is a pointer(stores the address) to the first row of the matrix w1_only_whitenoise)
    // w1[2][1] = *(*(w1 + 2) + 1) -> value at row 3, column 2
    const int8_t (*w1)[WINDOW_SIZE]; 
    const int32_t *b1;
    const int8_t *w2; // w2 is a pointer to a single int8_t element (ptr[0] = *(ptr + 0) = ptr*)
    const int32_t b2;
    const int32_t requant_multiplier;
    const uint8_t shift_value;
} nn_scenario_t;

// scenario: only_whitenoise
static const int8_t w1_only_whitenoise[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {51, 29, -21, -60, -88, -82, -119, -61, -41, -74, -35, -22, 51, 49, 83, 86},
  {127, 83, 21, 55, -11, -17, -75, -47, -56, -102, -76, -54, -67, -74, 24, 23},
  {-60, -51, -36, -6, 21, 52, 75, 93, 108, 87, 79, 47, 25, -16, -36, -55},
  {103, 75, 52, 56, -4, -59, -66, -85, -93, -110, -96, -113, -84, -66, -13, 23},
  {-38, -35, 109, -89, -72, -111, 46, 91, 115, -4, -93, -74, -70, 80, -33, 44},
  {20, 29, 110, 32, 49, 20, 10, -101, -70, -34, -33, -44, -2, 52, -68, -7},
  {67, 17, 2, -56, -86, -88, -107, -107, -82, -82, -56, -45, 23, 65, 80, 82},
  {60, -15, -1, -50, -55, -116, -64, -124, -114, -66, -88, -93, -63, 30, 48, 69}
};
static const int32_t b1_only_whitenoise[HIDDEN_NEURONS] = {19459, 17987, 25471, -35276, -3975, 11366, -30071, 16285};
static const int8_t w2_only_whitenoise[HIDDEN_NEURONS] = {-74, -90, 86, 123, 2, -38, 127, -71};
#define B2_ONLY_WHITENOISE 4623
#define REQUANT_MULT_ONLY_WHITENOISE 102
#define SHIFT_VALUE_ONLY_WHITENOISE 16


// scenario: only_echo
static const int8_t w1_only_echo[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {22, 2, -15, 9, -14, -59, -60, -46, -48, -31, -26, -83, -47, 40, 65, 48},
  {-14, -13, -26, 8, 52, 33, 9, -65, -53, -32, 25, -18, -40, -8, -55, -28},
  {61, 0, 11, -33, 33, -12, -37, -94, -109, -100, -20, -33, -13, -24, -12, 41},
  {-27, -32, -36, 33, -13, 2, -17, 3, -49, -46, 5, -28, 19, -29, -50, -21},
  {31, 38, 2, 73, -4, 0, -3, 15, -58, 21, -38, 7, 39, -11, 14, 28},
  {-106, -41, -122, -26, 69, 95, 40, 14, 70, 89, 10, 51, 31, 35, -70, -76},
  {-95, -127, -27, 50, 43, 60, 51, 28, 26, 98, 44, 29, 34, -71, -115, -99},
  {9, 3, -4, -21, 65, -1, 6, -35, -15, 28, 3, -66, -1, 27, -9, 13}
};
static const int32_t b1_only_echo[HIDDEN_NEURONS] = {4142, -1060, -5885, -6106, -7917, 7939, -7600, -7037};
static const int8_t w2_only_echo[HIDDEN_NEURONS] = {-90, -8, -127, -45, 38, 101, 56, -36};
#define B2_ONLY_ECHO -970
#define REQUANT_MULT_ONLY_ECHO 196
#define SHIFT_VALUE_ONLY_ECHO 16


// scenario: only_quantization
static const int8_t w1_only_quantization[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {57, -63, 46, -48, 26, 58, 1, 62, -58, -26, 40, -15, -14, -29, 47, 34},
  {46, -8, -37, 3, -18, -43, -66, -33, -42, -72, -41, -92, -81, -28, 3, 32},
  {-79, -83, -1, 18, 73, 5, 34, 93, 35, 44, 56, 31, 15, 46, 30, -28},
  {16, -26, -3, 2, 2, -22, -83, -47, -108, -37, -54, -35, -34, 23, 10, 63},
  {-68, -6, 11, 3, -5, 108, 119, 23, 81, 31, 10, 72, 73, -22, -64, 10},
  {-4, -43, -2, 22, 59, -7, 43, -29, -57, -56, 80, 61, 60, 34, 17, -50},
  {-26, 51, -12, -35, -31, 56, -8, -115, -96, -70, -51, -60, 15, -23, 71, 24},
  {-11, -6, -8, 14, 20, 47, 80, 127, 54, 69, 15, -4, 35, 37, 27, -15}
};
static const int32_t b1_only_quantization[HIDDEN_NEURONS] = {-6408, 3288, -4201, 3065, -1403, -9263, 11661, -10066};
static const int8_t w2_only_quantization[HIDDEN_NEURONS] = {-92, -113, 103, -127, 87, -7, -58, 81};
#define B2_ONLY_QUANTIZATION 2949
#define REQUANT_MULT_ONLY_QUANTIZATION 136
#define SHIFT_VALUE_ONLY_QUANTIZATION 16


// scenario: combined
static const int8_t w1_combined[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {25, 1, 23, 26, 5, 9, 9, 26, -20, -22, -9, -17, -19, 13, -27, -18},
  {-108, -106, -77, -45, -4, 21, 48, 80, 86, 96, 100, 108, 77, 39, 22, -3},
  {45, 37, 55, 24, 37, 6, -2, -6, -16, -18, -27, -45, -48, -39, -27, -14},
  {-115, -86, -33, 33, 72, 86, 107, 127, 119, 116, 96, 76, 27, -3, -48, -84},
  {-68, -29, -16, 13, 62, 55, 59, 45, 90, 82, 51, 45, 27, 3, -15, -80},
  {31, 46, 24, 57, 31, 14, 10, -32, -18, -33, -35, -41, -42, -27, -40, -56},
  {-56, -10, 6, 64, 79, 81, 104, 103, 82, 69, 25, 3, -27, -55, -100, -103},
  {-18, -43, -49, -55, -78, -54, -55, -33, -36, -16, 6, 30, 60, 37, 77, 82}
};
static const int32_t b1_combined[HIDDEN_NEURONS] = {2956, 11966, 7801, -17224, 6401, 8204, 10346, 11578};
static const int8_t w2_combined[HIDDEN_NEURONS] = {-14, 80, -41, -127, 28, -50, 87, -59};
#define B2_COMBINED -1655
#define REQUANT_MULT_COMBINED 121
#define SHIFT_VALUE_COMBINED 16


// global scenario config for main.c
extern const nn_scenario_t scenario_whitenoise;
extern const nn_scenario_t scenario_echo;
extern const nn_scenario_t scenario_quantization;
extern const nn_scenario_t scenario_combined;

int32_t predict(const int8_t *input_window, const nn_scenario_t *config);

#endif /* NN_DATA_H_ */
