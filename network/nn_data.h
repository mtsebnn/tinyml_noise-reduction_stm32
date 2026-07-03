
#ifndef NN_DATA_H_
#define NN_DATA_H_

#include <stdint.h> // declares sets of integer types with specific widths e.g. int8_t

#define WINDOW_SIZE 16
#define HIDDEN_NEURONS 8

typedef struct {
    const char *name;
    // w1 is a pointer to an array of WINDOW_SIZE int8_t elements
    // C array decay: In most expressions array "decays" into a pointer to its first element (i.e. 2D-array becomes a pointer to its first row)
    // i.e. w1 has type int8_t [HIDDEN_NEURONS][WINDOW_SIZE] -> decays to -> int8_t (*) [WINDOW_SIZE]
    // i.e. we are going to use: data.w1 = w1 = &w1[0] 
    // (means that w1 is a pointer(stores the address) to the first row of the matrix w1_only_whitenoise)
    // w1[2][1] = *(*(w1 + 2) + 1) -> value at row 3, column 2
    const int8_t (*w1)[WINDOW_SIZE]; 
    const int32_t *b1;
    const int8_t *w2; // w2 is a pointer to a single int8_t element (ptr[0] = *(ptr + 0) = ptr*)
    const int32_t b2;
    const int32_t requant_multiplier;
    const uint8_t shift_value;
} nn_data_t;

// data
static const int8_t w1[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {49, 20, -5, -17, -26, -31, -60, -69, -53, -58, -54, -30, -13, 20, 40, 49},
  {86, 73, 65, 5, -12, -36, -62, -86, -87, -76, -56, -40, -36, -12, 15, 35},
  {-77, -70, -53, -36, -6, 20, 39, 49, 67, 70, 80, 73, 58, 40, 12, -11},
  {77, 56, 2, -38, -73, -99, -111, -118, -127, -104, -66, -36, 5, 36, 71, 101},
  {-89, -77, -60, -26, 21, 23, 51, 82, 97, 96, 92, 80, 53, 33, 5, -46},
  {-2, 5, -8, 9, 5, -1, 2, -5, 0, 7, 11, 11, -10, 3, 0, -7},
  {44, 57, 84, 66, 61, 22, 4, -19, -49, -39, -35, -46, -54, -47, -43, -33},
  {15, -17, -27, -47, -71, -78, -55, -65, -52, -41, -15, 19, 30, 44, 61, 68}
};
static const int32_t b1[HIDDEN_NEURONS] = {8033, 6344, 11052, -16721, -18860, -2964, -17030, 10862};
static const int8_t w2[HIDDEN_NEURONS] = {-62, -84, 102, 127, -88, 12, 39, -104};
#define B2 2303
#define REQUANT_MULT 2292
#define SHIFT_VALUE 20


// global config for main.c
extern const nn_data_t data;

int32_t predict_nn(const int8_t *input_window, const nn_data_t *config);

#endif /* NN_DATA_H_ */
