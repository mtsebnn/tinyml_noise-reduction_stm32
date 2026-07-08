
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
    const int16_t (*w1)[WINDOW_SIZE]; 
    const int32_t *b1;
    const int16_t *w2; // w2 is a pointer to a single int8_t element (ptr[0] = *(ptr + 0) = ptr*)
    const int32_t b2;
    const int32_t requant_multiplier;
    const int64_t residual_multiplier;
    const uint8_t shift_value;
} nn_data_t;

// data
static const int16_t w1[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {-4056, -3981, 307, 2996, 7772, 9202, 7896, 9498, -2103, 9718, 6570, 4933, -469, -4092, -5319, -6480},
  {6748, 6194, 1582, -633, -4916, -7492, -10439, -9796, -18654, -9367, -8482, -6015, -1927, 863, 4050, 6982},
  {-5014, -7913, -5886, -5990, -2544, -1893, 1494, 756, 11110, 9331, 6497, 9497, 7925, 5772, 4989, 1340},
  {634, -2886, -5100, -8003, -6139, -6998, -8966, -6028, -11425, -2229, 316, 1215, 6221, 4842, 7521, 7002},
  {-3919, -1644, 3287, 6710, 8986, 10887, 12133, 10257, 16326, 8906, 6843, 2728, -1644, -6605, -7650, -9172},
  {-3883, -3546, -459, 1095, 1754, 3788, 5969, 4420, -32767, 5722, 4131, 3376, 1681, 678, -1870, -3631},
  {9568, 7438, 4619, -230, -2296, -6479, -10282, -11171, -7706, -11259, -10314, -7646, -6037, -1361, 2599, 5458},
  {10113, 7673, 7205, 4182, 2381, -2076, -5681, -8997, -9175, -9870, -10544, -9217, -9297, -5285, -2213, 1340}
};
static const int32_t b1[HIDDEN_NEURONS] = {944799, -1308707, -1315126, -1286149, -1463545, 1631852, 1224749, -1372097};
static const int16_t w2[HIDDEN_NEURONS] = {-18708, -24365, 19361, -16920, 26662, -31198, 32767, -25863};
#define B2 461093
#define REQUANT_MULT 17
#define RESIDUAL_MULT 28480690953
#define SHIFT_VALUE 20


// global config for main.c
extern const nn_data_t data;

int32_t predict_nn(const int8_t *input_window, const nn_data_t *config);

#endif /* NN_DATA_H_ */
