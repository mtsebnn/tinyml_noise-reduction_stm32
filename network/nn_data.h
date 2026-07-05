
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
    const int32_t residual_multiplier;
    const uint8_t shift_value;
} nn_data_t;

// data
static const int16_t w1[HIDDEN_NEURONS][WINDOW_SIZE] = {
  {-5944, -4837, -3646, -3039, -3086, -1658, -504, 1459, -1731, 4486, 4592, 4563, 5224, 4035, 3113, 1961},
  {11344, 8711, 5802, 3639, 436, -3717, -7872, -8907, -13605, -10422, -10681, -10199, -6723, -3598, 388, 4332},
  {7789, 7423, 6125, 4753, 2128, -239, -4077, -5250, -2103, -8105, -8897, -8416, -6568, -5695, -1886, 1630},
  {-2512, -1755, -625, 1484, 2674, 3658, 5246, 5560, -32767, 2148, 3560, 1546, 848, -1277, -1744, -3239},
  {-10525, -7235, -2584, 2099, 5495, 8169, 11809, 12664, 16362, 12421, 11251, 8014, 2283, 29, -5663, -8000},
  {-8041, -3256, -1369, 2384, 6507, 7409, 8924, 10195, -1966, 11040, 6357, 5171, 529, -1807, -5587, -7713},
  {2362, -491, -2834, -5500, -6228, -8793, -9320, -8133, -1918, -5174, -3101, -481, 2189, 4880, 7065, 7229},
  {3055, 2813, -3391, -5509, -7455, -8751, -11651, -10658, -14201, -4258, -5231, -2891, 2751, 4392, 7262, 9139}
};
static const int32_t b1[HIDDEN_NEURONS] = {554555, -1312492, 986022, 1582286, -1396585, 809965, 993101, -1364356};
static const int16_t w2[HIDDEN_NEURONS] = {-16817, -29993, 27947, -32471, 32767, -19821, 27239, -25106};
#define B2 520870
#define REQUANT_MULT 22
#define RESIDUAL_MULT 37837060936
#define SHIFT_VALUE 20


// global config for main.c
extern const nn_data_t data;

int32_t predict_nn(const int8_t *input_window, const nn_data_t *config);

#endif /* NN_DATA_H_ */
