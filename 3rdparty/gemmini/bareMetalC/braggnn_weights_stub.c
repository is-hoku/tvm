// Weight stub file - include actual braggnn.h weights
// This file provides weight definitions for the standalone x86 version

#include <stdint.h>

typedef int8_t elem_t;
typedef int32_t acc_t;

// Include the weight definitions from braggnn.h
#define INPUT_DIM 11
#define INPUT_CHANNELS 1
#define CONV1_FILTERS 64
#define CONV1_KERNEL 3
#define CONV1_DIM 9
#define CONV2_FILTERS 32
#define CONV2_KERNEL 3
#define CONV2_DIM 7
#define CONV3_FILTERS 8
#define CONV3_KERNEL 3
#define CONV3_DIM 5
#define CONV3_FLATTENED (CONV3_DIM * CONV3_DIM * CONV3_FILTERS)
#define FC1_UNITS 16
#define FC2_UNITS 8
#define FC3_UNITS 4
#define FC4_UNITS 2
#define OUTPUT_UNITS 2

// Include weight arrays from braggnn.h
// This requires the actual braggnn.h file
#include "braggnn.h"