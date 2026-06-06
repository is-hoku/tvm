// See LICENSE for license details.
// BraggNN Model Implementation for Baremetal Environment

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// BraggNN Model Architecture Configuration
// Input: 11x11 patches, 3 Conv layers + NLB, 4 FC layers
#define INPUT_DIM 11     // Input patch size (11x11)
#define INPUT_CHANNELS 1 // Grayscale input

#define BATCH 1

// Convolutional layers configuration
// 3 CNN layers: cnn_out_chs = (64, 32, 8), cnn_in_chs = (1, 64, 32)
#define CONV1_FILTERS 64 // Conv1: 11x11x1 -> 9x9x64
#define CONV1_KERNEL 3   // 3x3 kernel
#define CONV1_DIM 9
#define CONV1_2D CONV1_DIM *CONV1_DIM
#define CONV1_CHANNELS 1
#define CONV2_FILTERS 32 // Conv2: 9x9x64 -> 7x7x32
#define CONV2_KERNEL 3   // 3x3 kernel
#define CONV2_DIM 7
#define CONV2_CHANNELS 1
#define CONV3_FILTERS 8 // Conv3: 7x7x32 -> 5x5x8
#define CONV3_KERNEL 3  // 3x3 kernel
#define CONV3_DIM 5
#define CONV3_2D CONV3_DIM *CONV3_DIM
#define CONV3_FLATTENED CONV3_2D *CONV3_FILTERS
#define CONV3_CHANNELS 1

// Fully connected layers (fcsz=(16, 8, 4, 2))
#define FC1_UNITS 16   // First FC layer units
#define FC2_UNITS 8    // Second FC layer units
#define FC3_UNITS 4    // Third FC layer units
#define FC4_UNITS 2    // Fourth FC layer units
#define OUTPUT_UNITS 2 // Output predictions (x, y coordinates of peak center)

// Pre-trained weights (to be populated with actual trained values)
static elem_t conv1_weights[CONV1_FILTERS][CONV1_KERNEL][CONV1_KERNEL]
                           [INPUT_CHANNELS];
static acc_t conv1_bias[CONV1_FILTERS];
static elem_t conv2_weights[CONV2_FILTERS][CONV2_KERNEL][CONV2_KERNEL]
                           [CONV1_FILTERS];
static acc_t conv2_bias[CONV2_FILTERS];
static elem_t conv3_weights[CONV3_FILTERS][CONV3_KERNEL][CONV3_KERNEL]
                           [CONV2_FILTERS];
static acc_t conv3_bias[CONV3_FILTERS];

// Non-Local Block weights (theta, phi, g)
static elem_t nlb_theta_weights[CONV2_FILTERS][CONV1_FILTERS];
static acc_t nlb_theta_bias[CONV2_FILTERS];
static elem_t nlb_phi_weights[CONV2_FILTERS][CONV1_FILTERS];
static acc_t nlb_phi_bias[CONV2_FILTERS];
static elem_t nlb_g_weights[CONV2_FILTERS][CONV1_FILTERS];
static acc_t nlb_g_bias[CONV2_FILTERS];
static elem_t nlb_out_weights[CONV1_FILTERS][CONV2_FILTERS];
static acc_t nlb_out_bias[CONV1_FILTERS];

// FC layer weights - adjusted for flattened conv output
// After 3 conv layers: final feature map is 5x5x8 = 200 features
static elem_t fc1_weights[FC1_UNITS][CONV3_FLATTENED];
static acc_t fc1_bias[FC1_UNITS];
static elem_t fc2_weights[FC2_UNITS][FC1_UNITS];
static acc_t fc2_bias[FC2_UNITS];
static elem_t fc3_weights[FC3_UNITS][FC2_UNITS];
static acc_t fc3_bias[FC3_UNITS];
static elem_t fc4_weights[FC4_UNITS][FC3_UNITS];
static acc_t fc4_bias[FC4_UNITS];
static elem_t output_weights[OUTPUT_UNITS][FC4_UNITS];
static acc_t output_bias[OUTPUT_UNITS];

// LeakyReLU activation function (negative_slope=0.01)
// For int8, we approximate 0.01 as 1/100 ≈ 1/128 (right shift by 7)
static inline elem_t leaky_relu_activation(elem_t x) {
  return x > 0 ? x : (x >> 7); // Approximate 0.01 with 1/128 for int8
}

// Initialize weights with sample values (replace with actual trained weights)
void init_weights() {
  // Initialize conv1 weights (64 filters, 3x3 kernel)
  for (int f = 0; f < CONV1_FILTERS; f++) {
    for (int kr = 0; kr < CONV1_KERNEL; kr++) {
      for (int kc = 0; kc < CONV1_KERNEL; kc++) {
        for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
          // Use range -10 to 10 for int8 weights
          conv1_weights[f][kr][kc][ch] = (elem_t)((f + kr + kc + ch) % 21 - 10);
        }
      }
    }
    conv1_bias[f] = 1; // Small non-zero bias
  }

  // printf("conv1 initialized\n");
  //  Initialize conv2 weights (32 filters, 3x3 kernel)
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int kr = 0; kr < CONV2_KERNEL; kr++) {
      for (int kc = 0; kc < CONV2_KERNEL; kc++) {
        for (int ch = 0; ch < CONV1_FILTERS; ch++) {
          conv2_weights[f][kr][kc][ch] = (elem_t)((f + kr + kc + ch) % 21 - 10);
        }
      }
    }
    conv2_bias[f] = 1;
  }

  // printf("conv2 initialized\n");
  //  Initialize conv3 weights (8 filters, 3x3 kernel)
  for (int f = 0; f < CONV3_FILTERS; f++) {
    for (int kr = 0; kr < CONV3_KERNEL; kr++) {
      for (int kc = 0; kc < CONV3_KERNEL; kc++) {
        for (int ch = 0; ch < CONV2_FILTERS; ch++) {
          conv3_weights[f][kr][kc][ch] = (elem_t)((f + kr + kc + ch) % 21 - 10);
        }
      }
    }
    conv3_bias[f] = 1;
  }

  // printf("conv3 initialized\n");
  //  Initialize NLB weights (Non-Local Block)
  //  Theta weights (32 filters, 1x1 kernel, input channels = 64)
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int ch = 0; ch < CONV1_FILTERS; ch++) {
      nlb_theta_weights[f][ch] = (elem_t)((f + ch) % 21 - 10);
    }
    nlb_theta_bias[f] = 1;
  }

  // printf("nlb_theta initialized\n");
  //  Phi weights (32 filters, 1x1 kernel, input channels = 64)
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int ch = 0; ch < CONV1_FILTERS; ch++) {
      nlb_phi_weights[f][ch] = (elem_t)((f + ch) % 21 - 10);
    }
    nlb_phi_bias[f] = 1;
  }

  // printf("nlb_phi initialized\n");
  //  G weights (32 filters, 1x1 kernel, input channels = 64)
  for (int f = 0; f < CONV2_FILTERS; f++) {
    for (int ch = 0; ch < CONV1_FILTERS; ch++) {
      nlb_g_weights[f][ch] = (elem_t)((f + ch) % 21 - 10);
    }
    nlb_g_bias[f] = 1;
  }

  // printf("nlb_g initialized\n");
  //  Out conv weights (64 filters, 1x1 kernel, input channels = 32)
  for (int f = 0; f < CONV1_FILTERS; f++) {
    for (int ch = 0; ch < CONV2_FILTERS; ch++) {
      nlb_out_weights[f][ch] = (elem_t)((f + ch) % 21 - 10);
    }
    nlb_out_bias[f] = 1;
  }

  // printf("nlb_out initialized\n");
  //  Initialize FC1 weights (16 units, input: 5*5*8 = 200)
  for (int i = 0; i < FC1_UNITS; i++) {
    for (int j = 0; j < CONV3_FLATTENED; j++) {
      fc1_weights[i][j] = (elem_t)((i + j) % 21 - 10);
    }
    fc1_bias[i] = 1;
  }

  // printf("fc1 initialized\n");
  //  Initialize FC2 weights (8 units)
  for (int i = 0; i < FC2_UNITS; i++) {
    for (int j = 0; j < FC1_UNITS; j++) {
      fc2_weights[i][j] = (elem_t)((i + j) % 21 - 10);
    }
    fc2_bias[i] = 1;
  }

  // printf("fc2 initialized\n");
  //  Initialize FC3 weights (4 units)
  for (int i = 0; i < FC3_UNITS; i++) {
    for (int j = 0; j < FC2_UNITS; j++) {
      fc3_weights[i][j] = (elem_t)((i + j) % 21 - 10);
    }
    fc3_bias[i] = 1;
  }

  // printf("fc3 initialized\n");
  //  Initialize FC4 weights (2 units)
  for (int i = 0; i < FC4_UNITS; i++) {
    for (int j = 0; j < FC3_UNITS; j++) {
      fc4_weights[i][j] = (elem_t)((i + j) % 21 - 10);
    }
    fc4_bias[i] = 1;
  }

  // printf("fc4 initialized\n");
  //  Initialize output weights (2 outputs for x,y coordinates)
  for (int i = 0; i < OUTPUT_UNITS; i++) {
    for (int j = 0; j < FC4_UNITS; j++) {
      output_weights[i][j] = (elem_t)((i + j) % 21 - 10);
    }
    output_bias[i] = 1;
  }
  // printf("output initialized\n");
}

int main() {
#if defined(FAST) || !defined(HAS_NORMALIZATIONS)
  exit(0);
#endif
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("BraggNN Model Inference on Gemmini\n");
  printf("===================================\n\n");

  printf("Flushing Gemmini TLB\n");
  gemmini_flush(0);

  // Initialize weights (replace with actual trained weights)
  printf("Initializing model weights\n");
  init_weights();

  // Create sample input (11x11 grayscale patch)
  // Convert float values (0.0-1.0) to int8 (0-127)
  printf("Creating sample input data (11x11 patch)\n");
  elem_t input[INPUT_DIM * INPUT_DIM * INPUT_CHANNELS] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73,
      89, 42, 0, 0, 0,                      // 0.575659*127≈73, 0.696713*127≈89,
                                            // 0.333225*127≈42
      0, 0, 0, 0, 0, 39,                    // 0.303938*127≈39
      105, 127, 47, 0, 0, 0,                // 0.822649*127≈105, 1.0*127=127,
                                            // 0.373902*127≈47
      0, 0, 0, 0, 0, 37,                    // 0.291246*127≈37
      84, 91, 48, 0, 0, 0,                  // 0.658965*127≈84, 0.713635*127≈91,
                                            // 0.379434*127≈48
      0, 0, 0, 0, 0, 0, 41, 33, 0, 0, 0, 0, // 0.322161*127≈41,
                                            // 0.260657*127≈33
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0};

  static elem_t nlb_theta_reshaped[CONV2_FILTERS][CONV1_2D];
  static elem_t nlb_phi_reshaped[CONV2_FILTERS][CONV1_2D];
  static elem_t attention_matrix[CONV1_2D][CONV1_2D];

  // Initialize nlb_theta_reshaped and nlb_g_reshaped with test data
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int idx = 0; idx < CONV1_2D; idx++) {
      nlb_theta_reshaped[c][idx] = (elem_t)((c + idx) % 21 - 10);
      nlb_phi_reshaped[c][idx] = (elem_t)((c * 2 + idx) % 21 - 10);
    }
  }
  printf("Testing matmul with SOFTMAX\n");
  printf("Matrix dimensions: %d x %d x %d\n", CONV1_2D, CONV1_2D,
         CONV2_FILTERS);

  tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
                    (elem_t *)nlb_theta_reshaped,      // A [32x81]
                    (elem_t *)nlb_phi_reshaped,        // B [32x81]
                    NULL,                              // D (bias)
                    (elem_t *)attention_matrix,        // C [81x81]
                    CONV2_FILTERS, CONV1_2D, CONV1_2D,
                    CONV1_2D,            // strides (A, B, D, C)
                    MVIN_SCALE_IDENTITY, // A_scale_factor
                    MVIN_SCALE_IDENTITY, // B_scale_factor
                    MVIN_SCALE_IDENTITY, // D_scale_factor
                    SOFTMAX,             // act (softmax)
                    ACC_SCALE_IDENTITY,  // scale
                    0.05,                // bert_scale
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    WS                   // Use WS mode
  );

  printf("Matmul with SOFTMAX completed successfully\n");

  exit(0);
}
