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

void flatten_weights(
    int out_channels, int kernel_dim, int in_channels, int patch_size,
    elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
    elem_t weights_mat[patch_size][out_channels]) {

  assert(patch_size == kernel_dim * kernel_dim * in_channels);

  for (int outc = 0; outc < out_channels; outc++) {
    for (int krow = 0; krow < kernel_dim; krow++) {
      for (int kcol = 0; kcol < kernel_dim; kcol++) {
        for (int inc = 0; inc < in_channels; inc++) {
          int wmatrow =
              krow * kernel_dim * in_channels + kcol * in_channels + inc;
          weights_mat[wmatrow][outc] = weights[outc][krow][kcol][inc];
        }
      }
    }
  }
}

//
//
// CPU functions
//
//

// void cpu_conv2d(
//     int batch_size, int in_channels, int in_row_dim, int in_col_dim,
//     int out_channels, int kernel_dim, int out_row_dim, int out_col_dim,
//     int stride, int padding,
//     elem_t input[batch_size][in_row_dim][in_col_dim][in_channels],
//     elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
//     acc_t bias[out_channels],
//     elem_t output[batch_size][out_row_dim][out_col_dim][out_channels]) {
//
// #ifdef GEMMINI_ASSERTIONS
//   if (out_row_dim != (in_row_dim + 2 * padding - kernel_dim) / stride + 1) {
//     printf("conv out_row_dim is not correct\n");
//     exit(1);
//   }
//
//   if (out_col_dim != (in_col_dim + 2 * padding - kernel_dim) / stride + 1) {
//     printf("conv out_col_dim is not correct\n");
//     exit(1);
//   }
// #endif
//
//   for (int b = 0; b < batch_size; b++) {
//     for (int orow = 0; orow < out_row_dim; orow++) {
//       for (int ocol = 0; ocol < out_col_dim; ocol++) {
//         for (int och = 0; och < out_channels; och++) {
//           acc_t result = bias[och];
//
//           for (int krow = 0; krow < kernel_dim; krow++) {
//             for (int kcol = 0; kcol < kernel_dim; kcol++) {
//               for (int kch = 0; kch < in_channels; kch++) {
//                 int irow = orow * stride + krow - padding;
//                 int icol = ocol * stride + kcol - padding;
//
//                 elem_t pixel = irow < 0 || irow >= in_row_dim || icol < 0 ||
//                                        icol >= in_col_dim
//                                    ? 0
//                                    : input[b][irow][icol][kch];
//
//                 result += weights[och][krow][kcol][kch] * pixel;
//               }
//             }
//           }
//
//           // Clip result
//           result = result > elem_t_max
//                        ? elem_t_max
//                        : (result < elem_t_min ? elem_t_min : result);
//
//           output[b][orow][ocol][och] = result;
//         }
//       }
//     }
//   }
// }

// Fully connected layer using Gemmini with LeakyReLU
void cpu_fc(int batch, int in_features, int out_features, elem_t *input,
            elem_t *weights, acc_t *bias, elem_t *output) {
  // Create temporary buffer for transposed weights
  // static elem_t weights_transposed[200]
  //                                [64]; // Maximum size for largest FC layer

  // Manual transpose: weights[out_features][in_features] ->
  // weights_transposed[in_features][out_features]
  // for (int i = 0; i < out_features; i++) {
  //  for (int j = 0; j < in_features; j++) {
  //    weights_transposed[j][i] = ((elem_t(*)[in_features])weights)[i][j];
  //  }
  //}

  // Use WS mode with manually transposed weights
  // tiled_matmul_auto(batch, out_features, in_features, (elem_t *)input,
  //                  (elem_t *)weights_transposed, bias, output, in_features,
  //                  in_features, out_features, out_features,
  //                  MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                  MVIN_SCALE_IDENTITY, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
  //                  false, false, false, false, false, 0, CPU);
  tiled_matmul_auto(batch, out_features, in_features, (elem_t *)input,
                    (elem_t *)weights, bias, output, in_features, in_features,
                    out_features, out_features, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION,
                    ACC_SCALE_IDENTITY, 0, false, false, true, false, false, 0,
                    CPU);
}

// Non-Local Block implementation
void cpu_nlb(elem_t *input, elem_t *output) {
  // Theta path: 1x11x11x64 -> conv -> 1x9x9x32
  static elem_t nlb_theta_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                             [CONV2_FILTERS];

  tiled_conv_auto(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_CHANNELS,
                  CONV2_DIM, CONV2_DIM, 1, 1, 1, 0, CONV1_KERNEL, false, false,
                  false, false, false, input, (elem_t *)nlb_theta_weights,
                  nlb_theta_bias, (elem_t *)nlb_theta_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // printf("Conv theta \n");

  // Reshape theta from 1x9x9x32 to 1x32x81 then transpose to 1x81x32
  static elem_t nlb_theta_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  // static elem_t nlb_theta_transposed[81][32]; // 81x32 format (transposed)
  static elem_t identity_32x32[CONV2_FILTERS]
                              [CONV2_FILTERS]; // Identity matrix for transpose

  // Initialize identity matrix
  for (int i = 0; i < CONV2_FILTERS; i++) {
    for (int j = 0; j < CONV2_FILTERS; j++) {
      identity_32x32[i][j] = (i == j) ? 1 : 0;
    }
  }

  // printf("Init identity matrix \n");

  // Reshape: convert 9x9 spatial dimensions to 81 elements
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w; // Convert 2D (h,w) to 1D index
        nlb_theta_reshaped[c][idx] = nlb_theta_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape theta \n");

  // Phi path: 1x11x11x64 -> conv -> 1x9x9x32 -> reshape to 1x32x81
  static elem_t nlb_phi_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                           [CONV2_FILTERS];
  tiled_conv_auto(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_CHANNELS,
                  CONV2_DIM, CONV2_DIM, 1, 1, 1, 0, CONV1_KERNEL, false, false,
                  false, false, false, input, (elem_t *)nlb_phi_weights,
                  nlb_phi_bias, (elem_t *)nlb_phi_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // printf("Conv phi \n");

  static elem_t nlb_phi_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w;
        nlb_phi_reshaped[c][idx] = nlb_phi_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape phi \n");

  // G path: 1x11x11x64 -> conv -> 1x9x9x32 -> reshape to 1x32x81
  static elem_t nlb_g_out[CONV2_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  tiled_conv_auto(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_CHANNELS,
                  CONV2_DIM, CONV2_DIM, 1, 1, 1, 0, CONV1_KERNEL, false, false,
                  false, false, false, input, (elem_t *)nlb_g_weights,
                  nlb_g_bias, (elem_t *)nlb_g_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // printf("Conv g \n");

  static elem_t nlb_g_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  // static elem_t nlb_g_transposed[81][32]; // 81x32 format (transposed)
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int spatial_idx = h * CONV1_DIM + w;
        nlb_g_reshaped[c][spatial_idx] = nlb_g_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape g \n");

  // Manual transpose for theta: 32x81 -> 81x32
  // static elem_t nlb_theta_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_theta_transposed[j][i] = nlb_theta_reshaped[i][j];
  //  }
  //}

  // Manual transpose for phi: 32x81 -> 81x32
  // static elem_t nlb_phi_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_phi_transposed[j][i] = nlb_phi_reshaped[i][j];
  //  }
  //}

  // Attention computation: theta^T @ phi^T^T = theta^T @ phi = 81x32 @ 32x81 =
  // 81x81 Use OS mode with no transpose flags since we manually transposed
  static elem_t attention_matrix[CONV1_2D][CONV1_2D];
  // tiled_matmul_auto(
  //     CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
  //     (elem_t *)nlb_theta_transposed,    // A [81x32] (manually transposed)
  //     (elem_t *)nlb_phi_reshaped,        // B [32x81] (original)
  //     NULL,                              // D (bias)
  //     (elem_t *)attention_matrix,        // C [81x81]
  //     CONV2_FILTERS, CONV1_2D, CONV1_2D, CONV1_2D, // strides (A, B, D, C)
  //     MVIN_SCALE_IDENTITY,                         // A_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // B_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // D_scale_factor
  //     NO_ACTIVATION,      // NO_ACTIVATION (remove SOFTMAX)
  //     ACC_SCALE_IDENTITY, // scale
  //     0,                  // bert_scale
  //     false,              // repeating_bias
  //     false,              // transpose_A (already done manually)
  //     false,              // transpose_B (use original phi)
  //     false,              // full_C
  //     false,              // low_D
  //     0,                  // weightA
  //     CPU                 // mode
  //);
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
                    NO_ACTIVATION,       // NO_ACTIVATION (remove SOFTMAX)
                    ACC_SCALE_IDENTITY,  // scale
                    0,                   // bert_scale
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    CPU                  // mode
  );

  // printf("Matmul theta and phi \n");

  // Manual transpose for g: 32x81 -> 81x32
  // static elem_t nlb_g_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_g_transposed[j][i] = nlb_g_reshaped[i][j];
  //  }
  //}

  // Apply attention to g: attention @ g^T = 81x81 @ 81x32 = 81x32
  // Use OS mode with no transpose flags since we manually transposed g
  static elem_t attended_output[CONV1_2D][CONV2_FILTERS];
  // tiled_matmul_auto(
  //     CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
  //     (elem_t *)nlb_theta_transposed,    // A [81x32] (manually transposed)
  //     (elem_t *)nlb_phi_reshaped,        // B [32x81] (original)
  //     NULL,                              // D (bias)
  //     (elem_t *)attention_matrix,        // C [81x81]
  //     CONV2_FILTERS, CONV1_2D, CONV1_2D, CONV1_2D, // strides (A, B, D, C)
  //     MVIN_SCALE_IDENTITY,                         // A_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // B_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // D_scale_factor
  //     NO_ACTIVATION,                               // act
  //     ACC_SCALE_IDENTITY,                          // scale
  //     0,                                           // bert_scale
  //     false,                                       // repeating_bias
  //     false,                                       // transpose_A
  //     false, // transpose_B (already done manually)
  //     false, // full_C
  //     false, // low_D
  //     0,     // weightA
  //     CPU    // Use OS mode
  //);
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
                    NO_ACTIVATION,       // act
                    ACC_SCALE_IDENTITY,  // scale
                    0,                   // bert_scale
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    CPU                  // Use CPU mode
  );

  // printf("Matmul g \n");

  // TODO: Transpose
  // printf("Before Conv NLB \n");

  // Reshape back to spatial format: 81x32 -> 1x9x9x32
  static elem_t theta_phi_g_output[CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int idx = 0; idx < CONV1_2D; idx++) {
      int h = idx / CONV1_DIM;
      int w = idx % CONV1_DIM;
      ((elem_t(*)[CONV1_DIM][CONV1_DIM][CONV2_FILTERS])
           theta_phi_g_output)[0][h][w][c] = attended_output[idx][c];
    }
  }

  // printf("Reshape theta, phi, and g \n");

  static elem_t nlb_output[CONV1_DIM][CONV1_DIM][64];
  tiled_conv_auto(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_CHANNELS,
                  CONV2_DIM, CONV2_DIM, 1, 1, 1, 0, CONV1_KERNEL, false, false,
                  false, false, false, theta_phi_g_output,
                  (elem_t *)nlb_out_weights, nlb_out_bias, (elem_t *)nlb_output,
                  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // printf("Conv theta, phi, and g \n");

  resadd_cpu(CONV1_2D, CONV1_FILTERS, 1, MVIN_SCALE_IDENTITY,
             MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, (elem_t *)input,
             (elem_t *)nlb_output, (elem_t *)output, NO_ACTIVATION);

  // printf("Add \n");
}

void cpu_inference(elem_t *input, elem_t *output) {
  // Allocate intermediate buffers with correct feature map sizes
  // 3 CNN layers with Non-Local Block
  static elem_t conv1_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                         [CONV1_FILTERS]; // 11x11 -> 9x9x64 (3x3, no pad)
  static elem_t nlb_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                       [CONV1_FILTERS]; // NLB output: 9x9x64
  static elem_t conv2_out[CONV2_CHANNELS][CONV2_DIM][CONV2_DIM]
                         [CONV2_FILTERS]; // 9x9 -> 7x7x32 (3x3, no pad)
  static elem_t conv3_out[CONV3_KERNEL][CONV3_DIM][CONV3_DIM]
                         [CONV3_FILTERS]; // 7x7 -> 5x5x8 (3x3, no pad)

  static elem_t fc1_out[FC1_UNITS];
  static elem_t fc2_out[FC2_UNITS];
  static elem_t fc3_out[FC3_UNITS];
  static elem_t fc4_out[FC4_UNITS];

  // printf("BraggNN Inference: Starting forward pass \n");
  // printf("  Input: %dx%dx%d patch (Bragg peak)\n", INPUT_DIM, INPUT_DIM,
  //        INPUT_CHANNELS);

  // Conv1: 11x11x1 -> 9x9x64 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 1: Conv2D (3x3, 64 filters) -> 9x9x64\n");
  tiled_conv_auto(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_CHANNELS,
                  CONV1_DIM, CONV1_DIM, 1, 1, 1, 0, CONV1_KERNEL, false, false,
                  false, false, false, input, (elem_t *)conv1_weights,
                  conv1_bias, (elem_t *)conv1_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // Non-Local Block
  // printf("  Non-Local Block \n");
  cpu_nlb((elem_t *)conv1_out, (elem_t *)nlb_out);

  for (int i = 0; i < CONV1_DIM; i++) {
    for (int j = 0; j < CONV1_DIM; j++) {
      for (int k = 0; k < CONV1_FILTERS; k++) {
        nlb_out[0][i][j][k] = leaky_relu_activation(nlb_out[0][i][j][k]);
      }
    }
  }
  // Conv2: 9x9x64 -> 7x7x32 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 2: Conv2D (3x3, 32 filters) + LeakyReLU -> 7x7x32\n");
  tiled_conv_auto(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_CHANNELS,
                  CONV2_DIM, CONV2_DIM, 1, 1, 1, 0, CONV2_KERNEL, false, false,
                  false, false, false, nlb_out, (elem_t *)conv2_weights,
                  conv2_bias, (elem_t *)conv2_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  for (int i = 0; i < CONV2_DIM; i++) {
    for (int j = 0; j < CONV2_DIM; j++) {
      for (int k = 0; k < CONV2_FILTERS; k++) {
        conv2_out[0][i][j][k] = leaky_relu_activation(conv2_out[0][i][j][k]);
      }
    }
  }
  // Conv3: 7x7x32 -> 5x5x8 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 3: Conv2D (3x3, 8 filters) + LeakyReLU -> 5x5x8\n");
  tiled_conv_auto(BATCH, CONV2_DIM, CONV2_DIM, CONV2_CHANNELS, CONV3_CHANNELS,
                  CONV3_DIM, CONV3_DIM, 1, 1, 1, 0, CONV3_KERNEL, false, false,
                  false, false, false, conv2_out, (elem_t *)conv3_weights,
                  conv3_bias, (elem_t *)conv3_out, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, 0, 0, 0, CPU);

  // Flatten: 5x5x8 -> 200
  elem_t flattened[CONV3_FLATTENED];
  int idx = 0;
  for (int r = 0; r < CONV3_DIM; r++) {
    for (int c = 0; c < CONV3_DIM; c++) {
      for (int ch = 0; ch < CONV3_FILTERS; ch++) {
        flattened[idx++] = conv3_out[0][r][c][ch];
      }
    }
  }

  for (int i = 0; i < CONV3_FLATTENED; i++) {
    flattened[i] = leaky_relu_activation(flattened[i]);
  }
  // printf("  Layer 4: FC (200 -> 16) + LeakyReLU\n");
  cpu_fc(CONV3_CHANNELS, CONV3_FLATTENED, FC1_UNITS, flattened,
         (elem_t *)fc1_weights, fc1_bias, fc1_out);

  for (int i = 0; i < FC1_UNITS; i++) {
    fc1_out[i] = leaky_relu_activation(fc1_out[i]);
  }
  // printf("  Layer 5: FC (16 -> 8) + LeakyReLU\n");
  cpu_fc(1, FC1_UNITS, FC2_UNITS, fc1_out, (elem_t *)fc2_weights, fc2_bias,
         fc2_out);

  for (int i = 0; i < FC2_UNITS; i++) {
    fc2_out[i] = leaky_relu_activation(fc2_out[i]);
  }
  // printf("  Layer 6: FC (8 -> 4) + LeakyReLU\n");
  cpu_fc(1, FC2_UNITS, FC3_UNITS, fc2_out, (elem_t *)fc3_weights, fc3_bias,
         fc3_out);

  for (int i = 0; i < FC3_UNITS; i++) {
    fc3_out[i] = leaky_relu_activation(fc3_out[i]);
  }
  // printf("  Layer 7: FC (4 -> 2) + LeakyReLU\n");
  cpu_fc(1, FC3_UNITS, FC4_UNITS, fc3_out, (elem_t *)fc4_weights, fc4_bias,
         fc4_out);

  for (int i = 0; i < FC4_UNITS; i++) {
    fc4_out[i] = leaky_relu_activation(fc4_out[i]);
  }
  // Output: 8 -> 2 (x, y coordinates of peak center) - Linear (no activation)
  // printf("  Layer 8: Output FC (2 -> 2) [Linear]\n");
  cpu_fc(1, FC4_UNITS, OUTPUT_UNITS, fc4_out, (elem_t *)output_weights,
         output_bias, output);

  // printf("Gemmini Inference: Complete\n");
}

//
//
// Gemmini functions
//
//

// Convolution operation using Gemmini's tiled_conv_auto
void gemmini_conv2d(int batch, int in_rows, int in_cols, int in_channels,
                    int out_channels, int kernel_dim, int stride, int padding,
                    elem_t *input, elem_t *weights, acc_t *bias,
                    elem_t *output) {

  int out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
  int out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;
  int patch_size = kernel_dim * kernel_dim * in_channels;

  // Allocate memory for flattened weights (2D array format)
  elem_t weights_mat[patch_size][out_channels];

  // Flatten the weights for tiled_conv_auto
  flatten_weights(out_channels, kernel_dim, in_channels, patch_size,
                  (elem_t(*)[kernel_dim][kernel_dim][in_channels])weights,
                  weights_mat);

  // Use Gemmini's optimized convolution function
  tiled_conv_auto(batch, in_rows, in_cols, in_channels, out_channels, out_rows,
                  out_cols, stride, 1, 1, padding, kernel_dim, false, false,
                  false, false, false, input, (elem_t *)weights_mat, bias,
                  output, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, WS);

  // Apply LeakyReLU activation manually since tiled_conv_auto doesn't support
  // it
  // int total_outputs = batch * out_rows * out_cols * out_channels;
  // for (int i = 0; i < total_outputs; i++) {
  //  output[i] = leaky_relu_activation(output[i]);
  //}
}

// Fully connected layer using Gemmini with LeakyReLU
void gemmini_fc(int batch, int in_features, int out_features, elem_t *input,
                elem_t *weights, acc_t *bias, elem_t *output) {
  // Create temporary buffer for transposed weights
  // static elem_t weights_transposed[200]
  //                                [64]; // Maximum size for largest FC layer

  // Manual transpose: weights[out_features][in_features] ->
  // weights_transposed[in_features][out_features]
  // for (int i = 0; i < out_features; i++) {
  //  for (int j = 0; j < in_features; j++) {
  //    weights_transposed[j][i] = ((elem_t(*)[in_features])weights)[i][j];
  //  }
  //}

  // Use WS mode with manually transposed weights
  // tiled_matmul_auto(batch, out_features, in_features, (elem_t *)input,
  //                  (elem_t *)weights_transposed, bias, output, in_features,
  //                  in_features, out_features, out_features,
  //                  MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                  MVIN_SCALE_IDENTITY, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
  //                  false, false, false, false, false, 0, OS);
  tiled_matmul_auto(batch, out_features, in_features, (elem_t *)input,
                    (elem_t *)weights, bias, output, in_features, in_features,
                    out_features, out_features, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION,
                    ACC_SCALE_IDENTITY, 0, false, false, true, false, false, 0,
                    WS);
}

// Non-Local Block implementation
void gemmini_nlb(elem_t *input, elem_t *output) {
  // Theta path: 1x11x11x64 -> conv -> 1x9x9x32
  static elem_t nlb_theta_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                             [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_FILTERS,
                 CONV1_KERNEL, 1, 0, input, (elem_t *)nlb_theta_weights,
                 nlb_theta_bias, (elem_t *)nlb_theta_out);

  // printf("Conv theta \n");

  // Reshape theta from 1x9x9x32 to 1x32x81 then transpose to 1x81x32
  static elem_t nlb_theta_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  // static elem_t nlb_theta_transposed[81][32]; // 81x32 format (transposed)
  static elem_t identity_32x32[CONV2_FILTERS]
                              [CONV2_FILTERS]; // Identity matrix for transpose

  // Initialize identity matrix
  for (int i = 0; i < CONV2_FILTERS; i++) {
    for (int j = 0; j < CONV2_FILTERS; j++) {
      identity_32x32[i][j] = (i == j) ? 1 : 0;
    }
  }

  // printf("Init identity matrix \n");

  // Reshape: convert 9x9 spatial dimensions to 81 elements
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w; // Convert 2D (h,w) to 1D index
        nlb_theta_reshaped[c][idx] = nlb_theta_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape theta \n");

  // Phi path: 1x11x11x64 -> conv -> 1x9x9x32 -> reshape to 1x32x81
  static elem_t nlb_phi_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                           [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_FILTERS,
                 CONV1_KERNEL, 1, 0, input, (elem_t *)nlb_phi_weights,
                 nlb_phi_bias, (elem_t *)nlb_phi_out);

  // printf("Conv phi \n");

  static elem_t nlb_phi_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w;
        nlb_phi_reshaped[c][idx] = nlb_phi_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape phi \n");

  // G path: 1x11x11x64 -> conv -> 1x9x9x32 -> reshape to 1x32x81
  static elem_t nlb_g_out[CONV2_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_CHANNELS, CONV2_FILTERS,
                 CONV1_KERNEL, 1, 0, input, (elem_t *)nlb_g_weights, nlb_g_bias,
                 (elem_t *)nlb_g_out);

  // printf("Conv g \n");

  static elem_t nlb_g_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  // static elem_t nlb_g_transposed[81][32]; // 81x32 format (transposed)
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int spatial_idx = h * CONV1_DIM + w;
        nlb_g_reshaped[c][spatial_idx] = nlb_g_out[0][h][w][c];
      }
    }
  }

  // printf("Reshape g \n");

  // Manual transpose for theta: 32x81 -> 81x32
  // static elem_t nlb_theta_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_theta_transposed[j][i] = nlb_theta_reshaped[i][j];
  //  }
  //}

  // Manual transpose for phi: 32x81 -> 81x32
  // static elem_t nlb_phi_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_phi_transposed[j][i] = nlb_phi_reshaped[i][j];
  //  }
  //}

  // Attention computation: theta^T @ phi^T^T = theta^T @ phi = 81x32 @ 32x81 =
  // 81x81 Use OS mode with no transpose flags since we manually transposed
  static elem_t attention_matrix[CONV1_2D][CONV1_2D];
  // tiled_matmul_auto(
  //     CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
  //     (elem_t *)nlb_theta_transposed,    // A [81x32] (manually transposed)
  //     (elem_t *)nlb_phi_reshaped,        // B [32x81] (original)
  //     NULL,                              // D (bias)
  //     (elem_t *)attention_matrix,        // C [81x81]
  //     CONV2_FILTERS, CONV1_2D, CONV1_2D, CONV1_2D, // strides (A, B, D, C)
  //     MVIN_SCALE_IDENTITY,                         // A_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // B_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // D_scale_factor
  //     NO_ACTIVATION,      // NO_ACTIVATION (remove SOFTMAX)
  //     ACC_SCALE_IDENTITY, // scale
  //     0,                  // bert_scale
  //     false,              // repeating_bias
  //     false,              // transpose_A (already done manually)
  //     false,              // transpose_B (use original phi)
  //     false,              // full_C
  //     false,              // low_D
  //     0,                  // weightA
  //     OS                  // Use OS mode
  //);
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
                    NO_ACTIVATION,       // NO_ACTIVATION
                    ACC_SCALE_IDENTITY,  // scale
                    0,                   // bert_scale
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    WS                   // Use WS mode
  );

  // printf("Matmul theta and phi \n");

  // Manual transpose for g: 32x81 -> 81x32
  // static elem_t nlb_g_transposed[CONV1_2D][CONV2_FILTERS];
  // for (int i = 0; i < CONV2_FILTERS; i++) {
  //  for (int j = 0; j < CONV1_2D; j++) {
  //    nlb_g_transposed[j][i] = nlb_g_reshaped[i][j];
  //  }
  //}

  // Apply attention to g: attention @ g^T = 81x81 @ 81x32 = 81x32
  // Use OS mode with no transpose flags since we manually transposed g
  static elem_t attended_output[CONV1_2D][CONV2_FILTERS];
  // tiled_matmul_auto(
  //     CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
  //     (elem_t *)nlb_theta_transposed,    // A [81x32] (manually transposed)
  //     (elem_t *)nlb_phi_reshaped,        // B [32x81] (original)
  //     NULL,                              // D (bias)
  //     (elem_t *)attention_matrix,        // C [81x81]
  //     CONV2_FILTERS, CONV1_2D, CONV1_2D, CONV1_2D, // strides (A, B, D, C)
  //     MVIN_SCALE_IDENTITY,                         // A_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // B_scale_factor
  //     MVIN_SCALE_IDENTITY,                         // D_scale_factor
  //     NO_ACTIVATION,                               // act
  //     ACC_SCALE_IDENTITY,                          // scale
  //     0,                                           // bert_scale
  //     false,                                       // repeating_bias
  //     false,                                       // transpose_A
  //     false, // transpose_B (already done manually)
  //     false, // full_C
  //     false, // low_D
  //     0,     // weightA
  //     OS     // Use OS mode
  //);
  tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
                    (elem_t *)nlb_theta_reshaped,      // A [81x32]
                    (elem_t *)nlb_phi_reshaped,        // B [32x81]
                    NULL,                              // D (bias)
                    (elem_t *)attention_matrix,        // C [81x81]
                    CONV2_FILTERS, CONV1_2D, CONV1_2D,
                    CONV1_2D,            // strides (A, B, D, C)
                    MVIN_SCALE_IDENTITY, // A_scale_factor
                    MVIN_SCALE_IDENTITY, // B_scale_factor
                    MVIN_SCALE_IDENTITY, // D_scale_factor
                    NO_ACTIVATION,       // act
                    ACC_SCALE_IDENTITY,  // scale
                    0,                   // bert_scale
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    WS                   // Use WS mode
  );

  // printf("Matmul g \n");

  // TODO: Transpose
  // printf("Before Conv NLB \n");

  // Reshape back to spatial format: 81x32 -> 1x9x9x32
  static elem_t theta_phi_g_output[CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int idx = 0; idx < CONV1_2D; idx++) {
      int h = idx / CONV1_DIM;
      int w = idx % CONV1_DIM;
      ((elem_t(*)[CONV1_DIM][CONV1_DIM][CONV2_FILTERS])
           theta_phi_g_output)[0][h][w][c] = attended_output[idx][c];
    }
  }

  // printf("Reshape theta, phi, and g \n");

  static elem_t nlb_output[CONV1_DIM][CONV1_DIM][64];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS,
                 CONV1_KERNEL, 1, 0, theta_phi_g_output,
                 (elem_t *)nlb_out_weights, nlb_out_bias, (elem_t *)nlb_output);

  // printf("Conv theta, phi, and g \n");

  tiled_resadd_auto(CONV1_2D, CONV1_FILTERS, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, (elem_t *)input,
                    (elem_t *)nlb_output, (elem_t *)output, false, WS);

  // printf("Add \n");
}

void gemmini_inference(elem_t *input, elem_t *output) {
  // Allocate intermediate buffers with correct feature map sizes
  // 3 CNN layers with Non-Local Block
  static elem_t conv1_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                         [CONV1_FILTERS]; // 11x11 -> 9x9x64 (3x3, no pad)
  static elem_t nlb_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                       [CONV1_FILTERS]; // NLB output: 9x9x64
  static elem_t conv2_out[CONV2_CHANNELS][CONV2_DIM][CONV2_DIM]
                         [CONV2_FILTERS]; // 9x9 -> 7x7x32 (3x3, no pad)
  static elem_t conv3_out[CONV3_KERNEL][CONV3_DIM][CONV3_DIM]
                         [CONV3_FILTERS]; // 7x7 -> 5x5x8 (3x3, no pad)

  static elem_t fc1_out[FC1_UNITS];
  static elem_t fc2_out[FC2_UNITS];
  static elem_t fc3_out[FC3_UNITS];
  static elem_t fc4_out[FC4_UNITS];

  // printf("BraggNN Inference: Starting forward pass \n");
  // printf("  Input: %dx%dx%d patch (Bragg peak)\n", INPUT_DIM, INPUT_DIM,
  //        INPUT_CHANNELS);

  // Conv1: 11x11x1 -> 9x9x64 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 1: Conv2D (3x3, 64 filters) -> 9x9x64\n");
  gemmini_conv2d(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_FILTERS,
                 CONV1_KERNEL, 1, 0, input, (elem_t *)conv1_weights, conv1_bias,
                 (elem_t *)conv1_out);

  // Non-Local Block
  // printf("  Non-Local Block \n");
  gemmini_nlb((elem_t *)conv1_out, (elem_t *)nlb_out);

  for (int i = 0; i < CONV1_DIM; i++) {
    for (int j = 0; j < CONV1_DIM; j++) {
      for (int k = 0; k < CONV1_FILTERS; k++) {
        nlb_out[0][i][j][k] = leaky_relu_activation(nlb_out[0][i][j][k]);
      }
    }
  }
  // Conv2: 9x9x64 -> 7x7x32 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 2: Conv2D (3x3, 32 filters) + LeakyReLU -> 7x7x32\n");
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 CONV2_KERNEL, 1, 0, nlb_out, (elem_t *)conv2_weights,
                 conv2_bias, (elem_t *)conv2_out);

  for (int i = 0; i < CONV2_DIM; i++) {
    for (int j = 0; j < CONV2_DIM; j++) {
      for (int k = 0; k < CONV2_FILTERS; k++) {
        conv2_out[0][i][j][k] = leaky_relu_activation(conv2_out[0][i][j][k]);
      }
    }
  }
  // Conv3: 7x7x32 -> 5x5x8 (3x3 kernel, no padding) + LeakyReLU
  // printf("  Layer 3: Conv2D (3x3, 8 filters) + LeakyReLU -> 5x5x8\n");
  gemmini_conv2d(BATCH, CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS,
                 CONV3_KERNEL, 1, 0, conv2_out, (elem_t *)conv3_weights,
                 conv3_bias, (elem_t *)conv3_out);

  // Flatten: 5x5x8 -> 200
  elem_t flattened[CONV3_FLATTENED];
  int idx = 0;
  for (int r = 0; r < CONV3_DIM; r++) {
    for (int c = 0; c < CONV3_DIM; c++) {
      for (int ch = 0; ch < CONV3_FILTERS; ch++) {
        flattened[idx++] = conv3_out[0][r][c][ch];
      }
    }
  }

  for (int i = 0; i < CONV3_FLATTENED; i++) {
    flattened[i] = leaky_relu_activation(flattened[i]);
  }
  // printf("  Layer 4: FC (200 -> 16) + LeakyReLU\n");
  gemmini_fc(CONV3_CHANNELS, CONV3_FLATTENED, FC1_UNITS, flattened,
             (elem_t *)fc1_weights, fc1_bias, fc1_out);

  for (int i = 0; i < FC1_UNITS; i++) {
    fc1_out[i] = leaky_relu_activation(fc1_out[i]);
  }
  // printf("  Layer 5: FC (16 -> 8) + LeakyReLU\n");
  gemmini_fc(1, FC1_UNITS, FC2_UNITS, fc1_out, (elem_t *)fc2_weights, fc2_bias,
             fc2_out);

  for (int i = 0; i < FC2_UNITS; i++) {
    fc2_out[i] = leaky_relu_activation(fc2_out[i]);
  }
  // printf("  Layer 6: FC (8 -> 4) + LeakyReLU\n");
  gemmini_fc(1, FC2_UNITS, FC3_UNITS, fc2_out, (elem_t *)fc3_weights, fc3_bias,
             fc3_out);

  for (int i = 0; i < FC3_UNITS; i++) {
    fc3_out[i] = leaky_relu_activation(fc3_out[i]);
  }
  // printf("  Layer 7: FC (4 -> 2) + LeakyReLU\n");
  gemmini_fc(1, FC3_UNITS, FC4_UNITS, fc3_out, (elem_t *)fc4_weights, fc4_bias,
             fc4_out);

  for (int i = 0; i < FC4_UNITS; i++) {
    fc4_out[i] = leaky_relu_activation(fc4_out[i]);
  }
  // Output: 8 -> 2 (x, y coordinates of peak center) - Linear (no activation)
  // printf("  Layer 8: Output FC (2 -> 2) [Linear]\n");
  // for (int i = 0; i < OUTPUT_UNITS; i++) {
  //   acc_t sum = output_bias[i];
  //   for (int j = 0; j < FC4_UNITS; j++) {
  //     sum += fc4_out[j] * output_weights[i][j];
  //   }
  //   output[i] = (elem_t)sum; // Linear activation (no activation function)
  // }
  gemmini_fc(1, FC4_UNITS, OUTPUT_UNITS, fc4_out, (elem_t *)output_weights,
             output_bias, output);

  // printf("Gemmini Inference: Complete\n");
}

int main() {
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
  // for (int i = 0; i < INPUT_DIM * INPUT_DIM * INPUT_CHANNELS; i++) {
  //   // Simulate a Gaussian-like peak in the center for testing
  //   int row = i / INPUT_DIM;
  //   int col = i % INPUT_DIM;
  //   float dx = row - INPUT_DIM / 2.0;
  //   float dy = col - INPUT_DIM / 2.0;
  //   float dist_sq = dx * dx + dy * dy;
  //   input[i] = (elem_t)exp(-dist_sq / 8.0); // Gaussian peak
  // }

  // Output buffer for predictions
  elem_t predictions_gemmini[OUTPUT_UNITS];
  elem_t predictions_cpu[OUTPUT_UNITS];

  // Run inference
  printf("\nRunning Gemmini inference...\n");
  uint64_t start_gemmini = read_cycles();
  gemmini_inference(input, predictions_gemmini);
  uint64_t end_gemmini = read_cycles();
  printf("Gemmini BraggNN took %llu cycles\n", end_gemmini - start_gemmini);

  printf("\nRunning CPU inference...\n");
  uint64_t start_cpu = read_cycles();
  cpu_inference(input, predictions_cpu);
  uint64_t end_cpu = read_cycles();
  printf("CPU BraggNN took %llu cycles\n", end_cpu - start_cpu);

  // Display results
  printf("\nPrediction Results (Peak Center Coordinates):\n");
  printf("============================================\n");
  const char *param_names[2] = {"X coordinate", "Y coordinate"};
  // for (int i = 0; i < OUTPUT_UNITS; i++) {
  //   printf("Gemmini:  %s: %d\n", param_names[i], predictions_gemmini[i]);
  // }
  for (int i = 0; i < OUTPUT_UNITS; i++) {
    printf("CPU:  %s: %d\n", param_names[i], predictions_cpu[i]);
  }

  printf("\nInference completed successfully!\n");

  exit(0);
}
