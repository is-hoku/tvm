#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "braggnn.h"
#include "include/gemmini_testutils.h"


// LeakyReLU with requantization scale (for QLinearLeakyRelu nodes)
static inline elem_t leaky_relu_requant(elem_t x, float scale) {
  float val = (x > 0) ? (x * scale) : (x * 0.01f * scale);
  int result = (int)(val + (val >= 0 ? 0.5f : -0.5f));
  if (result > 127) result = 127;
  if (result < -128) result = -128;
  return (elem_t)result;
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


// Convolution operation using Gemmini's tiled_conv_auto
void gemmini_conv2d(int batch, int in_rows, int in_cols, int in_channels,
                    int out_channels, int kernel_dim, int stride, int padding,
                    elem_t *input, elem_t *weights, acc_t *bias, elem_t *output,
                    bool relu, acc_scale_t acc_scale) {

  int out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
  int out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;
  int patch_size = kernel_dim * kernel_dim * in_channels;

  elem_t weights_mat[patch_size][out_channels];

  flatten_weights(out_channels, kernel_dim, in_channels, patch_size,
                  (elem_t(*)[kernel_dim][kernel_dim][in_channels])weights,
                  weights_mat);

  tiled_conv_auto(batch, in_rows, in_cols, in_channels, out_channels, out_rows,
                  out_cols, stride, 1, 1, padding, kernel_dim, false, false,
                  false, false, false, (elem_t *)input, (elem_t *)weights_mat,
                  (acc_t *)bias, (elem_t *)output, relu ? RELU : NO_ACTIVATION,
                  acc_scale, 0, 0, 0, WS);

  // tiled_conv_auto does not have a fence (unlike tiled_matmul_outer).
  gemmini_fence();
}

// Fully connected layer using Gemmini with LeakyReLU
void gemmini_fc(int batch, int in_features, int out_features, elem_t *input,
                elem_t *weights, acc_t *bias, elem_t *output, bool relu, acc_scale_t acc_scale) {
  tiled_matmul_auto(
      batch, out_features, in_features, (elem_t *)input, (elem_t *)weights,
      (acc_t *)bias, (elem_t *)output, in_features, in_features, out_features,
      out_features, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY, relu ? RELU : NO_ACTIVATION, acc_scale, 0,
      false, false, true, false, false, 0, WS);
}

// Non-Local Block implementation
void gemmini_nlb(elem_t *input, elem_t *output) {
  //printf("[NLB] Starting theta conv...\n");
  //printf("theta weights[0][0..2]: %d, %d, %d\n", nlb_theta_weights[0][0], nlb_theta_weights[0][1], nlb_theta_weights[0][2]);
  //printf("theta bias[0..2]: %d, %d, %d\n", nlb_theta_bias[0], nlb_theta_bias[1], nlb_theta_bias[2]);

  static elem_t nlb_theta_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                             [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_THETA_KERNEL, 1, 0, (elem_t *)input,
                 (elem_t *)nlb_theta_weights, (acc_t *)nlb_theta_bias,
                 (elem_t *)nlb_theta_out, false, NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE);

  //printf("nlb_theta_out: %d, %d, %d\n", nlb_theta_out[0][0][0][0],
  //       nlb_theta_out[0][0][0][1], nlb_theta_out[0][0][0][2]);
  //printf("nlb_theta_out more: [0][1][0]=%d, [0][4][4]=%d, [0][8][8]=%d\n",
  //       nlb_theta_out[0][1][0][0], nlb_theta_out[0][4][4][0], nlb_theta_out[0][8][8][0]);

  static elem_t nlb_theta_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  static elem_t identity_32x32[CONV2_FILTERS]
                              [CONV2_FILTERS]; // Identity matrix for transpose

  // Initialize identity matrix
  for (int i = 0; i < CONV2_FILTERS; i++) {
    for (int j = 0; j < CONV2_FILTERS; j++) {
      identity_32x32[i][j] = (i == j) ? 1 : 0;
    }
  }

  // Reshape: convert 9x9 spatial dimensions to 81 elements
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w; // Convert 2D (h,w) to 1D index
        nlb_theta_reshaped[c][idx] = nlb_theta_out[0][h][w][c];
      }
    }
  }

  //printf("phi input[0..2]: %d, %d, %d\n", ((elem_t*)input)[0], ((elem_t*)input)[1], ((elem_t*)input)[2]);
  //printf("phi weights[0][0..2]: %d, %d, %d\n", nlb_phi_weights[0][0], nlb_phi_weights[0][1], nlb_phi_weights[0][2]);
  //printf("phi bias[0..2]: %d, %d, %d\n", nlb_phi_bias[0], nlb_phi_bias[1], nlb_phi_bias[2]);
  //printf("phi acc_scale: NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE\n");

  static elem_t nlb_phi_out[CONV1_CHANNELS][CONV1_DIM][CONV1_DIM]
                           [CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_PHI_KERNEL, 1, 0, (elem_t *)input, (elem_t *)nlb_phi_weights,
                 (acc_t *)nlb_phi_bias, (elem_t *)nlb_phi_out, false, NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE);

  //printf("nlb_phi_out: %d, %d, %d\n", nlb_phi_out[0][0][0][0],
  //       nlb_phi_out[0][0][0][1], nlb_phi_out[0][0][0][2]);
  //printf("nlb_phi_out more: [0][1][0]=%d, [0][4][4]=%d, [0][8][8]=%d\n",
  //       nlb_phi_out[0][1][0][0], nlb_phi_out[0][4][4][0], nlb_phi_out[0][8][8][0]);

  static elem_t nlb_phi_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int idx = h * CONV1_DIM + w;
        nlb_phi_reshaped[c][idx] = nlb_phi_out[0][h][w][c];
      }
    }
  }

  //printf("g weights[0][0..2]: %d, %d, %d\n", nlb_g_weights[0][0], nlb_g_weights[0][1], nlb_g_weights[0][2]);
  //printf("g bias[0..2]: %d, %d, %d\n", nlb_g_bias[0], nlb_g_bias[1], nlb_g_bias[2]);

  static elem_t nlb_g_out[CONV2_CHANNELS][CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 NLB_G_KERNEL, 1, 0, (elem_t *)input, (elem_t *)nlb_g_weights,
                 (acc_t *)nlb_g_bias, (elem_t *)nlb_g_out, false, NLB_G_LAYER_CONV_QUANT_ACC_SCALE);

  //printf("nlb_g_out: %d, %d, %d\n", nlb_g_out[0][0][0][0],
  //       nlb_g_out[0][0][0][1], nlb_g_out[0][0][0][2]);
  //printf("nlb_g_out more: [0][1][0]=%d, [0][4][4]=%d, [0][8][8]=%d\n",
  //       nlb_g_out[0][1][0][0], nlb_g_out[0][4][4][0], nlb_g_out[0][8][8][0]);

  static elem_t nlb_g_reshaped[CONV2_FILTERS][CONV1_2D]; // 32x81 format
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int h = 0; h < CONV1_DIM; h++) {
      for (int w = 0; w < CONV1_DIM; w++) {
        int spatial_idx = h * CONV1_DIM + w;
        nlb_g_reshaped[c][spatial_idx] = nlb_g_out[0][h][w][c];
      }
    }
  }

  // Attention computation: theta^T @ phi = 81x32 @ 32x81 = 81x81
  static elem_t attention_matrix[CONV1_2D][CONV1_2D];
  tiled_matmul_auto(CONV1_2D, CONV1_2D, CONV2_FILTERS, // dim_I, dim_J, dim_K
                    (elem_t *)nlb_theta_reshaped,      // A [32x81]
                    (elem_t *)nlb_phi_reshaped,        // B [32x81]
                    NULL,                              // D (bias)
                    (elem_t *)attention_matrix,        // C [81x81]
                    CONV1_2D, CONV1_2D, CONV1_2D,
                    CONV1_2D,            // strides (A, B, D, C)
                    MVIN_SCALE_IDENTITY, // A_scale_factor
                    MVIN_SCALE_IDENTITY, // B_scale_factor
                    MVIN_SCALE_IDENTITY, // D_scale_factor
                    NO_ACTIVATION,             // Softmax
                    NLB_MATMUL_QUANT_ACC_SCALE,  // scale
                    SOFTMAX_INPUT_SCALE, // bert scale (quantization scale of matmul output)
                    false,               // repeating_bias
                    true,                // transpose_A
                    false,               // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    WS                   // Use WS mode
  );

  // Manual Softmax implementation
  for (int i = 0; i < CONV1_2D; i++) {
    // Step 1: Dequantize to float and find max (for numerical stability)
    float row_float[CONV1_2D];
    float max_val = -1e9f;
    for (int j = 0; j < CONV1_2D; j++) {
      row_float[j] = (float)attention_matrix[i][j] * SOFTMAX_INPUT_SCALE;
      if (row_float[j] > max_val) max_val = row_float[j];
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < CONV1_2D; j++) {
      float x = row_float[j] - max_val;  // x <= 0
      // exp(x) approximation using Taylor series
      // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 for better accuracy
      float exp_val;
      if (x > -8.0f) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        exp_val = 1.0f + x + x2 * 0.5f + x3 * 0.166667f + x4 * 0.041667f;
        if (exp_val < 0.0f) exp_val = 0.0001f;
      } else {
        exp_val = 0.0001f;
      }
      row_float[j] = exp_val;
      sum_exp += exp_val;
    }

    // Step 3: Normalize and quantize
    for (int j = 0; j < CONV1_2D; j++) {
      float softmax_val = row_float[j] / sum_exp;
      // Quantize: int8_val = float_val / scale
      int quantized = (int)(softmax_val / SOFTMAX_OUTPUT_SCALE + 0.5f);
      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;
      attention_matrix[i][j] = (elem_t)quantized;
    }
  }

  //printf("attention_matrix (after softmax): %d, %d, %d\n", attention_matrix[0][0],
  //       attention_matrix[0][1], attention_matrix[0][2]);
  //int row_sum = 0;
  //for (int j = 0; j < CONV1_2D; j++) {
  //    row_sum += attention_matrix[0][j];
  //}
  //printf("attention row 0 sum: %d (expected ~990 for correct Softmax with scale 0.001)\n", row_sum);

  static elem_t attended_output[CONV1_2D][CONV2_FILTERS];
  tiled_matmul_auto(CONV1_2D, CONV2_FILTERS, CONV1_2D, // dim_I, dim_J, dim_K
                    (elem_t *)attention_matrix,        // A [81x81]
                    (elem_t *)nlb_g_reshaped,          // B [32x81]
                    NULL,                              // D (bias)
                    (elem_t *)attended_output,         // C [81x32]
                    CONV1_2D, CONV1_2D, CONV2_FILTERS,
                    CONV2_FILTERS,       // strides (A, B, D, C)
                    MVIN_SCALE_IDENTITY, // A_scale_factor
                    MVIN_SCALE_IDENTITY, // B_scale_factor
                    MVIN_SCALE_IDENTITY, // D_scale_factor
                    NO_ACTIVATION,       // act
                    NLB_MATMUL_1_QUANT_ACC_SCALE,  // scale
                    0.0,                 // bert scale
                    false,               // repeating_bias
                    false,               // transpose_A
                    true,                // transpose_B
                    false,               // full_C
                    false,               // low_D
                    0,                   // weightA
                    WS                   // Use WS mode
  );

  //printf("attended_output: %d, %d, %d\n", attended_output[0][0],
  //       attended_output[0][1], attended_output[0][2]);

  static elem_t theta_phi_g_output[CONV1_DIM][CONV1_DIM][CONV2_FILTERS];
  for (int c = 0; c < CONV2_FILTERS; c++) {
    for (int idx = 0; idx < CONV1_2D; idx++) {
      int h = idx / CONV1_DIM;
      int w = idx % CONV1_DIM;
      ((elem_t(*)[CONV1_DIM][CONV1_DIM][CONV2_FILTERS])
           theta_phi_g_output)[0][h][w][c] = attended_output[idx][c];
    }
  }

  //printf("theta_phi_g_output: %d, %d, %d\n", theta_phi_g_output[0][0][0],
  //       theta_phi_g_output[0][0][1], theta_phi_g_output[0][0][2]);

  static elem_t nlb_output[CONV1_DIM][CONV1_DIM][64];
  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS,
                 NLB_OUT_KERNEL, 1, 0, (elem_t *)theta_phi_g_output,
                 (elem_t *)nlb_out_weights, (acc_t *)nlb_out_bias,
                 (elem_t *)nlb_output, false, NLB_OUT_CNN_CONV_QUANT_ACC_SCALE);
  //printf("nlb_output (from out_conv): %d, %d, %d\n", nlb_output[0][0][0],
  //       nlb_output[0][0][1], nlb_output[0][0][2]);

  tiled_resadd_auto(CONV1_2D, CONV1_FILTERS,
                    NLB_ADD_B_SCALE,       // A_scale for input (skip connection)
                    NLB_ADD_A_SCALE,       // B_scale for nlb_output (out_cnn)
                    ACC_SCALE_IDENTITY,    // C_scale
                    (elem_t *)input,
                    (elem_t *)nlb_output,
                    (elem_t *)output, false, WS);
}

void gemmini_inference(
    float *fp32_input, elem_t *output) {

  // QuantizeLinear: y_scale = 0.007874015718698502
  elem_t input[BATCH][INPUT_DIM][INPUT_DIM][INPUT_CHANNELS];
  const acc_t y_scale = 127.0f;  // 1/0.007874 ≈ 127
  for (int i = 0; i < BATCH * INPUT_DIM * INPUT_DIM * INPUT_CHANNELS; i++){
      ((elem_t*)input)[i] = (elem_t)(((float*)fp32_input)[i] * y_scale);
  }

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

  gemmini_conv2d(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_FILTERS,
                 CONV1_KERNEL, 1, 0, (elem_t *)input, (elem_t *)conv1_weights,
                 (acc_t *)conv1_bias, (elem_t *)conv1_out, false, CNN_LAYERS_0_CONV_QUANT_ACC_SCALE);
  //printf("[DEBUG] Conv1 done\n");
  //printf("conv1_out ch0-7@(0,0): %d, %d, %d, %d, %d, %d, %d, %d\n",
  //       conv1_out[0][0][0][0], conv1_out[0][0][0][1], conv1_out[0][0][0][2],
  //       conv1_out[0][0][0][3], conv1_out[0][0][0][4], conv1_out[0][0][0][5],
  //       conv1_out[0][0][0][6], conv1_out[0][0][0][7]);
  //printf("conv1_out ch0-7@(4,4): %d, %d, %d, %d, %d, %d, %d, %d\n",
  //       conv1_out[0][4][4][0], conv1_out[0][4][4][1], conv1_out[0][4][4][2],
  //       conv1_out[0][4][4][3], conv1_out[0][4][4][4], conv1_out[0][4][4][5],
  //       conv1_out[0][4][4][6], conv1_out[0][4][4][7]);

  gemmini_nlb((elem_t *)conv1_out, (elem_t *)nlb_out);

  //printf("nlb_out: %d, %d, %d", nlb_out[0][0][0][0], nlb_out[0][0][0][1],
  //       nlb_out[0][0][0][2]);

  // LeakyReLU with requantization: NLB add output scale -> Conv2 input scale
  for (int i = 0; i < CONV1_DIM; i++) {
    for (int j = 0; j < CONV1_DIM; j++) {
      for (int k = 0; k < CONV1_FILTERS; k++) {
        nlb_out[0][i][j][k] = leaky_relu_requant(nlb_out[0][i][j][k],
                                                  CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE);
      }
    }
  }

  gemmini_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
                 CONV2_KERNEL, 1, 0, (elem_t *)nlb_out, (elem_t *)conv2_weights,
                 (acc_t *)conv2_bias, (elem_t *)conv2_out, false, CNN_LAYERS_2_CONV_QUANT_ACC_SCALE);

  // LeakyReLU with requantization: Conv2 output scale -> Conv3 input scale
  for (int i = 0; i < CONV2_DIM; i++) {
    for (int j = 0; j < CONV2_DIM; j++) {
      for (int k = 0; k < CONV2_FILTERS; k++) {
        conv2_out[0][i][j][k] = leaky_relu_requant(conv2_out[0][i][j][k],
                                                    CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE);
      }
    }
  }
  //printf("conv2_out: %d, %d, %d\n", conv2_out[0][0][0][0],
  //conv2_out[0][0][0][1],
  //      conv2_out[0][0][0][2]);
  gemmini_conv2d(BATCH, CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS,
                 CONV3_KERNEL, 1, 0, (elem_t *)conv2_out,
                 (elem_t *)conv3_weights, (acc_t *)conv3_bias,
                 (elem_t *)conv3_out, false, CNN_LAYERS_4_CONV_QUANT_ACC_SCALE);

  // Flatten: 5x5x8 -> 200 (NCHW order to match ONNX FC weight layout), Gemmini layout is OHWI
  elem_t flattened[CONV3_FLATTENED];
  int idx = 0;
  for (int ch = 0; ch < CONV3_FILTERS; ch++) {
    for (int r = 0; r < CONV3_DIM; r++) {
      for (int c = 0; c < CONV3_DIM; c++) {
        flattened[idx++] = conv3_out[0][r][c][ch];
      }
    }
  }
  //printf("conv3_out: %d, %d, %d\n", conv3_out[0][0][0][0],
  //conv3_out[0][0][0][1],
  //       conv3_out[0][0][0][2]);

  // LeakyReLU with requantization: Conv3 output scale -> FC1 input scale
  for (int i = 0; i < CONV3_FLATTENED; i++) {
    flattened[i] = leaky_relu_requant(flattened[i],
                                      CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE);
  }
  gemmini_fc(CONV3_CHANNELS, CONV3_FLATTENED, FC1_UNITS, (elem_t *)flattened,
             (elem_t *)fc1_weights, (acc_t *)fc1_bias, (elem_t *)fc1_out, false, DENSE_LAYERS_0_GEMM_ACC_SCALE);

  for (int i = 0; i < FC1_UNITS; i++) {
    fc1_out[i] = leaky_relu_requant(fc1_out[i],
                                     DENSE_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE);
  }
  //printf("fc1_out: %d, %d, %d\n", fc1_out[0], fc1_out[1], fc1_out[2]);
  gemmini_fc(1, FC1_UNITS, FC2_UNITS, (elem_t *)fc1_out, (elem_t *)fc2_weights,
             (acc_t *)fc2_bias, (elem_t *)fc2_out, false, DENSE_LAYERS_2_GEMM_ACC_SCALE);

  for (int i = 0; i < FC2_UNITS; i++) {
    fc2_out[i] = leaky_relu_requant(fc2_out[i],
                                     DENSE_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE);
  }
  //printf("fc2_out: %d, %d, %d\n", fc2_out[0], fc2_out[1], fc2_out[2]);
  gemmini_fc(1, FC2_UNITS, FC3_UNITS, (elem_t *)fc2_out, (elem_t *)fc3_weights,
             (acc_t *)fc3_bias, (elem_t *)fc3_out, false, DENSE_LAYERS_4_GEMM_ACC_SCALE);

  for (int i = 0; i < FC3_UNITS; i++) {
    fc3_out[i] = leaky_relu_requant(fc3_out[i],
                                     DENSE_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE);
  }
  //printf("fc3_out: %d, %d, %d\n", fc3_out[0], fc3_out[1], fc3_out[2]);
  gemmini_fc(1, FC3_UNITS, FC4_UNITS, (elem_t *)fc3_out, (elem_t *)fc4_weights,
             (acc_t *)fc4_bias, (elem_t *)fc4_out, false, DENSE_LAYERS_6_GEMM_ACC_SCALE);

  for (int i = 0; i < FC4_UNITS; i++) {
    fc4_out[i] = leaky_relu_requant(fc4_out[i],
                                     DENSE_LAYERS_7_LEAKYRELU_QUANT_ACC_SCALE);
  }
  //printf("fc4_out: %d, %d, %d\n", fc4_out[0], fc4_out[1], fc4_out[2]);
  gemmini_fc(1, FC4_UNITS, OUTPUT_UNITS, (elem_t *)fc4_out,
             (elem_t *)output_weights, (acc_t *)output_bias, (elem_t *)output,
             false, DENSE_LAYERS_8_GEMM_ACC_SCALE);
}

// Simple float absolute value
static inline float fabsf_custom(float x) {
  return x >= 0.0f ? x : -x;
}

// Print float with 4 decimal places using only integer printf
// (bare-metal printf does not support %f)
static void print_float4(float f) {
  if (f < 0) {
    printf("-");
    f = -f;
  }
  int integer_part = (int)f;
  int frac_part = (int)((f - (float)integer_part) * 10000.0f + 0.5f);
  if (frac_part >= 10000) {
    integer_part++;
    frac_part -= 10000;
  }
  printf("%d.%04d", integer_part, frac_part);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  printf("==============================================\n");
  printf("BraggNN Gemmini through native C\n");
  printf("==============================================\n");

  printf("Flushing Gemmini TLB\n");
  gemmini_flush(0);

  const float DEQUANT_SCALE = 11.0f / 127.0f;  // int8 -> pixel coordinates

  elem_t predictions[OUTPUT_UNITS];
  unsigned long long cycle_counts[NUM_TEST_PATCHES];
  unsigned long long total_cycles = 0;
  float total_x_error = 0.0f;
  float total_y_error = 0.0f;

  printf("\nInput shape: [%d, %d, %d, %d]\n", BATCH, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM);
  printf("Running %d inferences (%d warmup + %d measured)\n",
         NUM_TEST_PATCHES + 1, 1, NUM_TEST_PATCHES);

  // Warmup: drop first run to warm caches
  printf("\n--- Warmup (patch 0) ---\n");
  unsigned long long start = read_cycles();
  gemmini_inference(test_inputs[5], predictions);
  unsigned long long end = read_cycles();
  printf("Warmup done\n");
  float pred_x = predictions[0] * DEQUANT_SCALE;
  float pred_y = predictions[1] * DEQUANT_SCALE;
  float actual_x = test_labels[5][0];
  float actual_y = test_labels[5][1];
  float err_x = pred_x - actual_x;
  float err_y = pred_y - actual_y;

  unsigned long long elapsed = end - start;

  printf("cycles: %llu\n", elapsed);
  printf("pred: ("); print_float4(pred_x); printf(", "); print_float4(pred_y);
  printf("), actual: ("); print_float4(actual_x); printf(", "); print_float4(actual_y);
  printf("), error: ("); print_float4(err_x); printf(", "); print_float4(err_y);
  printf(")\n");

  // Measured iterations
  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_TEST_PATCHES);

    unsigned long long start = read_cycles();
    asm volatile(".word 0x8013");  // TracerV start trigger
    gemmini_inference(test_inputs[i], predictions);
    asm volatile(".word 0x10013"); // TracerV end trigger
    unsigned long long end = read_cycles();

    float pred_x = predictions[0] * DEQUANT_SCALE;
    float pred_y = predictions[1] * DEQUANT_SCALE;
    float actual_x = test_labels[i][0];
    float actual_y = test_labels[i][1];
    float err_x = pred_x - actual_x;
    float err_y = pred_y - actual_y;

    unsigned long long elapsed = end - start;
    cycle_counts[i] = elapsed;
    total_cycles += elapsed;
    total_x_error += fabsf_custom(err_x);
    total_y_error += fabsf_custom(err_y);

    printf("cycles: %llu\n", elapsed);
    printf("pred: ("); print_float4(pred_x); printf(", "); print_float4(pred_y);
    printf("), actual: ("); print_float4(actual_x); printf(", "); print_float4(actual_y);
    printf("), error: ("); print_float4(err_x); printf(", "); print_float4(err_y);
    printf(")\n");
  }

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_TEST_PATCHES,
         total_cycles / NUM_TEST_PATCHES);
  printf("Avg error over %d runs: (", NUM_TEST_PATCHES);
  print_float4(total_x_error / NUM_TEST_PATCHES); printf(", ");
  print_float4(total_y_error / NUM_TEST_PATCHES); printf(")\n");
  printf("==============================================\n");
  printf("BraggNN inference completed successfully\n");
  printf("==============================================\n");

  exit(0);
}
