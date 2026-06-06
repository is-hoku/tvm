#define CPU_X86

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int8_t elem_t;
typedef int32_t acc_t;

#include "braggnn.h"

static inline elem_t round_and_clamp(float val) {
  int r = (int)(val + (val >= 0 ? 0.5f : -0.5f));
  if (r > 127) return 127;
  if (r < -128) return -128;
  return (elem_t)r;
}

static inline elem_t leaky_relu_requant(elem_t x, float scale) {
  float val = (x > 0) ? (x * scale) : (x * 0.01f * scale);
  int result = (int)(val + (val >= 0 ? 0.5f : -0.5f));
  if (result > 127) result = 127;
  if (result < -128) result = -128;
  return (elem_t)result;
}

static void cpu_conv2d(int batch, int in_h, int in_w, int in_ch,
                       int out_ch, int kernel_size, int stride, int padding,
                       const elem_t *input, const elem_t *weights,
                       const acc_t *bias, elem_t *output,
                       float acc_scale) {
  int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
  int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

  for (int b = 0; b < batch; b++) {
    for (int oh = 0; oh < out_h; oh++) {
      for (int ow = 0; ow < out_w; ow++) {
        for (int oc = 0; oc < out_ch; oc++) {
          acc_t sum = bias[oc];

          for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int ih = oh * stride + kh - padding;
              int iw = ow * stride + kw - padding;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                for (int ic = 0; ic < in_ch; ic++) {
                  int in_idx = ((b * in_h + ih) * in_w + iw) * in_ch + ic;
                  int w_idx = ((oc * kernel_size + kh) * kernel_size + kw) * in_ch + ic;
                  sum += (acc_t)input[in_idx] * (acc_t)weights[w_idx];
                }
              }
            }
          }

          int out_idx = ((b * out_h + oh) * out_w + ow) * out_ch + oc;
          output[out_idx] = round_and_clamp((float)sum * acc_scale);
        }
      }
    }
  }
}

static void cpu_fc(int batch, int in_features, int out_features,
                   const elem_t *input, const elem_t *weights,
                   const acc_t *bias, elem_t *output,
                   float acc_scale) {
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < out_features; j++) {
      acc_t sum = bias[j];
      for (int k = 0; k < in_features; k++) {
        sum += (acc_t)input[b * in_features + k] *
               (acc_t)weights[j * in_features + k];
      }
      output[b * out_features + j] = round_and_clamp((float)sum * acc_scale);
    }
  }
}

static void cpu_matmul_transA(int M, int N, int K,
                              const elem_t *A, const elem_t *B, elem_t *C,
                              int stride_A, int stride_B, int stride_C,
                              float acc_scale) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      acc_t sum = 0;
      for (int k = 0; k < K; k++) {
        sum += (acc_t)A[k * stride_A + i] * (acc_t)B[k * stride_B + j];
      }
      C[i * stride_C + j] = round_and_clamp((float)sum * acc_scale);
    }
  }
}

static void cpu_matmul_transB(int M, int N, int K,
                              const elem_t *A, const elem_t *B, elem_t *C,
                              int stride_A, int stride_B, int stride_C,
                              float acc_scale) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      acc_t sum = 0;
      for (int k = 0; k < K; k++) {
        sum += (acc_t)A[i * stride_A + k] * (acc_t)B[j * stride_B + k];
      }
      C[i * stride_C + j] = round_and_clamp((float)sum * acc_scale);
    }
  }
}

static void cpu_resadd(int rows, int cols,
                       float A_scale, float B_scale,
                       const elem_t *A, const elem_t *B, elem_t *C) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      float val = (float)A[idx] * A_scale + (float)B[idx] * B_scale;
      C[idx] = round_and_clamp(val);
    }
  }
}

static void cpu_nlb(const elem_t *input, elem_t *output) {
  // --- Theta 1x1 conv: 9x9x64 -> 9x9x32 ---
  static elem_t theta_out[CONV1_DIM * CONV1_DIM * CONV2_FILTERS];
  cpu_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
             NLB_THETA_KERNEL, 1, 0, input,
             (const elem_t *)nlb_theta_weights, (const acc_t *)nlb_theta_bias,
             theta_out, NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE);

  // Reshape NHWC [9x9x32] -> [32][81]
  static elem_t theta_reshaped[CONV2_FILTERS][CONV1_2D];
  for (int c = 0; c < CONV2_FILTERS; c++)
    for (int h = 0; h < CONV1_DIM; h++)
      for (int w = 0; w < CONV1_DIM; w++)
        theta_reshaped[c][h * CONV1_DIM + w] =
            theta_out[(h * CONV1_DIM + w) * CONV2_FILTERS + c];

  // --- Phi 1x1 conv: 9x9x64 -> 9x9x32 ---
  static elem_t phi_out[CONV1_DIM * CONV1_DIM * CONV2_FILTERS];
  cpu_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
             NLB_PHI_KERNEL, 1, 0, input,
             (const elem_t *)nlb_phi_weights, (const acc_t *)nlb_phi_bias,
             phi_out, NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE);

  static elem_t phi_reshaped[CONV2_FILTERS][CONV1_2D];
  for (int c = 0; c < CONV2_FILTERS; c++)
    for (int h = 0; h < CONV1_DIM; h++)
      for (int w = 0; w < CONV1_DIM; w++)
        phi_reshaped[c][h * CONV1_DIM + w] =
            phi_out[(h * CONV1_DIM + w) * CONV2_FILTERS + c];

  // --- G 1x1 conv: 9x9x64 -> 9x9x32 ---
  static elem_t g_out[CONV1_DIM * CONV1_DIM * CONV2_FILTERS];
  cpu_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
             NLB_G_KERNEL, 1, 0, input,
             (const elem_t *)nlb_g_weights, (const acc_t *)nlb_g_bias,
             g_out, NLB_G_LAYER_CONV_QUANT_ACC_SCALE);

  static elem_t g_reshaped[CONV2_FILTERS][CONV1_2D];
  for (int c = 0; c < CONV2_FILTERS; c++)
    for (int h = 0; h < CONV1_DIM; h++)
      for (int w = 0; w < CONV1_DIM; w++)
        g_reshaped[c][h * CONV1_DIM + w] =
            g_out[(h * CONV1_DIM + w) * CONV2_FILTERS + c];

  // --- Attention: theta^T @ phi -> [81][81] ---
  // theta_reshaped[32][81], phi_reshaped[32][81]
  // C[81][81] = theta^T[81][32] @ phi[32][81]
  static elem_t attention[CONV1_2D][CONV1_2D];
  cpu_matmul_transA(CONV1_2D, CONV1_2D, CONV2_FILTERS,
                    (const elem_t *)theta_reshaped,
                    (const elem_t *)phi_reshaped,
                    (elem_t *)attention,
                    CONV1_2D, CONV1_2D, CONV1_2D,
                    NLB_MATMUL_QUANT_ACC_SCALE);

  // --- Softmax (per row, matching Gemmini Taylor approximation) ---
  for (int i = 0; i < CONV1_2D; i++) {
    float row_float[CONV1_2D];
    float max_val = -1e9f;
    for (int j = 0; j < CONV1_2D; j++) {
      row_float[j] = (float)attention[i][j] * SOFTMAX_INPUT_SCALE;
      if (row_float[j] > max_val) max_val = row_float[j];
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < CONV1_2D; j++) {
      float x = row_float[j] - max_val;
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

    for (int j = 0; j < CONV1_2D; j++) {
      float softmax_val = row_float[j] / sum_exp;
      int quantized = (int)(softmax_val / SOFTMAX_OUTPUT_SCALE + 0.5f);
      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;
      attention[i][j] = (elem_t)quantized;
    }
  }

  // --- Attended output: attention @ g^T -> [81][32] ---
  // attention[81][81], g_reshaped[32][81]
  // C[81][32] = attention[81][81] @ g^T[81][32]
  static elem_t attended[CONV1_2D][CONV2_FILTERS];
  cpu_matmul_transB(CONV1_2D, CONV2_FILTERS, CONV1_2D,
                    (const elem_t *)attention,
                    (const elem_t *)g_reshaped,
                    (elem_t *)attended,
                    CONV1_2D, CONV1_2D, CONV2_FILTERS,
                    NLB_MATMUL_1_QUANT_ACC_SCALE);

  // Reshape [81][32] -> NHWC [9][9][32]
  static elem_t tpg_output[CONV1_DIM * CONV1_DIM * CONV2_FILTERS];
  for (int c = 0; c < CONV2_FILTERS; c++)
    for (int idx = 0; idx < CONV1_2D; idx++) {
      int h = idx / CONV1_DIM;
      int w = idx % CONV1_DIM;
      tpg_output[(h * CONV1_DIM + w) * CONV2_FILTERS + c] = attended[idx][c];
    }

  // --- out_cnn 1x1 conv: 9x9x32 -> 9x9x64 ---
  static elem_t nlb_conv_out[CONV1_DIM * CONV1_DIM * CONV1_FILTERS];
  cpu_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV2_FILTERS, CONV1_FILTERS,
             NLB_OUT_KERNEL, 1, 0, tpg_output,
             (const elem_t *)nlb_out_weights, (const acc_t *)nlb_out_bias,
             nlb_conv_out, NLB_OUT_CNN_CONV_QUANT_ACC_SCALE);

  // --- Residual add: output = input * B_scale + nlb_conv_out * A_scale ---
  cpu_resadd(CONV1_2D, CONV1_FILTERS,
             NLB_ADD_B_SCALE, NLB_ADD_A_SCALE,
             input, nlb_conv_out, output);
}

static void cpu_inference(float *fp32_input, elem_t *output) {
  // QuantizeLinear: y_scale = 0.007874015718698502 -> 1/0.007874 = 127
  elem_t input[BATCH * INPUT_DIM * INPUT_DIM * INPUT_CHANNELS];
  for (int i = 0; i < BATCH * INPUT_DIM * INPUT_DIM * INPUT_CHANNELS; i++)
    input[i] = (elem_t)(fp32_input[i] * 127.0f);

  // Conv1: 11x11x1 -> 9x9x64
  static elem_t conv1_out[CONV1_DIM * CONV1_DIM * CONV1_FILTERS];
  cpu_conv2d(BATCH, INPUT_DIM, INPUT_DIM, INPUT_CHANNELS, CONV1_FILTERS,
             CONV1_KERNEL, 1, 0, input,
             (const elem_t *)conv1_weights, (const acc_t *)conv1_bias,
             conv1_out, CNN_LAYERS_0_CONV_QUANT_ACC_SCALE);

  // Non-Local Block: 9x9x64 -> 9x9x64
  static elem_t nlb_out[CONV1_DIM * CONV1_DIM * CONV1_FILTERS];
  cpu_nlb(conv1_out, nlb_out);

  // LeakyReLU + requant (NLB add output scale -> Conv2 input scale)
  for (int i = 0; i < CONV1_DIM * CONV1_DIM * CONV1_FILTERS; i++)
    nlb_out[i] = leaky_relu_requant(nlb_out[i],
                                     CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE);

  // Conv2: 9x9x64 -> 7x7x32
  static elem_t conv2_out[CONV2_DIM * CONV2_DIM * CONV2_FILTERS];
  cpu_conv2d(BATCH, CONV1_DIM, CONV1_DIM, CONV1_FILTERS, CONV2_FILTERS,
             CONV2_KERNEL, 1, 0, nlb_out,
             (const elem_t *)conv2_weights, (const acc_t *)conv2_bias,
             conv2_out, CNN_LAYERS_2_CONV_QUANT_ACC_SCALE);

  // LeakyReLU + requant (Conv2 output scale -> Conv3 input scale)
  for (int i = 0; i < CONV2_DIM * CONV2_DIM * CONV2_FILTERS; i++)
    conv2_out[i] = leaky_relu_requant(conv2_out[i],
                                       CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE);

  // Conv3: 7x7x32 -> 5x5x8
  static elem_t conv3_out[CONV3_DIM * CONV3_DIM * CONV3_FILTERS];
  cpu_conv2d(BATCH, CONV2_DIM, CONV2_DIM, CONV2_FILTERS, CONV3_FILTERS,
             CONV3_KERNEL, 1, 0, conv2_out,
             (const elem_t *)conv3_weights, (const acc_t *)conv3_bias,
             conv3_out, CNN_LAYERS_4_CONV_QUANT_ACC_SCALE);

  // Flatten: NHWC [5][5][8] -> NCHW order [8][5][5] = 200 elements
  elem_t flattened[CONV3_FLATTENED];
  int idx = 0;
  for (int ch = 0; ch < CONV3_FILTERS; ch++)
    for (int r = 0; r < CONV3_DIM; r++)
      for (int c = 0; c < CONV3_DIM; c++)
        flattened[idx++] = conv3_out[(r * CONV3_DIM + c) * CONV3_FILTERS + ch];

  // LeakyReLU + requant (Conv3 output scale -> FC1 input scale)
  for (int i = 0; i < CONV3_FLATTENED; i++)
    flattened[i] = leaky_relu_requant(flattened[i],
                                       CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE);

  // FC1: 200 -> 16
  static elem_t fc1_out[FC1_UNITS];
  cpu_fc(1, CONV3_FLATTENED, FC1_UNITS, flattened,
         (const elem_t *)fc1_weights, (const acc_t *)fc1_bias,
         fc1_out, DENSE_LAYERS_0_GEMM_ACC_SCALE);

  for (int i = 0; i < FC1_UNITS; i++)
    fc1_out[i] = leaky_relu_requant(fc1_out[i],
                                     DENSE_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE);

  // FC2: 16 -> 8
  static elem_t fc2_out[FC2_UNITS];
  cpu_fc(1, FC1_UNITS, FC2_UNITS, fc1_out,
         (const elem_t *)fc2_weights, (const acc_t *)fc2_bias,
         fc2_out, DENSE_LAYERS_2_GEMM_ACC_SCALE);

  for (int i = 0; i < FC2_UNITS; i++)
    fc2_out[i] = leaky_relu_requant(fc2_out[i],
                                     DENSE_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE);

  // FC3: 8 -> 4
  static elem_t fc3_out[FC3_UNITS];
  cpu_fc(1, FC2_UNITS, FC3_UNITS, fc2_out,
         (const elem_t *)fc3_weights, (const acc_t *)fc3_bias,
         fc3_out, DENSE_LAYERS_4_GEMM_ACC_SCALE);

  for (int i = 0; i < FC3_UNITS; i++)
    fc3_out[i] = leaky_relu_requant(fc3_out[i],
                                     DENSE_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE);

  // FC4: 4 -> 2
  static elem_t fc4_out[FC4_UNITS];
  cpu_fc(1, FC3_UNITS, FC4_UNITS, fc3_out,
         (const elem_t *)fc4_weights, (const acc_t *)fc4_bias,
         fc4_out, DENSE_LAYERS_6_GEMM_ACC_SCALE);

  for (int i = 0; i < FC4_UNITS; i++)
    fc4_out[i] = leaky_relu_requant(fc4_out[i],
                                     DENSE_LAYERS_7_LEAKYRELU_QUANT_ACC_SCALE);

  // Output: 2 -> 2
  cpu_fc(1, FC4_UNITS, OUTPUT_UNITS, fc4_out,
         (const elem_t *)output_weights, (const acc_t *)output_bias,
         output, DENSE_LAYERS_8_GEMM_ACC_SCALE);
}

static inline float fabsf_custom(float x) {
  return x >= 0.0f ? x : -x;
}

static inline unsigned long long read_cycles_x86(void) {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)hi << 32) | lo;
}

int main(void) {
  printf("==============================================\n");
  printf("BraggNN x86 CPU (native C, no accelerator)\n");
  printf("==============================================\n");

  const float DEQUANT_SCALE = 11.0f / 127.0f; // int8 -> pixel coordinates

  elem_t predictions[OUTPUT_UNITS];
  unsigned long long cycle_counts[NUM_TEST_PATCHES];
  unsigned long long total_cycles = 0;
  double elapsed_ms[NUM_TEST_PATCHES];
  double total_ms = 0.0;
  float total_x_error = 0.0f;
  float total_y_error = 0.0f;

  printf("\nInput shape: [%d, %d, %d, %d]\n",
         BATCH, INPUT_CHANNELS, INPUT_DIM, INPUT_DIM);
  printf("Running %d inferences (%d warmup + %d measured)\n",
         NUM_TEST_PATCHES + 1, 1, NUM_TEST_PATCHES);

  // Warmup: drop first run to warm caches
  printf("\n--- Warmup (patch 0) ---\n");
  cpu_inference(test_inputs[0], predictions);
  printf("Warmup done\n");

  // Measured iterations
  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_TEST_PATCHES);

    struct timespec ts_start, ts_end;
    unsigned long long cyc_start, cyc_end;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    cyc_start = read_cycles_x86();
    cpu_inference(test_inputs[i], predictions);
    cyc_end = read_cycles_x86();
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    unsigned long long elapsed_cyc = cyc_end - cyc_start;
    cycle_counts[i] = elapsed_cyc;
    total_cycles += elapsed_cyc;

    float pred_x = predictions[0] * DEQUANT_SCALE;
    float pred_y = predictions[1] * DEQUANT_SCALE;
    float actual_x = test_labels[i][0];
    float actual_y = test_labels[i][1];
    float err_x = pred_x - actual_x;
    float err_y = pred_y - actual_y;

    double dt = (ts_end.tv_sec - ts_start.tv_sec) * 1e3 +
                (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-6;
    elapsed_ms[i] = dt;
    total_ms += dt;
    total_x_error += fabsf_custom(err_x);
    total_y_error += fabsf_custom(err_y);

    printf("cycles: %llu, time: %.3f ms\n", elapsed_cyc, dt);
    printf("pred: (%.4f, %.4f), actual: (%.4f, %.4f), error: (%.4f, %.4f)\n",
           pred_x, pred_y, actual_x, actual_y, err_x, err_y);
  }

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n",
         NUM_TEST_PATCHES, total_cycles / NUM_TEST_PATCHES);
  printf("Avg time over %d runs: %.3f ms\n",
         NUM_TEST_PATCHES, total_ms / NUM_TEST_PATCHES);
  printf("Avg error over %d runs: (%.4f, %.4f)\n",
         NUM_TEST_PATCHES,
         total_x_error / NUM_TEST_PATCHES,
         total_y_error / NUM_TEST_PATCHES);
  printf("==============================================\n");
  printf("BraggNN inference completed successfully\n");
  printf("==============================================\n");

  return 0;
}
