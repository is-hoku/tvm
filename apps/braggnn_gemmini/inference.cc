#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>

#include <include/gemmini_params.h>
#include <include/gemmini.h>

#include <cmath>
#include <cstdio>

#include <unordered_map>
#include <vector>
#include <string>

static inline unsigned long long read_cycles() {
  uint64_t cycles;
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return cycles;
}

#define NUM_TEST_PATCHES 10
#define GEMMINI_NUM_CLOCK_CYCLES 14  // must match kGemminiNumClockCycles in codegen.cc

// Test patches and labels below are taken from
// 3rdparty/gemmini/bareMetalC/braggnn.h (test_input_0..9 / test_label_0..9).

// Patch 0: dataset index 0
static float kPatch0[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.748000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.646286f, 0.753714f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 1: dataset index 1533
static float kPatch1[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.679580f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.884584f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 2: dataset index 3066
static float kPatch2[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.671045f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.832239f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 3: dataset index 4599
static float kPatch3[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.713495f, 0.444870f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.385270f, 0.607067f, 1.000000f, 0.448276f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 4: dataset index 6132
static float kPatch4[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.036218f, 0.084531f, 0.050424f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.123966f, 0.714511f, 0.228968f, 0.032298f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.153516f, 1.000000f, 0.321540f, 0.039736f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.058867f, 0.204476f, 0.120749f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.031460f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 5: dataset index 7665
static float kPatch5[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.534039f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.603631f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 6: dataset index 9198
static float kPatch6[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.053925f, 0.123592f, 0.106640f, 0.051948f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.054531f, 0.281332f, 0.533199f, 0.260868f, 0.081736f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.138365f, 0.608032f, 1.000000f, 0.362866f, 0.055580f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.051544f, 0.216186f, 0.880404f, 0.686499f, 0.072371f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.053764f, 0.135096f, 0.123875f, 0.041615f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 7: dataset index 10731
static float kPatch7[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.006601f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.008891f, 0.020313f, 0.029576f, 0.021196f, 0.008912f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.005994f, 0.022838f, 0.108643f, 0.236791f, 0.111360f, 0.025610f, 0.006539f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.008318f, 0.043585f, 0.375003f, 1.000000f, 0.437695f, 0.049959f, 0.008194f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.007636f, 0.034963f, 0.228459f, 0.886929f, 0.306242f, 0.042530f, 0.008001f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.015347f, 0.047441f, 0.084612f, 0.054090f, 0.017478f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.005828f, 0.010319f, 0.013450f, 0.011139f, 0.006497f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 8: dataset index 12264
static float kPatch8[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.005956f, 0.005746f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.014282f, 0.029527f, 0.028048f, 0.011841f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.011081f, 0.056096f, 0.238751f, 0.189008f, 0.038369f, 0.008020f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.015705f, 0.119878f, 1.000000f, 0.830623f, 0.073691f, 0.011353f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.012622f, 0.074632f, 0.411532f, 0.359117f, 0.052916f, 0.008982f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.006388f, 0.020405f, 0.050169f, 0.046724f, 0.016876f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.006039f, 0.009108f, 0.008578f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

// Patch 9: dataset index 13798
static float kPatch9[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.146968f, 0.305620f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.374985f, 1.000000f, 0.148690f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.315213f, 0.935678f, 0.130857f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.107736f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};

static float* kPatches[NUM_TEST_PATCHES] = {
  (float*)kPatch0, (float*)kPatch1, (float*)kPatch2, (float*)kPatch3, (float*)kPatch4,
  (float*)kPatch5, (float*)kPatch6, (float*)kPatch7, (float*)kPatch8, (float*)kPatch9,
};

// Expected (x, y) labels, pixel coordinates in the 11x11 patch.
static float kLabels[NUM_TEST_PATCHES][2] = {
  {5.368042f, 5.426758f},
  {5.698364f, 5.197388f},
  {5.770508f, 5.900269f},
  {5.868042f, 5.709488f},
  {5.960708f, 5.935608f},
  {5.454346f, 5.076599f},
  {5.915284f, 5.269226f},
  {5.022949f, 5.347900f},
  {5.412720f, 5.121765f},
  {5.842285f, 5.395447f},
};

// Refrence implementation is in /3rdparty/tvm-ffi/examples/quickstart/load/load_cpp.cc
struct CPUAllocator {
  void AllocData(DLTensor* tensor) {
	int64_t numel = 1;
	for (int i = 0; i < tensor->ndim; ++i) {
		numel *= tensor->shape[i];
	}
	int64_t itemsize = (tensor->dtype.bits * tensor->dtype.lanes + 7) / 8;
    tensor->data = std::malloc(numel * itemsize);
  }
  void FreeData(DLTensor* tensor) { std::free(tensor->data); }
};

int main(int argc, char* argv[]){
	// Reference implementation is in /src/runtime/disco/builtin.cc
	std::string path = argc > 1 ? argv[1] : "./braggnn.so";
	tvm::ffi::Module ex = tvm::ffi::Module::LoadFromFile(path);
	//tvm::ObjectPtr<tvm::runtime::vm::VirtualMachine> vm = tvm::runtime::vm::VirtualMachine::Create();

	tvm::ffi::Optional<tvm::ffi::Function> vm_load_executable = ex->GetFunction("vm_load_executable");
	if (!vm_load_executable.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "File `" << path
        << "` does not contain a Relax VM executable";
		return 1;
	}
	auto mod = (*vm_load_executable)().cast<tvm::ffi::Module>();
	tvm::ffi::Optional<tvm::ffi::Function> vm_initialization = mod->GetFunction("vm_initialization");

  if (!vm_initialization.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "File `" << path
        << "` is not built by RelaxVM, because `vm_initialization` does not exist";
	return 1;
  }
  DLDevice dev{kDLCPU, 0};
  (*vm_initialization)(static_cast<int>(dev.device_type), static_cast<int>(dev.device_id),
                       static_cast<int>(tvm::runtime::AllocatorType::kPooled),
                       static_cast<int>(kDLCPU), 0,
                       static_cast<int>(tvm::runtime::AllocatorType::kPooled));

  tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(CPUAllocator(), {1, 11, 11, 1}, DLDataType{kDLFloat, 32, 1}, dev);


  // vm.set_instrument()
  static std::unordered_map<std::string, unsigned long long> g_call_cycle_totals;
  static std::unordered_map<std::string, unsigned long long> g_call_counts;
  static std::vector<unsigned long long> g_call_stack;
  
  tvm::ffi::Optional<tvm::ffi::Function> set_instrument = mod->GetFunction("set_instrument");
  if (set_instrument.has_value()) {
    auto instrument_fn = tvm::ffi::Function::FromPacked(
        [](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
          // args[0] = func (VMClosure/PackedFunc)
          // args[1] = func_symbol
          // args[2] = before_run
          // args[3] = ret_value (valid when before_run==false)
          std::string func_symbol = args[1].cast<std::string>();
          bool before_run = args[2].cast<bool>();
  
          if (before_run) {
            g_call_stack.push_back(read_cycles());
          } else {
            unsigned long long start = g_call_stack.back();
            g_call_stack.pop_back();
            unsigned long long elapsed = read_cycles() - start;
            g_call_cycle_totals[func_symbol] += elapsed;
            g_call_counts[func_symbol] += 1;
          }
          *rv = int64_t(0);  // VMInstrumentReturnKind::kNoOp
        });
    (*set_instrument)(instrument_fn);
  }



  tvm::ffi::Function main_func = mod->GetFunction("main").value();

  tvm::ffi::Optional<tvm::ffi::Function> clock_cycles_func =
      ex->GetFunction("gemmini_clock_cycles", /*query_imports=*/true);
  bool have_clock_cycles = clock_cycles_func.has_value();
  tvm::ffi::Tensor clock_cycles = tvm::ffi::Tensor::FromNDAlloc(
      CPUAllocator(), {GEMMINI_NUM_CLOCK_CYCLES}, DLDataType{kDLUInt, 64, 1}, dev);
  unsigned long long layer_cycles_total[GEMMINI_NUM_CLOCK_CYCLES] = {0};

  auto print_layer_cycles = [&](unsigned long long* accumulate_into) {
    if (!have_clock_cycles) return;
    (*clock_cycles_func)(clock_cycles);
    const uint64_t* c = static_cast<const uint64_t*>(clock_cycles.data_ptr());
    unsigned long long sum = 0;
    printf("layer cycles:");
    for (int k = 0; k < GEMMINI_NUM_CLOCK_CYCLES; k++) {
      printf(" %llu", (unsigned long long)c[k]);
      sum += c[k];
      if (accumulate_into) accumulate_into[k] += c[k];
    }
    printf(" (sum=%llu)\n", sum);
  };

  printf("==============================================\n");
  printf("BraggNN Gemmini through TVM Relax VM\n");
  printf("==============================================\n");
  printf("\nInput shape: [1, 11, 11, 1]\n");
  printf("Running %d inferences (1 warmup + %d measured)\n",
         NUM_TEST_PATCHES + 1, NUM_TEST_PATCHES);

  gemmini_flush(0);

  // Warmup: drop first run to warm caches (uses patch 5, not counted below).
  printf("\n--- Warmup ---\n");
  std::memcpy(input.data_ptr(), kPatches[5], 11 * 11 * sizeof(float));
  unsigned long long start = read_cycles();
  tvm::ffi::Tensor output = main_func(input).cast<tvm::ffi::Tensor>();
  unsigned long long end = read_cycles();
  {
    const float* out_data = static_cast<const float*>(output.data_ptr());
    printf("cycles: %llu\n", end - start);
    printf("(x, y) = (%.4f, %.4f)\n", out_data[0] * 11, out_data[1] * 11);
    print_layer_cycles(nullptr);  // warmup: not counted in the running average
  }

  g_call_cycle_totals.clear();
  g_call_counts.clear();

  unsigned long long total_cycles = 0;
  double total_abs_error_x = 0.0;
  double total_abs_error_y = 0.0;

  for (int i = 0; i < NUM_TEST_PATCHES; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_TEST_PATCHES);

    std::memcpy(input.data_ptr(), kPatches[i], 11 * 11 * sizeof(float));

    start = read_cycles();
    asm volatile(".word 0x8013");  // TracerV start trigger
    output = main_func(input).cast<tvm::ffi::Tensor>();
    asm volatile(".word 0x10013"); // TracerV end trigger
    end = read_cycles();

    unsigned long long elapsed = end - start;
    total_cycles += elapsed;

    const float* out_data = static_cast<const float*>(output.data_ptr());
    float pred_x = out_data[0] * 11;
    float pred_y = out_data[1] * 11;
    float label_x = kLabels[i][0];
    float label_y = kLabels[i][1];
    float abs_err_x = std::fabs(pred_x - label_x);
    float abs_err_y = std::fabs(pred_y - label_y);
    total_abs_error_x += abs_err_x;
    total_abs_error_y += abs_err_y;

    printf("cycles: %llu\n", elapsed);
    printf("predicted (x, y) = (%.4f, %.4f)\n", pred_x, pred_y);
    printf("label     (x, y) = (%.4f, %.4f)\n", label_x, label_y);
    printf("abs error (x, y) = (%.4f, %.4f)\n", abs_err_x, abs_err_y);
    print_layer_cycles(layer_cycles_total);
  }

  double mae_x = total_abs_error_x / NUM_TEST_PATCHES;
  double mae_y = total_abs_error_y / NUM_TEST_PATCHES;

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_TEST_PATCHES,
         total_cycles / NUM_TEST_PATCHES);
  printf("MAE_x over %d runs: %.4f\n", NUM_TEST_PATCHES, mae_x);
  printf("MAE_y over %d runs: %.4f\n", NUM_TEST_PATCHES, mae_y);
  printf("==============================================\n");
  printf("BraggNN inference completed successfully\n");
  printf("==============================================\n");


  printf("\n--- Per-Call cycle breakdown (VM instrument) ---\n");
  for (auto& kv : g_call_cycle_totals) {
    printf("%-100s total=%12llu  calls/inference=%4llu  total/calls=%10llu  calls/inference=%10llu  total/inference=%10llu\n",
           kv.first.c_str(), kv.second, g_call_counts[kv.first],
           kv.second / g_call_counts[kv.first],
		   g_call_counts[kv.first] / NUM_TEST_PATCHES,
		   kv.second / NUM_TEST_PATCHES);
  }

  return 0;
}
