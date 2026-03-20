/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exhaustive ULP evaluation tool for single-argument fast_math functions.
// Uses dispenso::parallel_for for fast evaluation across concentric bands.
// Tests both +x and -x for each representable float in each band.
//
// Usage: buck run <target>
// To add a new function, add an entry to the `funcs` vector in main().

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <dispenso/parallel_for.h>

#include <dispenso/fast_math/fast_math.h>
#include <dispenso/fast_math/util.h>

namespace dfm = dispenso::fast_math;

struct Band {
  float lo, hi;
  const char* label;
};

static constexpr float kPi = 3.14159265358979323846f;

// Bands for periodic functions (sin, cos, tan).
static Band kTrigBands[] = {
    {0.0f, kPi / 2, "0 to pi/2"},
    {kPi / 2, kPi, "pi/2 to pi"},
    {kPi, 2 * kPi, "pi to 2pi"},
    {2 * kPi, 4 * kPi, "2pi to 4pi"},
    {4 * kPi, 8 * kPi, "4pi to 8pi"},
    {16 * kPi, 32 * kPi, "16pi to 32pi"},
    {64 * kPi, 128 * kPi, "64pi to 128pi"},
    {512 * kPi, 1024 * kPi, "512pi to 1024pi"},
    {65536 * kPi, 131072 * kPi, "65536pi to 131072pi"},
    {524288 * kPi, 1048576 * kPi, "524288pi to 1Mpi"},
};

// Bands for [0, FLT_MAX] functions (exp, log, cbrt, sqrt).
static Band kPositiveBands[] = {
    {0.0f, 1.0f, "0 to 1"},
    {1.0f, 10.0f, "1 to 10"},
    {10.0f, 100.0f, "10 to 100"},
    {100.0f, 1000.0f, "100 to 1000"},
    {1000.0f, 1e6f, "1e3 to 1e6"},
    {1e6f, 1e10f, "1e6 to 1e10"},
    {1e10f, 1e20f, "1e10 to 1e20"},
    {1e20f, 1e30f, "1e20 to 1e30"},
    {1e30f, 1e38f, "1e30 to 1e38"},
};

// Bands for [-1, 1] functions (asin, acos).
static Band kUnitBands[] = {
    {0.0f, 0.25f, "0 to 0.25"},
    {0.25f, 0.5f, "0.25 to 0.5"},
    {0.5f, 0.75f, "0.5 to 0.75"},
    {0.75f, 1.0f, "0.75 to 1.0"},
};

// Bands for [0, FLT_MAX] functions (atan).
static Band kAtanBands[] = {
    {0.0f, 0.5f, "0 to 0.5"},
    {0.5f, 1.0f, "0.5 to 1"},
    {1.0f, 2.0f, "1 to 2"},
    {2.0f, 10.0f, "2 to 10"},
    {10.0f, 100.0f, "10 to 100"},
    {100.0f, 1e6f, "100 to 1e6"},
    {1e6f, 1e20f, "1e6 to 1e20"},
    {1e20f, 1e38f, "1e20 to 1e38"},
};

using FnType = float (*)(float);

static void evalBands(const char* name, FnType gt, FnType fn, Band* bands, int32_t numBands) {
  printf("%-20s  %5s  %8s  %-s\n", name, "ULP", ">=3ULP", "Worst input");
  for (int32_t bi = 0; bi < numBands; ++bi) {
    const auto& b = bands[bi];
    uint32_t lo_bits = dfm::bit_cast<uint32_t>(b.lo);
    uint32_t hi_bits = dfm::bit_cast<uint32_t>(b.hi);
    uint32_t count = hi_bits - lo_bits + 1;

    std::atomic<uint32_t> maxUlp{0};
    std::atomic<float> worstX{0.0f};
    std::atomic<uint32_t> count3{0};

    dispenso::parallel_for(
        dispenso::makeChunkedRange(0u, count, 64u * 1024u), [&](uint32_t begin, uint32_t end) {
          uint32_t localMax = 0;
          float localWorst = 0.0f;
          uint32_t localCount3 = 0;
          for (uint32_t idx = begin; idx < end; ++idx) {
            uint32_t bits = lo_bits + idx;
            float f = dfm::bit_cast<float>(bits);
            for (float x : {f, -f}) {
              float g = gt(x);
              float a = fn(x);
              uint32_t d = dfm::float_distance(g, a);
              if (d >= 3)
                localCount3++;
              if (d > localMax) {
                localMax = d;
                localWorst = x;
              }
            }
          }
          count3.fetch_add(localCount3, std::memory_order_relaxed);
          uint32_t prev = maxUlp.load(std::memory_order_relaxed);
          while (localMax > prev) {
            if (maxUlp.compare_exchange_weak(prev, localMax, std::memory_order_relaxed)) {
              worstX.store(localWorst, std::memory_order_relaxed);
              break;
            }
          }
        });

    printf("  %-18s  %5u  %8u  x=%.9g\n", b.label, maxUlp.load(), count3.load(), worstX.load());
  }
}

template <int32_t N>
static void eval(const char* name, FnType gt, FnType fn, Band (&bands)[N]) {
  evalBands(name, gt, fn, bands, N);
}

// Ground truth wrappers — use long double on macOS for better reference accuracy.
static float gt_sin(float x) {
#if defined(__APPLE__)
  return static_cast<float>(::sinl(x));
#else
  return ::sinf(x);
#endif
}
static float gt_cos(float x) {
#if defined(__APPLE__)
  return static_cast<float>(::cosl(x));
#else
  return ::cosf(x);
#endif
}
static float gt_tan(float x) {
  return static_cast<float>(::tanl(x));
}
static float gt_asin(float x) {
  return ::asinf(x);
}
static float gt_acos(float x) {
  return ::acosf(x);
}
static float gt_atan(float x) {
  return ::atanf(x);
}
static float gt_exp(float x) {
  return ::expf(x);
}
static float gt_exp2(float x) {
  return ::exp2f(x);
}
static float gt_log(float x) {
  return ::logf(x);
}
static float gt_log2(float x) {
  return ::log2f(x);
}
static float gt_log10(float x) {
  return ::log10f(x);
}
static float gt_cbrt(float x) {
  return ::cbrtf(x);
}

// Bands for exp functions — [-89, 89] covers the non-overflow range.
static Band kExpBands[] = {
    {0.0f, 1.0f, "0 to 1"},
    {1.0f, 10.0f, "1 to 10"},
    {10.0f, 40.0f, "10 to 40"},
    {40.0f, 89.0f, "40 to 89"},
};

int main() {
  printf("=== Trigonometric ===\n");
  eval("sin", gt_sin, dfm::sin<float>, kTrigBands);
  printf("\n");
  eval("cos", gt_cos, dfm::cos<float>, kTrigBands);
  printf("\n");
  eval("tan", gt_tan, dfm::tan<float>, kTrigBands);

  printf("\n=== Inverse Trig ===\n");
  eval("asin", gt_asin, dfm::asin<float>, kUnitBands);
  printf("\n");
  eval("acos", gt_acos, dfm::acos<float>, kUnitBands);
  printf("\n");
  eval("atan", gt_atan, dfm::atan<float>, kAtanBands);

  printf("\n=== Exponential ===\n");
  eval("exp", gt_exp, dfm::exp<float>, kExpBands);
  printf("\n");
  eval("exp2", gt_exp2, dfm::exp2<float>, kExpBands);

  printf("\n=== Logarithmic ===\n");
  eval("log", gt_log, dfm::log<float>, kPositiveBands);
  printf("\n");
  eval("log2", gt_log2, dfm::log2<float>, kPositiveBands);
  printf("\n");
  eval("log10", gt_log10, dfm::log10<float>, kPositiveBands);

  printf("\n=== Other ===\n");
  eval("cbrt", gt_cbrt, dfm::cbrt<float>, kPositiveBands);

  return 0;
}
