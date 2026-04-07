/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exhaustive ULP evaluation for fast_math::pow across scalar and SIMD backends.
// Uses Halton low-discrepancy sampling over 2D (base, exponent) domains.
//
// Usage: buck run <target> [-- [--samples N] [--filter name]]

#include <cmath>

#include <dispenso/fast_math/fast_math.h>

#include "bivariate_ulp_eval.h"

namespace dfm = dispenso::fast_math;

// --- Ground truth (double-precision internally) ---

static float gt_pow(float x, float y) {
  return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

// --- Domains ---
// Positive bases only (negative base + non-integer exp is NaN by definition).

static dfm::Domain2D kPowDomains[] = {
    {0.01f, 100.0f, -8.0f, 8.0f, "moderate base, moderate exp"},
    {1e-10f, 1e10f, -4.0f, 4.0f, "wide base, small exp"},
    {0.5f, 2.0f, -50.0f, 50.0f, "near-unity base, large exp"},
    {1e-6f, 1e-1f, 0.1f, 4.0f, "small base, positive exp"},
    {1e2f, 1e20f, -2.0f, 2.0f, "large base, small exp"},
    {0.01f, 100.0f, 1.0f, 50.0f, "moderate base, integer-like exp"},
};

// --- Batch runners ---

static void pow_scalar(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i)
    out[i] = dfm::pow(xs[i], ys[i]);
}

static void pow_scalar_accurate(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i)
    out[i] = dfm::pow<float, dfm::MaxAccuracyTraits>(xs[i], ys[i]);
}

#if defined(__SSE4_1__)
static void pow_sse(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 4) {
    dfm::SseFloat vx{_mm_load_ps(&xs[i])};
    dfm::SseFloat vy{_mm_load_ps(&ys[i])};
    dfm::SseFloat vr = dfm::pow(vx, vy);
    _mm_store_ps(&out[i], vr.v);
  }
}

static void pow_sse_accurate(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 4) {
    dfm::SseFloat vx{_mm_load_ps(&xs[i])};
    dfm::SseFloat vy{_mm_load_ps(&ys[i])};
    dfm::SseFloat vr = dfm::pow<dfm::SseFloat, dfm::MaxAccuracyTraits>(vx, vy);
    _mm_store_ps(&out[i], vr.v);
  }
}
#endif // __SSE4_1__

#if defined(__AVX2__)
static void pow_avx(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 8) {
    dfm::AvxFloat vx{_mm256_load_ps(&xs[i])};
    dfm::AvxFloat vy{_mm256_load_ps(&ys[i])};
    dfm::AvxFloat vr = dfm::pow(vx, vy);
    _mm256_store_ps(&out[i], vr.v);
  }
}

static void pow_avx_accurate(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 8) {
    dfm::AvxFloat vx{_mm256_load_ps(&xs[i])};
    dfm::AvxFloat vy{_mm256_load_ps(&ys[i])};
    dfm::AvxFloat vr = dfm::pow<dfm::AvxFloat, dfm::MaxAccuracyTraits>(vx, vy);
    _mm256_store_ps(&out[i], vr.v);
  }
}
#endif // __AVX2__

#if defined(__AVX512F__)
static void pow_avx512(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 16) {
    dfm::Avx512Float vx{_mm512_load_ps(&xs[i])};
    dfm::Avx512Float vy{_mm512_load_ps(&ys[i])};
    dfm::Avx512Float vr = dfm::pow(vx, vy);
    _mm512_store_ps(&out[i], vr.v);
  }
}

static void pow_avx512_accurate(const float* xs, const float* ys, float* out, int32_t n) {
  for (int32_t i = 0; i < n; i += 16) {
    dfm::Avx512Float vx{_mm512_load_ps(&xs[i])};
    dfm::Avx512Float vy{_mm512_load_ps(&ys[i])};
    dfm::Avx512Float vr = dfm::pow<dfm::Avx512Float, dfm::MaxAccuracyTraits>(vx, vy);
    _mm512_store_ps(&out[i], vr.v);
  }
}
#endif // __AVX512F__

// --- Main ---

int main(int argc, char** argv) {
  auto opts = dfm::parseEvalOptions(argc, argv);
  uint64_t n = opts.numSamples;

  printf(
      "pow ULP evaluation — %llu Halton samples per domain\n\n",
      static_cast<unsigned long long>(n));

  if (dfm::shouldRun(opts, "scalar")) {
    printf("=== pow scalar ===\n");
    dfm::evalFunc2D("pow scalar", gt_pow, pow_scalar, kPowDomains, n);
    printf("\n");
    printf("=== pow scalar MaxAccuracy ===\n");
    dfm::evalFunc2D("pow scalar accurate", gt_pow, pow_scalar_accurate, kPowDomains, n);
    printf("\n");
  }

#if defined(__SSE4_1__)
  if (dfm::shouldRun(opts, "sse")) {
    printf("=== pow SSE ===\n");
    dfm::evalFunc2D("pow SSE", gt_pow, pow_sse, kPowDomains, n);
    printf("\n");
    printf("=== pow SSE MaxAccuracy ===\n");
    dfm::evalFunc2D("pow SSE accurate", gt_pow, pow_sse_accurate, kPowDomains, n);
    printf("\n");
  }
#endif

#if defined(__AVX2__)
  if (dfm::shouldRun(opts, "avx")) {
    printf("=== pow AVX ===\n");
    dfm::evalFunc2D("pow AVX", gt_pow, pow_avx, kPowDomains, n);
    printf("\n");
    printf("=== pow AVX MaxAccuracy ===\n");
    dfm::evalFunc2D("pow AVX accurate", gt_pow, pow_avx_accurate, kPowDomains, n);
    printf("\n");
  }
#endif

#if defined(__AVX512F__)
  if (dfm::shouldRun(opts, "avx512")) {
    printf("=== pow AVX-512 ===\n");
    dfm::evalFunc2D("pow AVX-512", gt_pow, pow_avx512, kPowDomains, n);
    printf("\n");
    printf("=== pow AVX-512 MaxAccuracy ===\n");
    dfm::evalFunc2D("pow AVX-512 accurate", gt_pow, pow_avx512_accurate, kPowDomains, n);
    printf("\n");
  }
#endif

  return 0;
}
