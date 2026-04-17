/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unified test infrastructure for fast_math functions across scalar and SIMD.
//
// Provides:
//   - SimdTestTraits<Flt>: lane count, load/store, lane extraction for any Flt type
//   - evalAccuracy<Flt>(): exhaustive accuracy evaluator that works identically for
//     scalar (float) and SIMD types — same function, same threshold, same call site
//   - checkLaneByLane<Flt>(): verify specific inputs against scalar ground truth
//   - FAST_MATH_ACCURACY_TESTS(): macro that generates one test per available backend,
//     all sharing a single ULP threshold (the source of truth)
//
// Usage:
//   #include "simd_test_utils.h"
//   using namespace dispenso::fast_math::testing;
//
//   static float gt_sin(float x) { return ::sinf(x); }
//   constexpr uint32_t kSinUlps = 1;
//   FAST_MATH_ACCURACY_TESTS(Sin, gt_sin, dfm::sin, -128*kPi, 128*kPi, kSinUlps)

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <dispenso/fast_math/fast_math.h>
#include <dispenso/fast_math/util.h>

#include "eval.h"

#include <gtest/gtest.h>

namespace dispenso {
namespace fast_math {
namespace testing {

// ---------------------------------------------------------------------------
// SimdTestTraits<Flt> — lane count, load, store, lane extraction.
// ---------------------------------------------------------------------------

// Maximum lane count we'll ever encounter (2048-bit vectors).
constexpr int32_t kMaxSimdLanes = 64;

// Primary template: scalar float.
template <typename Flt>
struct SimdTestTraits {
  static int32_t laneCount() {
    return 1;
  }
  static Flt load(const float* data) {
    return *data;
  }
  static void store(Flt v, float* data) {
    *data = v;
  }
  static float lane(Flt v, int32_t /*i*/) {
    return v;
  }
};

// --- x86 SSE ---
#if defined(__SSE4_1__)
template <>
struct SimdTestTraits<__m128> {
  static int32_t laneCount() {
    return 4;
  }
  static __m128 load(const float* data) {
    return _mm_load_ps(data);
  }
  static void store(__m128 v, float* data) {
    _mm_store_ps(data, v);
  }
  static float lane(__m128 v, int32_t i) {
    alignas(16) float buf[4];
    _mm_store_ps(buf, v);
    return buf[i];
  }
};
#endif

// --- x86 AVX2 ---
#if defined(__AVX2__)
template <>
struct SimdTestTraits<__m256> {
  static int32_t laneCount() {
    return 8;
  }
  static __m256 load(const float* data) {
    return _mm256_load_ps(data);
  }
  static void store(__m256 v, float* data) {
    _mm256_store_ps(data, v);
  }
  static float lane(__m256 v, int32_t i) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    return buf[i];
  }
};
#endif

// --- x86 AVX-512 ---
#if defined(__AVX512F__)
template <>
struct SimdTestTraits<__m512> {
  static int32_t laneCount() {
    return 16;
  }
  static __m512 load(const float* data) {
    return _mm512_load_ps(data);
  }
  static void store(__m512 v, float* data) {
    _mm512_store_ps(data, v);
  }
  static float lane(__m512 v, int32_t i) {
    alignas(64) float buf[16];
    _mm512_store_ps(buf, v);
    return buf[i];
  }
};
#endif

// --- ARM NEON ---
#if defined(__aarch64__)
template <>
struct SimdTestTraits<float32x4_t> {
  static int32_t laneCount() {
    return 4;
  }
  static float32x4_t load(const float* data) {
    return vld1q_f32(data);
  }
  static void store(float32x4_t v, float* data) {
    vst1q_f32(data, v);
  }
  static float lane(float32x4_t v, int32_t i) {
    float buf[4];
    vst1q_f32(buf, v);
    return buf[i];
  }
};
#endif

// --- Highway ---
#if __has_include("hwy/highway.h")
template <>
struct SimdTestTraits<HwyFloat> {
  static int32_t laneCount() {
    return static_cast<int32_t>(hwy::HWY_NAMESPACE::Lanes(HwyFloatTag{}));
  }
  static HwyFloat load(const float* data) {
    return hwy::HWY_NAMESPACE::Load(HwyFloatTag{}, data);
  }
  static void store(HwyFloat v, float* data) {
    hwy::HWY_NAMESPACE::Store(v.v, HwyFloatTag{}, data);
  }
  static float lane(HwyFloat v, int32_t i) {
    return hwy::HWY_NAMESPACE::ExtractLane(v.v, static_cast<size_t>(i));
  }
};
#endif

// ---------------------------------------------------------------------------
// evalAccuracy<Flt>() — unified accuracy evaluator for scalar and SIMD.
// ---------------------------------------------------------------------------

// Iterates every representable float in [lo, hi] via detail::nextafter.
// For SIMD types, packs N consecutive floats per call — processes the full
// domain in 1/N the iterations of scalar. Returns max ULP error observed.
//
// Ground truth `gt` is always called with scalar float.
// Approximation `fn` is called with Flt (float or SIMD vector).
template <typename Flt, typename GT, typename FN>
uint32_t evalAccuracy(GT gt, FN fn, float lo, float hi) {
  using Traits = SimdTestTraits<Flt>;
  const int32_t N = Traits::laneCount();
  uint32_t maxUlp = 0;

  if constexpr (std::is_same_v<Flt, float>) {
    // Scalar path — matches existing evalAccuracy in eval.h exactly.
    for (float f = lo; f <= hi; f = detail::nextafter(f)) {
      float expected = gt(f);
      float actual = fn(f);
      uint32_t d = float_distance(expected, actual);
      maxUlp = std::max(maxUlp, d);
      if (d > 100) {
        printf("%d, f(%.9g): %.9g, %.9g\n", d, f, expected, actual);
      }
    }
  } else {
    // SIMD path — pack N floats, call fn, compare per-lane.
    alignas(64) float inputs[kMaxSimdLanes];
    alignas(64) float outputs[kMaxSimdLanes];
    float f = lo;

    while (f <= hi) {
      // Fill input lanes.
      for (int32_t i = 0; i < N; ++i) {
        inputs[i] = f;
        if (f <= hi) {
          f = detail::nextafter(f);
        }
      }

      Flt result = fn(Traits::load(inputs));
      Traits::store(result, outputs);

      // Compare each lane against ground truth.
      for (int32_t i = 0; i < N; ++i) {
        float expected = gt(inputs[i]);
        float actual = outputs[i];
        // Skip non-finite results (NaN/Inf comparisons are meaningless).
        if (!std::isfinite(expected) || !std::isfinite(actual)) {
          continue;
        }
        uint32_t d = float_distance(expected, actual);
        maxUlp = std::max(maxUlp, d);
        if (d > 100) {
          printf("%d, f(%.9g): %.9g, %.9g\n", d, inputs[i], expected, actual);
        }
      }
    }
  }

  return maxUlp;
}

// ---------------------------------------------------------------------------
// checkLaneByLane<Flt>() — verify specific inputs against ground truth.
// ---------------------------------------------------------------------------

// Packs `numInputs` floats into SIMD vectors and verifies each output lane
// against `gt`. For float, just checks one value at a time.
// numInputs does not need to be a multiple of the lane count; excess inputs
// are checked in additional calls.
template <typename Flt, typename GT, typename FN>
void checkLaneByLane(GT gt, FN fn, const float* inputs, int32_t numInputs, uint32_t maxUlps) {
  using Traits = SimdTestTraits<Flt>;
  const int32_t N = Traits::laneCount();

  alignas(64) float buf[kMaxSimdLanes];
  alignas(64) float out[kMaxSimdLanes];

  for (int32_t base = 0; base < numInputs; base += N) {
    // Fill — pad with last valid input if numInputs isn't a multiple of N.
    int32_t count = std::min(N, numInputs - base);
    for (int32_t i = 0; i < count; ++i) {
      buf[i] = inputs[base + i];
    }
    for (int32_t i = count; i < N; ++i) {
      buf[i] = buf[count - 1];
    }

    Flt result = fn(Traits::load(buf));
    Traits::store(result, out);

    for (int32_t i = 0; i < count; ++i) {
      float expected = gt(buf[i]);
      float actual = out[i];
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual)) << "input=" << buf[i] << " expected NaN, got " << actual;
      } else if (std::isinf(expected)) {
        EXPECT_EQ(expected, actual)
            << "input=" << buf[i] << " expected=" << expected << " actual=" << actual;
      } else {
        uint32_t dist = float_distance(expected, actual);
        EXPECT_LE(dist, maxUlps) << "input=" << buf[i] << " expected=" << expected
                                 << " actual=" << actual << " ulps=" << dist;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// FAST_MATH_ACCURACY_TESTS — generate scalar + all SIMD backend tests.
// ---------------------------------------------------------------------------

// Usage:
//   FAST_MATH_ACCURACY_TESTS(Sin, gt_sin, dfm::sin, -128*kPi, 128*kPi, 1)
//
// Expands to one TEST per available backend:
//   TEST(Sin, Scalar)        — evalAccuracy<float>
//   TEST(SinSse, Range)      — evalAccuracy<__m128>
//   TEST(SinAvx, Range)      — evalAccuracy<__m256>
//   TEST(SinAvx512, Range)   — evalAccuracy<__m512>
//   TEST(SinNeon, Range)     — evalAccuracy<float32x4_t>
//   TEST(SinHwy, Range)      — evalAccuracy<HwyFloat>
//
// For traits variants, define a wrapper template:
//   template <typename Flt>
//   Flt sin_max(Flt x) { return dfm::sin<Flt, dfm::MaxAccuracyTraits>(x); }
//   FAST_MATH_ACCURACY_TESTS(SinMax, gt_sin, sin_max, ...)

// Per-backend macros: expand to a TEST or to nothing based on ISA availability.

#if defined(__SSE4_1__)
#define FAST_MATH_SSE_TEST(Suite, gt, func, lo, hi, maxUlps)                                     \
  TEST(Suite##Sse, Range) {                                                                      \
    EXPECT_LE((evalAccuracy<__m128>(gt, func<__m128>, lo, hi)), static_cast<uint32_t>(maxUlps)); \
  }
#else
#define FAST_MATH_SSE_TEST(Suite, gt, func, lo, hi, maxUlps)
#endif

#if defined(__AVX2__)
#define FAST_MATH_AVX_TEST(Suite, gt, func, lo, hi, maxUlps)                                     \
  TEST(Suite##Avx, Range) {                                                                      \
    EXPECT_LE((evalAccuracy<__m256>(gt, func<__m256>, lo, hi)), static_cast<uint32_t>(maxUlps)); \
  }
#else
#define FAST_MATH_AVX_TEST(Suite, gt, func, lo, hi, maxUlps)
#endif

#if defined(__AVX512F__)
#define FAST_MATH_AVX512_TEST(Suite, gt, func, lo, hi, maxUlps)                                  \
  TEST(Suite##Avx512, Range) {                                                                   \
    EXPECT_LE((evalAccuracy<__m512>(gt, func<__m512>, lo, hi)), static_cast<uint32_t>(maxUlps)); \
  }
#else
#define FAST_MATH_AVX512_TEST(Suite, gt, func, lo, hi, maxUlps)
#endif

#if defined(__aarch64__)
#define FAST_MATH_NEON_TEST(Suite, gt, func, lo, hi, maxUlps)       \
  TEST(Suite##Neon, Range) {                                        \
    EXPECT_LE(                                                      \
        (evalAccuracy<float32x4_t>(gt, func<float32x4_t>, lo, hi)), \
        static_cast<uint32_t>(maxUlps));                            \
  }
#else
#define FAST_MATH_NEON_TEST(Suite, gt, func, lo, hi, maxUlps)
#endif

#if __has_include("hwy/highway.h")
#define FAST_MATH_HWY_TEST(Suite, gt, func, lo, hi, maxUlps)   \
  TEST(Suite##Hwy, Range) {                                    \
    EXPECT_LE(                                                 \
        (evalAccuracy<dispenso::fast_math::HwyFloat>(          \
            gt, func<dispenso::fast_math::HwyFloat>, lo, hi)), \
        static_cast<uint32_t>(maxUlps));                       \
  }
#else
#define FAST_MATH_HWY_TEST(Suite, gt, func, lo, hi, maxUlps)
#endif

// Main macro: generates one test per available backend.
#define FAST_MATH_ACCURACY_TESTS(Suite, gt, func, lo, hi, maxUlps)                             \
  TEST(Suite, Scalar) {                                                                        \
    EXPECT_LE((evalAccuracy<float>(gt, func<float>, lo, hi)), static_cast<uint32_t>(maxUlps)); \
  }                                                                                            \
  FAST_MATH_SSE_TEST(Suite, gt, func, lo, hi, maxUlps)                                         \
  FAST_MATH_AVX_TEST(Suite, gt, func, lo, hi, maxUlps)                                         \
  FAST_MATH_AVX512_TEST(Suite, gt, func, lo, hi, maxUlps)                                      \
  FAST_MATH_NEON_TEST(Suite, gt, func, lo, hi, maxUlps)                                        \
  FAST_MATH_HWY_TEST(Suite, gt, func, lo, hi, maxUlps)

// ---------------------------------------------------------------------------
// FAST_MATH_SPECIAL_TESTS — test hand-picked special values across backends.
// ---------------------------------------------------------------------------
//
// Usage:
//   static const float kAtanSpecials[] = {0.0f, -1e7f, 1e7f, NaN, Inf, -Inf};
//   FAST_MATH_SPECIAL_TESTS(AtanSpecial, ::atanf, dfm::atan, kAtanSpecials, 0)
//
// Uses checkLaneByLane to verify each output lane against scalar ground truth.
// NaN inputs are checked for NaN output; finite inputs are checked within maxUlps.

#if defined(__SSE4_1__)
#define FAST_MATH_SSE_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite##Sse, SpecialVals) {                                    \
    checkLaneByLane<__m128>(                                         \
        gt,                                                          \
        func<__m128>,                                                \
        inputs,                                                      \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])),    \
        maxUlps);                                                    \
  }
#else
#define FAST_MATH_SSE_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)
#endif

#if defined(__AVX2__)
#define FAST_MATH_AVX_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite##Avx, SpecialVals) {                                    \
    checkLaneByLane<__m256>(                                         \
        gt,                                                          \
        func<__m256>,                                                \
        inputs,                                                      \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])),    \
        maxUlps);                                                    \
  }
#else
#define FAST_MATH_AVX_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)
#endif

#if defined(__AVX512F__)
#define FAST_MATH_AVX512_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite##Avx512, SpecialVals) {                                    \
    checkLaneByLane<__m512>(                                            \
        gt,                                                             \
        func<__m512>,                                                   \
        inputs,                                                         \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])),       \
        maxUlps);                                                       \
  }
#else
#define FAST_MATH_AVX512_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)
#endif

#if defined(__aarch64__)
#define FAST_MATH_NEON_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite##Neon, SpecialVals) {                                    \
    checkLaneByLane<float32x4_t>(                                     \
        gt,                                                           \
        func<float32x4_t>,                                            \
        inputs,                                                       \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])),     \
        maxUlps);                                                     \
  }
#else
#define FAST_MATH_NEON_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)
#endif

#if __has_include("hwy/highway.h")
#define FAST_MATH_HWY_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite##Hwy, SpecialVals) {                                    \
    checkLaneByLane<dispenso::fast_math::HwyFloat>(                  \
        gt,                                                          \
        func<dispenso::fast_math::HwyFloat>,                         \
        inputs,                                                      \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])),    \
        maxUlps);                                                    \
  }
#else
#define FAST_MATH_HWY_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)
#endif

// Main macro: generates one special-values test per available backend.
#define FAST_MATH_SPECIAL_TESTS(Suite, gt, func, inputs, maxUlps) \
  TEST(Suite, SpecialScalar) {                                    \
    checkLaneByLane<float>(                                       \
        gt,                                                       \
        func<float>,                                              \
        inputs,                                                   \
        static_cast<int32_t>(sizeof(inputs) / sizeof(inputs[0])), \
        maxUlps);                                                 \
  }                                                               \
  FAST_MATH_SSE_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)    \
  FAST_MATH_AVX_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)    \
  FAST_MATH_AVX512_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps) \
  FAST_MATH_NEON_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)   \
  FAST_MATH_HWY_SPECIAL_TEST(Suite, gt, func, inputs, maxUlps)

} // namespace testing
} // namespace fast_math
} // namespace dispenso
