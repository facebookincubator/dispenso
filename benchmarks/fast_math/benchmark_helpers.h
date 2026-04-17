/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <benchmark/benchmark.h>
#include <dispenso/fast_math/fast_math.h>

#if __has_include("hwy/highway.h")
#include "hwy/highway.h"
#endif

namespace dispenso {
namespace fast_math {
namespace bench {

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;
constexpr int32_t kMaxBenchLanes = 64;

// --- Per-type primitives ---
//
// Each SIMD type needs: laneCount, loadVec, zeroVec, addVec, consumeResult.
// These are template specializations so the compiler sees the exact operations
// at each call site, ensuring zero overhead.

template <typename Flt>
inline int32_t laneCount();
template <typename Flt>
inline Flt loadVec(const float* data);
template <typename Flt>
inline Flt zeroVec();
template <typename Flt>
inline Flt addVec(Flt a, Flt b);
template <typename Flt>
inline void consumeResult(Flt sum);

// --- Scalar (float) ---

template <>
inline int32_t laneCount<float>() {
  return 1;
}
template <>
inline float loadVec<float>(const float* data) {
  return *data;
}
template <>
inline float zeroVec<float>() {
  return 0.0f;
}
template <>
inline float addVec<float>(float a, float b) {
  return a + b;
}
template <>
inline void consumeResult<float>(float sum) {
  std::cout << sum << std::endl;
}

// --- SSE (__m128) ---

#if defined(__SSE4_1__)

template <>
inline int32_t laneCount<__m128>() {
  return 4;
}
template <>
inline __m128 loadVec<__m128>(const float* data) {
  return _mm_load_ps(data);
}
template <>
inline __m128 zeroVec<__m128>() {
  return _mm_setzero_ps();
}
template <>
inline __m128 addVec<__m128>(__m128 a, __m128 b) {
  return _mm_add_ps(a, b);
}
template <>
inline void consumeResult<__m128>(__m128 sum) {
  alignas(16) float buf[4];
  _mm_store_ps(buf, sum);
  std::cout << buf[0] + buf[1] + buf[2] + buf[3] << std::endl;
}

#endif // __SSE4_1__

// --- AVX (__m256) ---

#if defined(__AVX2__)

template <>
inline int32_t laneCount<__m256>() {
  return 8;
}
template <>
inline __m256 loadVec<__m256>(const float* data) {
  return _mm256_load_ps(data);
}
template <>
inline __m256 zeroVec<__m256>() {
  return _mm256_setzero_ps();
}
template <>
inline __m256 addVec<__m256>(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
}
template <>
inline void consumeResult<__m256>(__m256 sum) {
  alignas(32) float buf[8];
  _mm256_store_ps(buf, sum);
  float total = 0.0f;
  for (int32_t i = 0; i < 8; ++i)
    total += buf[i];
  std::cout << total << std::endl;
}

#endif // __AVX2__

// --- AVX-512 (__m512) ---

#if defined(__AVX512F__)

template <>
inline int32_t laneCount<__m512>() {
  return 16;
}
template <>
inline __m512 loadVec<__m512>(const float* data) {
  return _mm512_load_ps(data);
}
template <>
inline __m512 zeroVec<__m512>() {
  return _mm512_setzero_ps();
}
template <>
inline __m512 addVec<__m512>(__m512 a, __m512 b) {
  return _mm512_add_ps(a, b);
}
template <>
inline void consumeResult<__m512>(__m512 sum) {
  alignas(64) float buf[16];
  _mm512_store_ps(buf, sum);
  float total = 0.0f;
  for (int32_t i = 0; i < 16; ++i)
    total += buf[i];
  std::cout << total << std::endl;
}

#endif // __AVX512F__

// --- NEON (float32x4_t) ---

#if defined(__aarch64__)

template <>
inline int32_t laneCount<float32x4_t>() {
  return 4;
}
template <>
inline float32x4_t loadVec<float32x4_t>(const float* data) {
  return vld1q_f32(data);
}
template <>
inline float32x4_t zeroVec<float32x4_t>() {
  return vdupq_n_f32(0.0f);
}
template <>
inline float32x4_t addVec<float32x4_t>(float32x4_t a, float32x4_t b) {
  return vaddq_f32(a, b);
}
template <>
inline void consumeResult<float32x4_t>(float32x4_t sum) {
  alignas(16) float buf[4];
  vst1q_f32(buf, sum);
  std::cout << buf[0] + buf[1] + buf[2] + buf[3] << std::endl;
}

#endif // __aarch64__

// --- Highway (HwyFloat) ---

#if __has_include("hwy/highway.h")

namespace hn = hwy::HWY_NAMESPACE;

template <>
inline int32_t laneCount<HwyFloat>() {
  return static_cast<int32_t>(hn::Lanes(HwyFloatTag()));
}
template <>
inline HwyFloat loadVec<HwyFloat>(const float* data) {
  return hn::LoadU(HwyFloatTag(), data);
}
template <>
inline HwyFloat zeroVec<HwyFloat>() {
  return hn::Zero(HwyFloatTag());
}
template <>
inline HwyFloat addVec<HwyFloat>(HwyFloat a, HwyFloat b) {
  return hn::Add(a.v, b.v);
}
template <>
inline void consumeResult<HwyFloat>(HwyFloat sum) {
  const HwyFloatTag d;
  constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
  HWY_ALIGN float buf[kMaxLanes];
  hn::StoreU(sum.v, d, buf);
  float total = 0.0f;
  const size_t N = hn::Lanes(d);
  for (size_t i = 0; i < N; ++i) {
    total += buf[i];
  }
  std::cout << total << std::endl;
}

#endif // hwy/highway.h

// --- Input generation ---
//
// makeInputs<Flt>(lo, hi) generates kNumInputs vectors covering the scalar
// range [lo, hi] at delta = (hi - lo) / kNumInputs spacing. Each SIMD vector
// packs N consecutive values, so the total coverage is N * (hi - lo).

template <typename Flt>
inline std::vector<Flt> makeInputs(float lo, float hi) {
  const int32_t N = laneCount<Flt>();
  float delta = (hi - lo) / static_cast<float>(kNumInputs);
  std::vector<Flt> inputs;
  inputs.reserve(kNumInputs);
  alignas(64) float buf[kMaxBenchLanes];
  float f = lo;
  for (size_t i = 0; i < kNumInputs; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      buf[j] = f + static_cast<float>(j) * delta;
    }
    inputs.push_back(loadVec<Flt>(buf));
    f += static_cast<float>(N) * delta;
  }
  return inputs;
}

// --- Pre-defined input factories ---
//
// These match the ranges used across all SIMD benchmark files.

template <typename Flt>
inline const std::vector<Flt>& sinInputs() {
  static auto v = makeInputs<Flt>(static_cast<float>(-M_PI / 2.0), static_cast<float>(M_PI / 2.0));
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& expInputs() {
  static auto v = makeInputs<Flt>(-10.0f, 10.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& logInputs() {
  static auto v = makeInputs<Flt>(0.001f, 10000.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& acosInputs() {
  static auto v = makeInputs<Flt>(-0.999f, 0.999f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& hypotInputs() {
  static auto v = makeInputs<Flt>(-100000.0f, 100000.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& tanhInputs() {
  static auto v = makeInputs<Flt>(-5.0f, 5.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& erfInputs() {
  static auto v = makeInputs<Flt>(-4.0f, 4.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& powBaseInputs() {
  static auto v = makeInputs<Flt>(0.01f, 100.0f);
  return v;
}

template <typename Flt>
inline const std::vector<Flt>& powExpInputs() {
  static auto v = makeInputs<Flt>(-8.0f, 8.0f);
  return v;
}

// --- Benchmark runners ---
//
// runBench: one-arg function benchmark. Func is a template parameter (lambda),
// so the compiler sees the exact call target and inlines at -O2. This produces
// the same assembly as hand-written code.
//
// runBench2: two-arg function benchmark (atan2, hypot, pow).

template <typename Flt, typename Func>
inline void runBench(benchmark::State& state, const std::vector<Flt>& inputs, Func fn) {
  size_t idx = 0;
  Flt sum = zeroVec<Flt>();
  for (auto _ : state) {
    (void)_;
    sum = addVec<Flt>(sum, fn(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(laneCount<Flt>()));
  consumeResult(sum);
}

template <typename Flt, typename Func>
inline void runBench2(
    benchmark::State& state,
    const std::vector<Flt>& xInputs,
    const std::vector<Flt>& yInputs,
    Func fn) {
  size_t idx = 0;
  Flt sum = zeroVec<Flt>();
  for (auto _ : state) {
    (void)_;
    sum = addVec<Flt>(sum, fn(xInputs[idx], yInputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(laneCount<Flt>()));
  consumeResult(sum);
}

} // namespace bench
} // namespace fast_math
} // namespace dispenso
