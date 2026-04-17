/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark: erf S16 (float, t-substitution + inline exp) vs S21 (double, pure polynomial Estrin).
// Also benchmarks libc erff for reference.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <benchmark/benchmark.h>
#include <dispenso/fast_math/fast_math.h>

#if defined(__SSE4_1__)
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED_VAR myLocalForLoopVar __attribute__((unused))
#elif defined(_MSC_VER)
#define UNUSED_VAR myLocalForLoopVar __pragma(warning(suppress : 4100))
#else
#define UNUSED_VAR myLocalForLoopVar
#endif

namespace dfm = dispenso::fast_math;

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;

// --- Inputs: [-4, 4] ---

const std::vector<float>& erfScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = 8.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -4.0f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

const std::vector<__m128>& erfSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = 8.0f / kNumInputs;
    std::vector<__m128> inp;
    float f = -4.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
    }
    return inp;
  }();
  return inputs;
}

#if defined(__AVX2__)
const std::vector<__m256>& erfAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 8.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -4.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm256_set_ps(
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 8 * delta;
    }
    return inp;
  }();
  return inputs;
}
#endif

// --- S16: float, t-substitution + inline exp, 2 ULP ---

static inline float erf_s16(float x) {
  float ax = std::fabs(x);
  float result;
  if (ax >= 3.92f) {
    result = 1.0f;
  } else if (ax >= 0.875f) {
    constexpr float p = 0.45f;
    float t = 1.0f / std::fma(p, ax, 1.0f);

    constexpr float c0 = 0x1.04873ep-2f;
    constexpr float c1 = 0x1.f81fc6p-3f;
    constexpr float c2 = 0x1.189f42p-2f;
    constexpr float c3 = 0x1.15aaa6p-5f;
    constexpr float c4 = 0x1.65d24ep-2f;
    constexpr float c5 = -0x1.4432a4p-3f;

    float poly =
        t * std::fma(std::fma(std::fma(std::fma(std::fma(c5, t, c4), t, c3), t, c2), t, c1), t, c0);

    float u = ax * ax;
    constexpr float kLog2e = 0x1.715476p+0f;
    constexpr float kLn2hi = 0x1.62e400p-1f;
    constexpr float kLn2lo = 0x1.7f7d1cp-20f;
    float k = std::floor(u * kLog2e);
    float f = std::fma(k, -kLn2hi, u);
    f = std::fma(k, -kLn2lo, f);
    constexpr float e0 = 0x1.fffffep-1f, e1 = -0x1.ffff1ep-1f;
    constexpr float e2 = 0x1.ffe314p-2f, e3 = -0x1.53f876p-3f;
    constexpr float e4 = 0x1.462f16p-5f, e5 = -0x1.80e5b2p-8f;
    float exp_neg_f =
        std::fma(std::fma(std::fma(std::fma(std::fma(e5, f, e4), f, e3), f, e2), f, e1), f, e0);
    int32_t ki = static_cast<int32_t>(k);
    float pow2_neg_k = dfm::bit_cast<float>((127 - ki) << 23);

    result = 1.0f - pow2_neg_k * exp_neg_f * poly;
  } else {
    constexpr float c0 = 0x1.20dd76p+0f;
    constexpr float q0 = -0x1.812746p-2f, q1 = 0x1.ce2ec6p-4f, q2 = -0x1.b81edep-6f;
    constexpr float q3 = 0x1.556b48p-8f, q4 = -0x1.b0255p-11f, q5 = 0x1.7149c8p-14f;
    float u = ax * ax;
    float q =
        std::fma(std::fma(std::fma(std::fma(std::fma(q5, u, q4), u, q3), u, q2), u, q1), u, q0);
    result = ax * std::fma(q, u, c0);
  }
  return x < 0.0f ? -result : result;
}

// --- S21: double, pure polynomial Estrin, 1 ULP ---

static inline float erf_s21(float x) {
  double ax = std::fabs(static_cast<double>(x));
  double result;
  if (ax >= 3.92) {
    result = 1.0;
  } else {
    double u = ax * ax;

    constexpr double c0 = 0x1.20dd74ce6dac1p0;
    constexpr double c1 = -0x1.812728fedb0c3p-2;
    constexpr double c2 = 0x1.ce2c679f0f94dp-4;
    constexpr double c3 = -0x1.b81379b046993p-6;
    constexpr double c4 = 0x1.55decae500c6cp-8;
    constexpr double c5 = -0x1.bd402ca3b1d09p-11;
    constexpr double c6 = 0x1.edfbbbd68d00ep-14;
    constexpr double c7 = -0x1.d43f94bdfb90fp-17;
    constexpr double c8 = 0x1.77643daca82f5p-20;
    constexpr double c9 = -0x1.f276d5cf346ecp-24;
    constexpr double c10 = 0x1.0a42c17eedcadp-27;
    constexpr double c11 = -0x1.b999e591ae6bap-32;
    constexpr double c12 = 0x1.0f73303821975p-36;
    constexpr double c13 = -0x1.ce969a50741b3p-42;
    constexpr double c14 = 0x1.e56af7f1b38e4p-48;
    constexpr double c15 = -0x1.d6f65766c68e5p-55;

    // Estrin's scheme: 4-level tree
    double p0 = std::fma(c1, u, c0);
    double p1 = std::fma(c3, u, c2);
    double p2 = std::fma(c5, u, c4);
    double p3 = std::fma(c7, u, c6);
    double p4 = std::fma(c9, u, c8);
    double p5 = std::fma(c11, u, c10);
    double p6 = std::fma(c13, u, c12);
    double p7 = std::fma(c15, u, c14);
    double u2 = u * u;

    double q0 = std::fma(p1, u2, p0);
    double q1 = std::fma(p3, u2, p2);
    double q2 = std::fma(p5, u2, p4);
    double q3 = std::fma(p7, u2, p6);
    double u4 = u2 * u2;

    double r0 = std::fma(q1, u4, q0);
    double r1 = std::fma(q3, u4, q2);
    double u8 = u4 * u4;

    double R = std::fma(r1, u8, r0);

    result = ax * R;
  }
  return static_cast<float>(x < 0.0f ? -result : result);
}

// --- SSE S16 (float4, t-substitution + inline exp) ---

static inline __m128 erf_s16_sse(__m128 x) {
  __m128 ax = dfm::fabs(x);
  __m128 sign = _mm_and_ps(x, _mm_set1_ps(-0.0f));

  // Near-zero path: erf(x) = x * (c0 + u * Q(u))
  __m128 u = _mm_mul_ps(ax, ax);
  constexpr float nc0 = 0x1.20dd76p+0f;
  constexpr float nq0 = -0x1.812746p-2f, nq1 = 0x1.ce2ec6p-4f, nq2 = -0x1.b81edep-6f;
  constexpr float nq3 = 0x1.556b48p-8f, nq4 = -0x1.b0255p-11f, nq5 = 0x1.7149c8p-14f;
  __m128 q = _mm_fmadd_ps(
      _mm_fmadd_ps(
          _mm_fmadd_ps(
              _mm_fmadd_ps(
                  _mm_fmadd_ps(_mm_set1_ps(nq5), u, _mm_set1_ps(nq4)), u, _mm_set1_ps(nq3)),
              u,
              _mm_set1_ps(nq2)),
          u,
          _mm_set1_ps(nq1)),
      u,
      _mm_set1_ps(nq0));
  __m128 near_zero = _mm_mul_ps(ax, _mm_fmadd_ps(q, u, _mm_set1_ps(nc0)));

  // Erfc path: erf(x) = 1 - t*P(t)*exp(-x^2)
  constexpr float p = 0.45f;
  __m128 t = _mm_div_ps(_mm_set1_ps(1.0f), _mm_fmadd_ps(_mm_set1_ps(p), ax, _mm_set1_ps(1.0f)));

  constexpr float pc0 = 0x1.04873ep-2f, pc1 = 0x1.f81fc6p-3f, pc2 = 0x1.189f42p-2f;
  constexpr float pc3 = 0x1.15aaa6p-5f, pc4 = 0x1.65d24ep-2f, pc5 = -0x1.4432a4p-3f;
  __m128 ppoly = _mm_mul_ps(
      t,
      _mm_fmadd_ps(
          _mm_fmadd_ps(
              _mm_fmadd_ps(
                  _mm_fmadd_ps(
                      _mm_fmadd_ps(_mm_set1_ps(pc5), t, _mm_set1_ps(pc4)), t, _mm_set1_ps(pc3)),
                  t,
                  _mm_set1_ps(pc2)),
              t,
              _mm_set1_ps(pc1)),
          t,
          _mm_set1_ps(pc0)));

  // Inline exp(-x^2): exp(-u) = 2^(-k) * exp(-f)
  constexpr float kLog2e = 0x1.715476p+0f;
  constexpr float kLn2hi = 0x1.62e400p-1f;
  constexpr float kLn2lo = 0x1.7f7d1cp-20f;
  __m128 kv = _mm_floor_ps(_mm_mul_ps(u, _mm_set1_ps(kLog2e)));
  __m128 f = _mm_fmadd_ps(kv, _mm_set1_ps(-kLn2hi), u);
  f = _mm_fmadd_ps(kv, _mm_set1_ps(-kLn2lo), f);
  constexpr float e0 = 0x1.fffffep-1f, e1 = -0x1.ffff1ep-1f;
  constexpr float e2 = 0x1.ffe314p-2f, e3 = -0x1.53f876p-3f;
  constexpr float e4 = 0x1.462f16p-5f, e5 = -0x1.80e5b2p-8f;
  __m128 exp_neg_f = _mm_fmadd_ps(
      _mm_fmadd_ps(
          _mm_fmadd_ps(
              _mm_fmadd_ps(_mm_fmadd_ps(_mm_set1_ps(e5), f, _mm_set1_ps(e4)), f, _mm_set1_ps(e3)),
              f,
              _mm_set1_ps(e2)),
          f,
          _mm_set1_ps(e1)),
      f,
      _mm_set1_ps(e0));
  __m128i ki = _mm_cvtps_epi32(kv);
  __m128 pow2_neg_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_sub_epi32(_mm_set1_epi32(127), ki), 23));

  __m128 erfc_result =
      _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(_mm_mul_ps(pow2_neg_k, exp_neg_f), ppoly));

  // Blend: use near_zero for |x| < 0.875, erfc for |x| >= 0.875
  __m128 use_erfc = _mm_cmpge_ps(ax, _mm_set1_ps(0.875f));
  __m128 result = _mm_blendv_ps(near_zero, erfc_result, use_erfc);

  // Clamp to 1 for |x| >= 3.92
  __m128 saturated = _mm_cmpge_ps(ax, _mm_set1_ps(3.92f));
  result = _mm_blendv_ps(result, _mm_set1_ps(1.0f), saturated);

  // Restore sign
  return _mm_or_ps(result, sign);
}

// --- SSE S21 (double poly Estrin, process 4 floats via 2x double2) ---

static inline __m128 erf_s21_sse(__m128 x) {
  __m128 ax_f = dfm::fabs(x);
  __m128 sign = _mm_and_ps(x, _mm_set1_ps(-0.0f));

  // Convert to double: split into low 2 and high 2
  __m128d ax_lo = _mm_cvtps_pd(ax_f);
  __m128d ax_hi = _mm_cvtps_pd(_mm_movehl_ps(ax_f, ax_f));

  auto eval_poly = [](__m128d ax) -> __m128d {
    __m128d u = _mm_mul_pd(ax, ax);

    constexpr double c0 = 0x1.20dd74ce6dac1p0;
    constexpr double c1 = -0x1.812728fedb0c3p-2;
    constexpr double c2 = 0x1.ce2c679f0f94dp-4;
    constexpr double c3 = -0x1.b81379b046993p-6;
    constexpr double c4 = 0x1.55decae500c6cp-8;
    constexpr double c5 = -0x1.bd402ca3b1d09p-11;
    constexpr double c6 = 0x1.edfbbbd68d00ep-14;
    constexpr double c7 = -0x1.d43f94bdfb90fp-17;
    constexpr double c8 = 0x1.77643daca82f5p-20;
    constexpr double c9 = -0x1.f276d5cf346ecp-24;
    constexpr double c10 = 0x1.0a42c17eedcadp-27;
    constexpr double c11 = -0x1.b999e591ae6bap-32;
    constexpr double c12 = 0x1.0f73303821975p-36;
    constexpr double c13 = -0x1.ce969a50741b3p-42;
    constexpr double c14 = 0x1.e56af7f1b38e4p-48;
    constexpr double c15 = -0x1.d6f65766c68e5p-55;

    // Estrin level 0
    __m128d p0 = _mm_fmadd_pd(_mm_set1_pd(c1), u, _mm_set1_pd(c0));
    __m128d p1 = _mm_fmadd_pd(_mm_set1_pd(c3), u, _mm_set1_pd(c2));
    __m128d p2 = _mm_fmadd_pd(_mm_set1_pd(c5), u, _mm_set1_pd(c4));
    __m128d p3 = _mm_fmadd_pd(_mm_set1_pd(c7), u, _mm_set1_pd(c6));
    __m128d p4 = _mm_fmadd_pd(_mm_set1_pd(c9), u, _mm_set1_pd(c8));
    __m128d p5 = _mm_fmadd_pd(_mm_set1_pd(c11), u, _mm_set1_pd(c10));
    __m128d p6 = _mm_fmadd_pd(_mm_set1_pd(c13), u, _mm_set1_pd(c12));
    __m128d p7 = _mm_fmadd_pd(_mm_set1_pd(c15), u, _mm_set1_pd(c14));
    __m128d u2 = _mm_mul_pd(u, u);

    // Level 1
    __m128d q0 = _mm_fmadd_pd(p1, u2, p0);
    __m128d q1 = _mm_fmadd_pd(p3, u2, p2);
    __m128d q2 = _mm_fmadd_pd(p5, u2, p4);
    __m128d q3 = _mm_fmadd_pd(p7, u2, p6);
    __m128d u4 = _mm_mul_pd(u2, u2);

    // Level 2
    __m128d r0 = _mm_fmadd_pd(q1, u4, q0);
    __m128d r1 = _mm_fmadd_pd(q3, u4, q2);
    __m128d u8 = _mm_mul_pd(u4, u4);

    // Level 3
    __m128d R = _mm_fmadd_pd(r1, u8, r0);

    return _mm_mul_pd(ax, R);
  };

  __m128d res_lo = eval_poly(ax_lo);
  __m128d res_hi = eval_poly(ax_hi);

  // Convert back to float
  __m128 result = _mm_movelh_ps(_mm_cvtpd_ps(res_lo), _mm_cvtpd_ps(res_hi));

  // Clamp to 1 for |x| >= 3.92
  __m128 saturated = _mm_cmpge_ps(ax_f, _mm_set1_ps(3.92f));
  result = _mm_blendv_ps(result, _mm_set1_ps(1.0f), saturated);

  // Restore sign
  return _mm_or_ps(result, sign);
}

static void consumeSum(__m128 sum) {
  alignas(16) float buf[4];
  _mm_store_ps(buf, sum);
  std::cout << buf[0] + buf[1] + buf[2] + buf[3] << std::endl;
}

// --- Scalar benchmarks ---

void BM_erf_libc(benchmark::State& state) {
  const auto& inputs = erfScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::erff(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_erf_s16_scalar(benchmark::State& state) {
  const auto& inputs = erfScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += erf_s16(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_erf_s21_scalar(benchmark::State& state) {
  const auto& inputs = erfScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += erf_s21(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- SSE benchmarks ---

void BM_erf_s16_sse(benchmark::State& state) {
  const auto& inputs = erfSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, erf_s16_sse(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_erf_s21_sse(benchmark::State& state) {
  const auto& inputs = erfSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, erf_s21_sse(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

BENCHMARK(BM_erf_libc);
BENCHMARK(BM_erf_s16_scalar);
BENCHMARK(BM_erf_s21_scalar);
BENCHMARK(BM_erf_s16_sse);
BENCHMARK(BM_erf_s21_sse);

#if defined(__AVX2__)

static void consumeSum256(__m256 sum) {
  alignas(32) float buf[8];
  _mm256_store_ps(buf, sum);
  float s = 0;
  for (int i = 0; i < 8; ++i)
    s += buf[i];
  std::cout << s << std::endl;
}

// --- AVX S16 ---

static inline __m256 erf_s16_avx(__m256 x) {
  __m256 ax = dfm::fabs(x);
  __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));

  // Near-zero path
  __m256 u = _mm256_mul_ps(ax, ax);
  constexpr float nc0 = 0x1.20dd76p+0f;
  constexpr float nq0 = -0x1.812746p-2f, nq1 = 0x1.ce2ec6p-4f, nq2 = -0x1.b81edep-6f;
  constexpr float nq3 = 0x1.556b48p-8f, nq4 = -0x1.b0255p-11f, nq5 = 0x1.7149c8p-14f;
  __m256 q = _mm256_fmadd_ps(
      _mm256_fmadd_ps(
          _mm256_fmadd_ps(
              _mm256_fmadd_ps(
                  _mm256_fmadd_ps(_mm256_set1_ps(nq5), u, _mm256_set1_ps(nq4)),
                  u,
                  _mm256_set1_ps(nq3)),
              u,
              _mm256_set1_ps(nq2)),
          u,
          _mm256_set1_ps(nq1)),
      u,
      _mm256_set1_ps(nq0));
  __m256 near_zero = _mm256_mul_ps(ax, _mm256_fmadd_ps(q, u, _mm256_set1_ps(nc0)));

  // Erfc path
  constexpr float p = 0.45f;
  __m256 t = _mm256_div_ps(
      _mm256_set1_ps(1.0f), _mm256_fmadd_ps(_mm256_set1_ps(p), ax, _mm256_set1_ps(1.0f)));

  constexpr float pc0 = 0x1.04873ep-2f, pc1 = 0x1.f81fc6p-3f, pc2 = 0x1.189f42p-2f;
  constexpr float pc3 = 0x1.15aaa6p-5f, pc4 = 0x1.65d24ep-2f, pc5 = -0x1.4432a4p-3f;
  __m256 ppoly = _mm256_mul_ps(
      t,
      _mm256_fmadd_ps(
          _mm256_fmadd_ps(
              _mm256_fmadd_ps(
                  _mm256_fmadd_ps(
                      _mm256_fmadd_ps(_mm256_set1_ps(pc5), t, _mm256_set1_ps(pc4)),
                      t,
                      _mm256_set1_ps(pc3)),
                  t,
                  _mm256_set1_ps(pc2)),
              t,
              _mm256_set1_ps(pc1)),
          t,
          _mm256_set1_ps(pc0)));

  // Inline exp(-x^2)
  constexpr float kLog2e = 0x1.715476p+0f;
  constexpr float kLn2hi = 0x1.62e400p-1f;
  constexpr float kLn2lo = 0x1.7f7d1cp-20f;
  __m256 kv = _mm256_floor_ps(_mm256_mul_ps(u, _mm256_set1_ps(kLog2e)));
  __m256 f = _mm256_fmadd_ps(kv, _mm256_set1_ps(-kLn2hi), u);
  f = _mm256_fmadd_ps(kv, _mm256_set1_ps(-kLn2lo), f);
  constexpr float e0 = 0x1.fffffep-1f, e1 = -0x1.ffff1ep-1f;
  constexpr float e2 = 0x1.ffe314p-2f, e3 = -0x1.53f876p-3f;
  constexpr float e4 = 0x1.462f16p-5f, e5 = -0x1.80e5b2p-8f;
  __m256 exp_neg_f = _mm256_fmadd_ps(
      _mm256_fmadd_ps(
          _mm256_fmadd_ps(
              _mm256_fmadd_ps(
                  _mm256_fmadd_ps(_mm256_set1_ps(e5), f, _mm256_set1_ps(e4)),
                  f,
                  _mm256_set1_ps(e3)),
              f,
              _mm256_set1_ps(e2)),
          f,
          _mm256_set1_ps(e1)),
      f,
      _mm256_set1_ps(e0));
  __m256i ki = _mm256_cvtps_epi32(kv);
  __m256 pow2_neg_k =
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(127), ki), 23));

  __m256 erfc_result = _mm256_sub_ps(
      _mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_mul_ps(pow2_neg_k, exp_neg_f), ppoly));

  // Blend
  __m256 use_erfc = _mm256_cmp_ps(ax, _mm256_set1_ps(0.875f), _CMP_GE_OQ);
  __m256 result = _mm256_blendv_ps(near_zero, erfc_result, use_erfc);

  __m256 saturated = _mm256_cmp_ps(ax, _mm256_set1_ps(3.92f), _CMP_GE_OQ);
  result = _mm256_blendv_ps(result, _mm256_set1_ps(1.0f), saturated);

  return _mm256_or_ps(result, sign);
}

// --- AVX S21 (process 8 floats via 4x double2) ---

static inline __m256 erf_s21_avx(__m256 x) {
  __m256 ax_f = dfm::fabs(x);
  __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));

  // Split 8 floats into 4 groups of 2 doubles
  __m128 lo4 = _mm256_castps256_ps128(ax_f);
  __m128 hi4 = _mm256_extractf128_ps(ax_f, 1);
  __m128d ax_0 = _mm_cvtps_pd(lo4);
  __m128d ax_1 = _mm_cvtps_pd(_mm_movehl_ps(lo4, lo4));
  __m128d ax_2 = _mm_cvtps_pd(hi4);
  __m128d ax_3 = _mm_cvtps_pd(_mm_movehl_ps(hi4, hi4));

  auto eval_poly = [](__m128d ax) -> __m128d {
    __m128d u = _mm_mul_pd(ax, ax);
    constexpr double c0 = 0x1.20dd74ce6dac1p0;
    constexpr double c1 = -0x1.812728fedb0c3p-2;
    constexpr double c2 = 0x1.ce2c679f0f94dp-4;
    constexpr double c3 = -0x1.b81379b046993p-6;
    constexpr double c4 = 0x1.55decae500c6cp-8;
    constexpr double c5 = -0x1.bd402ca3b1d09p-11;
    constexpr double c6 = 0x1.edfbbbd68d00ep-14;
    constexpr double c7 = -0x1.d43f94bdfb90fp-17;
    constexpr double c8 = 0x1.77643daca82f5p-20;
    constexpr double c9 = -0x1.f276d5cf346ecp-24;
    constexpr double c10 = 0x1.0a42c17eedcadp-27;
    constexpr double c11 = -0x1.b999e591ae6bap-32;
    constexpr double c12 = 0x1.0f73303821975p-36;
    constexpr double c13 = -0x1.ce969a50741b3p-42;
    constexpr double c14 = 0x1.e56af7f1b38e4p-48;
    constexpr double c15 = -0x1.d6f65766c68e5p-55;

    __m128d p0 = _mm_fmadd_pd(_mm_set1_pd(c1), u, _mm_set1_pd(c0));
    __m128d p1 = _mm_fmadd_pd(_mm_set1_pd(c3), u, _mm_set1_pd(c2));
    __m128d p2 = _mm_fmadd_pd(_mm_set1_pd(c5), u, _mm_set1_pd(c4));
    __m128d p3 = _mm_fmadd_pd(_mm_set1_pd(c7), u, _mm_set1_pd(c6));
    __m128d p4 = _mm_fmadd_pd(_mm_set1_pd(c9), u, _mm_set1_pd(c8));
    __m128d p5 = _mm_fmadd_pd(_mm_set1_pd(c11), u, _mm_set1_pd(c10));
    __m128d p6 = _mm_fmadd_pd(_mm_set1_pd(c13), u, _mm_set1_pd(c12));
    __m128d p7 = _mm_fmadd_pd(_mm_set1_pd(c15), u, _mm_set1_pd(c14));
    __m128d u2 = _mm_mul_pd(u, u);

    __m128d q0 = _mm_fmadd_pd(p1, u2, p0);
    __m128d q1 = _mm_fmadd_pd(p3, u2, p2);
    __m128d q2 = _mm_fmadd_pd(p5, u2, p4);
    __m128d q3 = _mm_fmadd_pd(p7, u2, p6);
    __m128d u4 = _mm_mul_pd(u2, u2);

    __m128d r0 = _mm_fmadd_pd(q1, u4, q0);
    __m128d r1 = _mm_fmadd_pd(q3, u4, q2);
    __m128d u8 = _mm_mul_pd(u4, u4);

    __m128d R = _mm_fmadd_pd(r1, u8, r0);
    return _mm_mul_pd(ax, R);
  };

  __m128d res_0 = eval_poly(ax_0);
  __m128d res_1 = eval_poly(ax_1);
  __m128d res_2 = eval_poly(ax_2);
  __m128d res_3 = eval_poly(ax_3);

  // Convert back: 4x double2 -> 2x float4 -> 1x float8
  __m128 lo_f = _mm_movelh_ps(_mm_cvtpd_ps(res_0), _mm_cvtpd_ps(res_1));
  __m128 hi_f = _mm_movelh_ps(_mm_cvtpd_ps(res_2), _mm_cvtpd_ps(res_3));
  __m256 result = _mm256_set_m128(hi_f, lo_f);

  // Clamp
  __m256 saturated = _mm256_cmp_ps(ax_f, _mm256_set1_ps(3.92f), _CMP_GE_OQ);
  result = _mm256_blendv_ps(result, _mm256_set1_ps(1.0f), saturated);

  return _mm256_or_ps(result, sign);
}

void BM_erf_s16_avx(benchmark::State& state) {
  const auto& inputs = erfAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, erf_s16_avx(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum256(sum);
}

void BM_erf_s21_avx(benchmark::State& state) {
  const auto& inputs = erfAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, erf_s21_avx(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum256(sum);
}

BENCHMARK(BM_erf_s16_avx);
BENCHMARK(BM_erf_s21_avx);

#endif // __AVX2__

#else // !__SSE4_1__

int main() {
  std::cout << "SSE4.1 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // __SSE4_1__
