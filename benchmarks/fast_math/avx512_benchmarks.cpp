/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include <benchmark/benchmark.h>
#include <dispenso/fast_math/fast_math.h>

#if defined(__AVX512F__)

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

// Helper to sum lanes and prevent optimization.
static void consumeSum(__m512 sum) {
  alignas(64) float buf[16];
  _mm512_store_ps(buf, sum);
  float total = 0.0f;
  for (int i = 0; i < 16; ++i)
    total += buf[i];
  std::cout << total << std::endl;
}

// --- sin ---

const std::vector<__m512>& sinAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<__m512> inp;
    float f = static_cast<float>(-M_PI / 2.0);
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& sinScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<float> inp;
    for (float f = static_cast<float>(-M_PI / 2.0); inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

void BM_sin_scalar(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::sin(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_sin_avx512(benchmark::State& state) {
  const auto& inputs = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::sin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_sin_libc(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::sinf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- cos ---

void BM_cos_scalar(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::cos(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_cos_avx512(benchmark::State& state) {
  const auto& inputs = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::cos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- exp ---

const std::vector<__m512>& expAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 20.0f / kNumInputs;
    std::vector<__m512> inp;
    float f = -10.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& expScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = 20.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -10.0f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

void BM_exp_scalar(benchmark::State& state) {
  const auto& inputs = expScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::exp(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp_avx512(benchmark::State& state) {
  const auto& inputs = expAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::exp(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_exp_libc(benchmark::State& state) {
  const auto& inputs = expScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::expf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- log ---

const std::vector<__m512>& logAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<__m512> inp;
    float f = 0.001f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& logScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = 0.001f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

void BM_log_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::log(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_log_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::log(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_log_libc(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::logf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- cbrt ---

void BM_cbrt_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::cbrt(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_cbrt_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::cbrt(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- atan ---

void BM_atan_scalar(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::atan(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_atan_avx512(benchmark::State& state) {
  const auto& inputs = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::atan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- acos ---

const std::vector<__m512>& acosAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<__m512> inp;
    float f = -0.999f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& acosScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -0.999f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

void BM_acos_scalar(benchmark::State& state) {
  const auto& inputs = acosScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::acos(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_acos_avx512(benchmark::State& state) {
  const auto& inputs = acosAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::acos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- tan ---

void BM_tan_scalar(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::tan(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_tan_avx512(benchmark::State& state) {
  const auto& inputs = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::tan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- asin ---

void BM_asin_scalar(benchmark::State& state) {
  const auto& inputs = acosScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::asin(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_asin_avx512(benchmark::State& state) {
  const auto& inputs = acosAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::asin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- atan2 ---

void BM_atan2_scalar(benchmark::State& state) {
  const auto& inputs = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum += dfm::atan2(inputs[idx], inputs[idx2]);
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_atan2_avx512(benchmark::State& state) {
  const auto& inputs = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = _mm512_add_ps(sum, dfm::atan2(inputs[idx], inputs[idx2]));
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- exp2 ---

void BM_exp2_scalar(benchmark::State& state) {
  const auto& inputs = expScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::exp2(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp2_avx512(benchmark::State& state) {
  const auto& inputs = expAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::exp2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- exp10 ---

void BM_exp10_scalar(benchmark::State& state) {
  const auto& inputs = expScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::exp10(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp10_avx512(benchmark::State& state) {
  const auto& inputs = expAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::exp10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- log2 ---

void BM_log2_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::log2(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_log2_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::log2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- log10 ---

void BM_log10_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::log10(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_log10_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::log10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- frexp ---

void BM_frexp_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    int32_t e;
    sum += dfm::frexp(inputs[idx], &e);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_frexp_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    dfm::IntType_t<__m512> e;
    sum = _mm512_add_ps(sum, dfm::frexp(inputs[idx], &e));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// --- ldexp ---

void BM_ldexp_scalar(benchmark::State& state) {
  const auto& inputs = logScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::ldexp(inputs[idx], int32_t(3));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_ldexp_avx512(benchmark::State& state) {
  const auto& inputs = logAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::ldexp(inputs[idx], _mm512_set1_epi32(3)));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

// Registrations.
BENCHMARK(BM_sin_libc);
BENCHMARK(BM_sin_scalar);
BENCHMARK(BM_sin_avx512);

BENCHMARK(BM_cos_scalar);
BENCHMARK(BM_cos_avx512);

BENCHMARK(BM_tan_scalar);
BENCHMARK(BM_tan_avx512);

BENCHMARK(BM_exp_libc);
BENCHMARK(BM_exp_scalar);
BENCHMARK(BM_exp_avx512);

BENCHMARK(BM_exp2_scalar);
BENCHMARK(BM_exp2_avx512);

BENCHMARK(BM_exp10_scalar);
BENCHMARK(BM_exp10_avx512);

BENCHMARK(BM_log_libc);
BENCHMARK(BM_log_scalar);
BENCHMARK(BM_log_avx512);

BENCHMARK(BM_log2_scalar);
BENCHMARK(BM_log2_avx512);

BENCHMARK(BM_log10_scalar);
BENCHMARK(BM_log10_avx512);

BENCHMARK(BM_cbrt_scalar);
BENCHMARK(BM_cbrt_avx512);

BENCHMARK(BM_atan_scalar);
BENCHMARK(BM_atan_avx512);

BENCHMARK(BM_acos_scalar);
BENCHMARK(BM_acos_avx512);

BENCHMARK(BM_asin_scalar);
BENCHMARK(BM_asin_avx512);

BENCHMARK(BM_atan2_scalar);
BENCHMARK(BM_atan2_avx512);

BENCHMARK(BM_frexp_scalar);
BENCHMARK(BM_frexp_avx512);

BENCHMARK(BM_ldexp_scalar);
BENCHMARK(BM_ldexp_avx512);

// --- hypot ---

const std::vector<__m512>& hypotAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<__m512> inp;
    float f = -100000.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

void BM_hypot_avx512(benchmark::State& state) {
  const auto& inputs = hypotAvx512Inputs();
  const auto& inputs2 = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::hypot(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_hypot_avx512_bounds(benchmark::State& state) {
  const auto& inputs = hypotAvx512Inputs();
  const auto& inputs2 = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::hypot<__m512, dfm::MaxAccuracyTraits>(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_hypot_libc_avx512(benchmark::State& state) {
  const auto& inputs = hypotAvx512Inputs();
  const auto& inputs2 = sinAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    alignas(64) float x[16], y[16], r[16];
    _mm512_store_ps(x, inputs[idx]);
    _mm512_store_ps(y, inputs2[idx]);
    for (int i = 0; i < 16; ++i) {
      r[i] = ::hypotf(x[i], y[i]);
    }
    sum = _mm512_add_ps(sum, _mm512_load_ps(r));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

BENCHMARK(BM_hypot_avx512);
BENCHMARK(BM_hypot_avx512_bounds);
BENCHMARK(BM_hypot_libc_avx512);

// --- pow ---

const std::vector<__m512>& powBaseAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 99.99f / kNumInputs;
    std::vector<__m512> inp;
    float f = 0.01f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

const std::vector<__m512>& powExpAvx512Inputs() {
  static std::vector<__m512> inputs = []() {
    float delta = 16.0f / kNumInputs;
    std::vector<__m512> inp;
    float f = -8.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm512_set_ps(
          f + 15 * delta,
          f + 14 * delta,
          f + 13 * delta,
          f + 12 * delta,
          f + 11 * delta,
          f + 10 * delta,
          f + 9 * delta,
          f + 8 * delta,
          f + 7 * delta,
          f + 6 * delta,
          f + 5 * delta,
          f + 4 * delta,
          f + 3 * delta,
          f + 2 * delta,
          f + delta,
          f));
      f += 16 * delta;
    }
    return inp;
  }();
  return inputs;
}

void BM_pow_avx512(benchmark::State& state) {
  const auto& bases = powBaseAvx512Inputs();
  const auto& exps = powExpAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::pow(bases[idx], exps[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_pow_avx512_accurate(benchmark::State& state) {
  const auto& bases = powBaseAvx512Inputs();
  const auto& exps = powExpAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::pow<__m512, dfm::MaxAccuracyTraits>(bases[idx], exps[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_pow_avx512_scalar_exp(benchmark::State& state) {
  const auto& bases = powBaseAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm512_add_ps(sum, dfm::pow(bases[idx], 2.5f));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

void BM_pow_libc_avx512(benchmark::State& state) {
  const auto& bases = powBaseAvx512Inputs();
  const auto& exps = powExpAvx512Inputs();
  size_t idx = 0;
  __m512 sum = _mm512_setzero_ps();
  for (auto UNUSED_VAR : state) {
    alignas(64) float x[16], y[16], r[16];
    _mm512_store_ps(x, bases[idx]);
    _mm512_store_ps(y, exps[idx]);
    for (int32_t j = 0; j < 16; ++j)
      r[j] = ::powf(x[j], y[j]);
    sum = _mm512_add_ps(sum, _mm512_load_ps(r));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 16);
  consumeSum(sum);
}

BENCHMARK(BM_pow_libc_avx512);
BENCHMARK(BM_pow_avx512);
BENCHMARK(BM_pow_avx512_accurate);
BENCHMARK(BM_pow_avx512_scalar_exp);

#else // !defined(__AVX512F__)

int main() {
  std::cout << "AVX-512 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__AVX512F__)
