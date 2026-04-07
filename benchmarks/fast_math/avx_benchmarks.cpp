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

#if defined(__AVX2__)

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
static void consumeSum(__m256 sum) {
  alignas(32) float buf[8];
  _mm256_store_ps(buf, sum);
  float total = 0.0f;
  for (int i = 0; i < 8; ++i)
    total += buf[i];
  std::cout << total << std::endl;
}

// --- sin ---

const std::vector<__m256>& sinAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<__m256> inp;
    float f = static_cast<float>(-M_PI / 2.0);
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

void BM_sin_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::sin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_cos_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::cos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

// --- exp ---

const std::vector<__m256>& expAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 20.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -10.0f;
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

void BM_exp_avx(benchmark::State& state) {
  const auto& inputs = expAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::exp(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

const std::vector<__m256>& logAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = 0.001f;
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

void BM_log_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::log(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_cbrt_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::cbrt(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_atan_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::atan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

// --- acos ---

const std::vector<__m256>& acosAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -0.999f;
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

void BM_acos_avx(benchmark::State& state) {
  const auto& inputs = acosAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::acos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

// --- Accurate and bounds variants ---

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

void BM_sin_avx_accurate(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::sin<__m256, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_cos_avx_accurate(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::cos<__m256, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_exp_avx_accurate(benchmark::State& state) {
  const auto& inputs = expAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::exp<__m256, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_exp_avx_bounds(benchmark::State& state) {
  const auto& inputs = expAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::exp<__m256, BoundsTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_log_avx_accurate(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::log<__m256, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_cbrt_avx_accurate(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::cbrt<__m256, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_tan_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::tan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_asin_avx(benchmark::State& state) {
  const auto& inputs = acosAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::asin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_atan2_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = _mm256_add_ps(sum, dfm::atan2(inputs[idx], inputs[idx2]));
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_exp2_avx(benchmark::State& state) {
  const auto& inputs = expAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::exp2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_exp10_avx(benchmark::State& state) {
  const auto& inputs = expAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::exp10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_log2_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::log2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_log10_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::log10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_frexp_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    dfm::IntType_t<__m256> e;
    sum = _mm256_add_ps(sum, dfm::frexp(inputs[idx], &e));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
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

void BM_ldexp_avx(benchmark::State& state) {
  const auto& inputs = logAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::ldexp(inputs[idx], _mm256_set1_epi32(3)));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

// --- hypot ---

const std::vector<__m256>& hypotAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -100000.0f;
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

const std::vector<float>& hypotScalarInputs() {
  static std::vector<float> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -100000.0f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

void BM_hypot_scalar(benchmark::State& state) {
  const auto& inputs = hypotScalarInputs();
  const auto& inputs2 = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dfm::hypot(inputs[idx], inputs2[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_hypot_avx(benchmark::State& state) {
  const auto& inputs = hypotAvxInputs();
  const auto& inputs2 = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::hypot(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_hypot_avx_bounds(benchmark::State& state) {
  const auto& inputs = hypotAvxInputs();
  const auto& inputs2 = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::hypot<__m256, dfm::MaxAccuracyTraits>(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

// Registrations.
BENCHMARK(BM_sin_libc);
BENCHMARK(BM_sin_scalar);
BENCHMARK(BM_sin_avx);
BENCHMARK(BM_sin_avx_accurate);

BENCHMARK(BM_cos_scalar);
BENCHMARK(BM_cos_avx);
BENCHMARK(BM_cos_avx_accurate);

BENCHMARK(BM_tan_scalar);
BENCHMARK(BM_tan_avx);

BENCHMARK(BM_exp_libc);
BENCHMARK(BM_exp_scalar);
BENCHMARK(BM_exp_avx);
BENCHMARK(BM_exp_avx_bounds);
BENCHMARK(BM_exp_avx_accurate);

BENCHMARK(BM_exp2_scalar);
BENCHMARK(BM_exp2_avx);

BENCHMARK(BM_exp10_scalar);
BENCHMARK(BM_exp10_avx);

BENCHMARK(BM_log_libc);
BENCHMARK(BM_log_scalar);
BENCHMARK(BM_log_avx);
BENCHMARK(BM_log_avx_accurate);

BENCHMARK(BM_log2_scalar);
BENCHMARK(BM_log2_avx);

BENCHMARK(BM_log10_scalar);
BENCHMARK(BM_log10_avx);

BENCHMARK(BM_cbrt_scalar);
BENCHMARK(BM_cbrt_avx);
BENCHMARK(BM_cbrt_avx_accurate);

BENCHMARK(BM_atan_scalar);
BENCHMARK(BM_atan_avx);

BENCHMARK(BM_acos_scalar);
BENCHMARK(BM_acos_avx);

BENCHMARK(BM_asin_scalar);
BENCHMARK(BM_asin_avx);

BENCHMARK(BM_atan2_scalar);
BENCHMARK(BM_atan2_avx);

BENCHMARK(BM_frexp_scalar);
BENCHMARK(BM_frexp_avx);

BENCHMARK(BM_ldexp_scalar);
BENCHMARK(BM_ldexp_avx);

BENCHMARK(BM_hypot_scalar);
BENCHMARK(BM_hypot_avx);
BENCHMARK(BM_hypot_avx_bounds);

// --- pow ---

const std::vector<__m256>& powBaseAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 99.99f / kNumInputs;
    std::vector<__m256> inp;
    float f = 0.01f;
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

const std::vector<__m256>& powExpAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 16.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -8.0f;
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

void BM_pow_libc_avx(benchmark::State& state) {
  const auto& bases = powBaseAvxInputs();
  const auto& exps = powExpAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    alignas(32) float x[8], y[8], r[8];
    _mm256_store_ps(x, bases[idx]);
    _mm256_store_ps(y, exps[idx]);
    for (int i = 0; i < 8; ++i) {
      r[i] = ::powf(x[i], y[i]);
    }
    sum = _mm256_add_ps(sum, _mm256_load_ps(r));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_pow_avx(benchmark::State& state) {
  const auto& bases = powBaseAvxInputs();
  const auto& exps = powExpAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::pow(bases[idx], exps[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_pow_avx_accurate(benchmark::State& state) {
  const auto& bases = powBaseAvxInputs();
  const auto& exps = powExpAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::pow<__m256, dfm::MaxAccuracyTraits>(bases[idx], exps[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

void BM_pow_avx_scalar_exp(benchmark::State& state) {
  const auto& bases = powBaseAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::pow(bases[idx], 2.5f));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

BENCHMARK(BM_pow_libc_avx);
BENCHMARK(BM_pow_avx);
BENCHMARK(BM_pow_avx_accurate);
BENCHMARK(BM_pow_avx_scalar_exp);

// --- expm1 ---

void BM_expm1_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::expm1(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

BENCHMARK(BM_expm1_avx);

// --- log1p ---

void BM_log1p_avx(benchmark::State& state) {
  const auto& inputs = sinAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    // Use fabs to keep inputs positive (log1p needs x > -1).
    __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), inputs[idx]);
    sum = _mm256_add_ps(sum, dfm::log1p(ax));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

BENCHMARK(BM_log1p_avx);

// --- tanh ---

const std::vector<__m256>& tanhAvxInputs() {
  static std::vector<__m256> inputs = []() {
    float delta = 10.0f / kNumInputs;
    std::vector<__m256> inp;
    float f = -5.0f;
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

void BM_tanh_avx(benchmark::State& state) {
  const auto& inputs = tanhAvxInputs();
  size_t idx = 0;
  __m256 sum = _mm256_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm256_add_ps(sum, dfm::tanh(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 8);
  consumeSum(sum);
}

BENCHMARK(BM_tanh_avx);

#else // !defined(__AVX2__)

int main() {
  std::cout << "AVX2 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__AVX2__)
