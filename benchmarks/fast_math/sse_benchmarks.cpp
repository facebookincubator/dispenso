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

#if defined(__SSE4_1__)

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
static void consumeSum(__m128 sum) {
  alignas(16) float buf[4];
  _mm_store_ps(buf, sum);
  std::cout << buf[0] + buf[1] + buf[2] + buf[3] << std::endl;
}

// --- sin ---

const std::vector<__m128>& sinSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<__m128> inp;
    float f = static_cast<float>(-M_PI / 2.0);
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
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

void BM_sin_sse(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::sin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_cos_sse(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::cos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- exp ---

const std::vector<__m128>& expSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = 20.0f / kNumInputs;
    std::vector<__m128> inp;
    float f = -10.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
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

void BM_exp_sse(benchmark::State& state) {
  const auto& inputs = expSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::exp(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

const std::vector<__m128>& logSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<__m128> inp;
    float f = 0.001f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
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

void BM_log_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::log(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_cbrt_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::cbrt(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_atan_sse(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::atan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- acos ---

const std::vector<__m128>& acosSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<__m128> inp;
    float f = -0.999f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
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

void BM_acos_sse(benchmark::State& state) {
  const auto& inputs = acosSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::acos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- Accurate and bounds variants ---

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

void BM_sin_sse_accurate(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::sin<__m128, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_cos_sse_accurate(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::cos<__m128, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_exp_sse_accurate(benchmark::State& state) {
  const auto& inputs = expSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::exp<__m128, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_exp_sse_bounds(benchmark::State& state) {
  const auto& inputs = expSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::exp<__m128, BoundsTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_log_sse_accurate(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::log<__m128, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_cbrt_sse_accurate(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::cbrt<__m128, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_tan_sse(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::tan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_asin_sse(benchmark::State& state) {
  const auto& inputs = acosSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::asin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_atan2_sse(benchmark::State& state) {
  const auto& inputs = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = _mm_add_ps(sum, dfm::atan2(inputs[idx], inputs[idx2]));
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_exp2_sse(benchmark::State& state) {
  const auto& inputs = expSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::exp2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_exp10_sse(benchmark::State& state) {
  const auto& inputs = expSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::exp10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_log2_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::log2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_log10_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::log10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_frexp_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    dfm::IntType_t<__m128> e;
    sum = _mm_add_ps(sum, dfm::frexp(inputs[idx], &e));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
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

void BM_ldexp_sse(benchmark::State& state) {
  const auto& inputs = logSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::ldexp(inputs[idx], _mm_set1_epi32(3)));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- hypot ---

const std::vector<__m128>& hypotSseInputs() {
  static std::vector<__m128> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<__m128> inp;
    float f = -100000.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      inp.emplace_back(_mm_set_ps(f + 3 * delta, f + 2 * delta, f + delta, f));
      f += 4 * delta;
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

void BM_hypot_sse(benchmark::State& state) {
  const auto& inputs = hypotSseInputs();
  const auto& inputs2 = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::hypot(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

BENCHMARK(BM_hypot_scalar);
BENCHMARK(BM_hypot_sse);

void BM_hypot_sse_bounds(benchmark::State& state) {
  const auto& inputs = hypotSseInputs();
  const auto& inputs2 = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    sum = _mm_add_ps(sum, dfm::hypot<__m128, dfm::MaxAccuracyTraits>(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

BENCHMARK(BM_hypot_sse_bounds);

void BM_hypot_libc(benchmark::State& state) {
  const auto& inputs = hypotSseInputs();
  const auto& inputs2 = sinSseInputs();
  size_t idx = 0;
  __m128 sum = _mm_setzero_ps();
  for (auto UNUSED_VAR : state) {
    // 4x scalar hypotf to match SSE lane count.
    alignas(16) float x[4], y[4], r[4];
    _mm_store_ps(x, inputs[idx]);
    _mm_store_ps(y, inputs2[idx]);
    r[0] = ::hypotf(x[0], y[0]);
    r[1] = ::hypotf(x[1], y[1]);
    r[2] = ::hypotf(x[2], y[2]);
    r[3] = ::hypotf(x[3], y[3]);
    sum = _mm_add_ps(sum, _mm_load_ps(r));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

BENCHMARK(BM_hypot_libc);

// Registrations.
BENCHMARK(BM_sin_libc);
BENCHMARK(BM_sin_scalar);
BENCHMARK(BM_sin_sse);
BENCHMARK(BM_sin_sse_accurate);

BENCHMARK(BM_cos_scalar);
BENCHMARK(BM_cos_sse);
BENCHMARK(BM_cos_sse_accurate);

BENCHMARK(BM_tan_scalar);
BENCHMARK(BM_tan_sse);

BENCHMARK(BM_exp_libc);
BENCHMARK(BM_exp_scalar);
BENCHMARK(BM_exp_sse);
BENCHMARK(BM_exp_sse_bounds);
BENCHMARK(BM_exp_sse_accurate);

BENCHMARK(BM_exp2_scalar);
BENCHMARK(BM_exp2_sse);

BENCHMARK(BM_exp10_scalar);
BENCHMARK(BM_exp10_sse);

BENCHMARK(BM_log_libc);
BENCHMARK(BM_log_scalar);
BENCHMARK(BM_log_sse);
BENCHMARK(BM_log_sse_accurate);

BENCHMARK(BM_log2_scalar);
BENCHMARK(BM_log2_sse);

BENCHMARK(BM_log10_scalar);
BENCHMARK(BM_log10_sse);

BENCHMARK(BM_cbrt_scalar);
BENCHMARK(BM_cbrt_sse);
BENCHMARK(BM_cbrt_sse_accurate);

BENCHMARK(BM_atan_scalar);
BENCHMARK(BM_atan_sse);

BENCHMARK(BM_acos_scalar);
BENCHMARK(BM_acos_sse);

BENCHMARK(BM_asin_scalar);
BENCHMARK(BM_asin_sse);

BENCHMARK(BM_atan2_scalar);
BENCHMARK(BM_atan2_sse);

BENCHMARK(BM_frexp_scalar);
BENCHMARK(BM_frexp_sse);

BENCHMARK(BM_ldexp_scalar);
BENCHMARK(BM_ldexp_sse);

#else // !defined(__SSE4_1__)

// If SSE4.1 is not available, provide a minimal main.
int main() {
  std::cout << "SSE4.1 not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__SSE4_1__)
