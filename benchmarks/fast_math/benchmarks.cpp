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

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED_VAR myLocalForLoopVar __attribute__((unused))
#elif defined(_MSC_VER)
#define UNUSED_VAR myLocalForLoopVar __pragma(warning(suppress : 4100))
#else
#define UNUSED_VAR myLocalForLoopVar
#endif

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;

const std::vector<float>& acosInputs() {
  static std::vector<float> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -1.0f; f <= 1.0f; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();

  while (inputs.size() < kNumInputs) {
    inputs.push_back(inputs.back());
  }

  return inputs;
}

void BM_acos(benchmark::State& state) {
  const auto& inputs = acosInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::acosf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_acos(benchmark::State& state) {
  const auto& inputs = acosInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::acos(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_asin(benchmark::State& state) {
  const auto& inputs = acosInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::asinf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_asin(benchmark::State& state) {
  const auto& inputs = acosInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::asin(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

const std::vector<float>& cbrtInputs() {
  static std::vector<float> inputs = []() {
    float delta = 50000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -50000.f; f <= 50000.f; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();

  while (inputs.size() < kNumInputs) {
    inputs.push_back(inputs.back());
  }

  return inputs;
}

void BM_atan(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::atanf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_atan(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::atan(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

const std::vector<float>& sinInputs() {
  static std::vector<float> inputs = []() {
    float delta = M_PI / kNumInputs;
    std::vector<float> inp;
    for (float f = -M_PI / 2.0; f <= M_PI / 2.0; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();

  while (inputs.size() < kNumInputs) {
    inputs.push_back(inputs.back());
  }

  return inputs;
}

void BM_cbrt(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::cbrtf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_cbrt(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::cbrt(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_cbrt_accurate(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::cbrt<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_sin(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::sinf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_sin(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::sin(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_sin_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::sin<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_cos(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::cosf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_cos(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::cos(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_cos_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::cos<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_frexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  int exp;
  int64_t expSum = 0;
  for (auto UNUSED_VAR : state) {
    sum += ::frexpf(inputs[idx], &exp);
    expSum += exp;
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << " " << expSum << std::endl;
}

void BM_fastm_frexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  int exp;
  int64_t expSum = 0;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::frexp(inputs[idx], &exp);
    expSum += exp;
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << " " << expSum << std::endl;
}

void BM_ldexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::ldexpf(inputs[idx], idx & 7);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_ldexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::ldexp(inputs[idx], idx & 7);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_tan(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::tanf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_tan(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::tan(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_tan_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::tan<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp2(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::exp2f(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp2(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp2(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp2_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp2<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::expf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};
void BM_fastm_exp_bounds(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;

  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp<float, BoundsTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_exp10(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::powf(10.0f, inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp10(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp10(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_exp10_accurate(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::exp10<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

const std::vector<float>& logInputs() {
  static std::vector<float> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = 0.0f; f <= 10000.0f; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();

  while (inputs.size() < kNumInputs) {
    inputs.push_back(inputs.back());
  }

  return inputs;
}

void BM_log2(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::log2f(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log2(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log2(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log2_accurate(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log2<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_log(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::logf(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log_accurate(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_log10(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::log10f(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log10(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log10(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_log10_accurate(benchmark::State& state) {
  const auto& inputs = logInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::log10<float, dispenso::fast_math::MaxAccuracyTraits>(inputs[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_atan2(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  const auto& inputs2 = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::atan2f(inputs[idx], inputs2[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_atan2(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  const auto& inputs2 = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::atan2(inputs[idx], inputs2[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

void BM_fastm_atan2_bounds(benchmark::State& state) {
  const auto& inputs = cbrtInputs();
  const auto& inputs2 = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += dispenso::fast_math::atan2<float, BoundsTraits>(inputs[idx], inputs2[idx]);
    idx = (idx + 1) & kInputsMask;
  }

  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- Batch benchmarks: explicit SIMD via SseFloat (4-wide SSE) ---
// Compare pi/4 blend (sin) vs pi/2 single-poly (sin_wide) throughput.

#if defined(__SSE4_1__)
#include <dispenso/fast_math/float_traits_x86.h>

static void BM_batch_sinf(benchmark::State& state) {
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto UNUSED_VAR : state) {
    for (size_t i = 0; i < kNumInputs; ++i) {
      outputs[i] = ::sinf(inputs[i]);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

static void BM_batch_sin_scalar(benchmark::State& state) {
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto UNUSED_VAR : state) {
    for (size_t i = 0; i < kNumInputs; ++i) {
      outputs[i] = dispenso::fast_math::sin(inputs[i]);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

static void BM_batch_sin_pi4_sse(benchmark::State& state) {
  using namespace dispenso::fast_math;
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto UNUSED_VAR : state) {
    for (size_t i = 0; i < kNumInputs; i += 4) {
      SseFloat x = _mm_loadu_ps(&inputs[i]);
      SseFloat r = sin<SseFloat>(x);
      _mm_storeu_ps(&outputs[i], r.v);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

BENCHMARK(BM_batch_sinf);
BENCHMARK(BM_batch_sin_scalar);
BENCHMARK(BM_batch_sin_pi4_sse);
#endif // __SSE4_1__

BENCHMARK(BM_acos);
BENCHMARK(BM_fastm_acos);

BENCHMARK(BM_asin);
BENCHMARK(BM_fastm_asin);

BENCHMARK(BM_atan);
BENCHMARK(BM_fastm_atan);

BENCHMARK(BM_atan2);
BENCHMARK(BM_fastm_atan2);
BENCHMARK(BM_fastm_atan2_bounds);

BENCHMARK(BM_cbrt);
BENCHMARK(BM_fastm_cbrt);
BENCHMARK(BM_fastm_cbrt_accurate);

BENCHMARK(BM_exp);
BENCHMARK(BM_fastm_exp);
BENCHMARK(BM_fastm_exp_bounds);
BENCHMARK(BM_fastm_exp_accurate);

BENCHMARK(BM_exp10);
BENCHMARK(BM_fastm_exp10);
BENCHMARK(BM_fastm_exp10_accurate);

BENCHMARK(BM_exp2);
BENCHMARK(BM_fastm_exp2);
BENCHMARK(BM_fastm_exp2_accurate);

BENCHMARK(BM_log);
BENCHMARK(BM_fastm_log);
BENCHMARK(BM_fastm_log_accurate);

BENCHMARK(BM_log2);
BENCHMARK(BM_fastm_log2);
BENCHMARK(BM_fastm_log2_accurate);

BENCHMARK(BM_log10);
BENCHMARK(BM_fastm_log10);
BENCHMARK(BM_fastm_log10_accurate);

BENCHMARK(BM_sin);
BENCHMARK(BM_fastm_sin);
BENCHMARK(BM_fastm_sin_accurate);

BENCHMARK(BM_cos);
BENCHMARK(BM_fastm_cos);
BENCHMARK(BM_fastm_cos_accurate);

BENCHMARK(BM_frexp);
BENCHMARK(BM_fastm_frexp);

BENCHMARK(BM_ldexp);
BENCHMARK(BM_fastm_ldexp);

BENCHMARK(BM_tan);
BENCHMARK(BM_fastm_tan);
BENCHMARK(BM_fastm_tan_accurate);
