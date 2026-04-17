/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "benchmark_helpers.h"

namespace dfm = dispenso::fast_math;
namespace bench = dispenso::fast_math::bench;

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// --- Input generators (scalar-specific ranges, different from SIMD) ---

const std::vector<float>& acosInputs() {
  static std::vector<float> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -1.0f; f <= 1.0f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& cbrtInputs() {
  static std::vector<float> inputs = []() {
    float delta = 50000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -50000.f; f <= 50000.f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& sinInputs() {
  static std::vector<float> inputs = []() {
    float delta = M_PI / kNumInputs;
    std::vector<float> inp;
    for (float f = -M_PI / 2.0; f <= M_PI / 2.0; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& logInputs() {
  static std::vector<float> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = 0.0f; f <= 10000.0f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& hypotInputs() {
  static std::vector<float> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -100000.f; f <= 100000.f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& powBaseInputs() {
  static std::vector<float> inputs = []() {
    float delta = 99.99f / kNumInputs;
    std::vector<float> inp;
    for (float f = 0.01f; f <= 100.0f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& powExpInputs() {
  static std::vector<float> inputs = []() {
    float delta = 16.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -8.0f; f <= 8.0f; f += delta) {
      inp.push_back(f);
    }
    while (inp.size() < kNumInputs) {
      inp.push_back(inp.back());
    }
    return inp;
  }();
  return inputs;
}

const std::vector<float>& tanhInputs() {
  static std::vector<float> inputs = []() {
    float delta = 10.0f / kNumInputs;
    std::vector<float> inp;
    for (float f = -5.0f; inp.size() < kNumInputs; f += delta) {
      inp.push_back(f);
    }
    return inp;
  }();
  return inputs;
}

// --- Libc benchmarks ---

void BM_acos(benchmark::State& state) {
  bench::runBench(state, acosInputs(), [](auto x) { return ::acosf(x); });
}
void BM_asin(benchmark::State& state) {
  bench::runBench(state, acosInputs(), [](auto x) { return ::asinf(x); });
}
void BM_atan(benchmark::State& state) {
  bench::runBench(state, cbrtInputs(), [](auto x) { return ::atanf(x); });
}
void BM_cbrt(benchmark::State& state) {
  bench::runBench(state, cbrtInputs(), [](auto x) { return ::cbrtf(x); });
}
void BM_sin(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::sinf(x); });
}
void BM_cos(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::cosf(x); });
}
void BM_tan(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::tanf(x); });
}
void BM_exp(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::expf(x); });
}
void BM_exp2(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::exp2f(x); });
}
void BM_exp10(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::powf(10.0f, x); });
}
void BM_log(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return ::logf(x); });
}
void BM_log2(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return ::log2f(x); });
}
void BM_log10(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return ::log10f(x); });
}
void BM_expm1(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::expm1f(x); });
}
void BM_log1p(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::log1pf(std::fabs(x)); });
}
void BM_tanh(benchmark::State& state) {
  bench::runBench(state, tanhInputs(), [](auto x) { return ::tanhf(x); });
}
void BM_sin_plus_cos(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return ::sinf(x) + ::cosf(x); });
}

void BM_atan2(benchmark::State& state) {
  bench::runBench2(state, cbrtInputs(), sinInputs(), [](auto y, auto x) { return ::atan2f(y, x); });
}
void BM_hypot(benchmark::State& state) {
  bench::runBench2(
      state, hypotInputs(), sinInputs(), [](auto x, auto y) { return ::hypotf(x, y); });
}
void BM_pow(benchmark::State& state) {
  bench::runBench2(
      state, powBaseInputs(), powExpInputs(), [](auto b, auto e) { return ::powf(b, e); });
}

// frexp and ldexp use non-standard loop patterns — kept hand-written.
void BM_frexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  int exp;
  int64_t expSum = 0;
  for (auto _ : state) {
    (void)_;
    sum += ::frexpf(inputs[idx], &exp);
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
  for (auto _ : state) {
    (void)_;
    sum += ::ldexpf(inputs[idx], idx & 7);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- Dispenso fast_math benchmarks ---

void BM_fastm_acos(benchmark::State& state) {
  bench::runBench(state, acosInputs(), [](auto x) { return dfm::acos(x); });
}
void BM_fastm_asin(benchmark::State& state) {
  bench::runBench(state, acosInputs(), [](auto x) { return dfm::asin(x); });
}
void BM_fastm_atan(benchmark::State& state) {
  bench::runBench(state, cbrtInputs(), [](auto x) { return dfm::atan(x); });
}
void BM_fastm_cbrt(benchmark::State& state) {
  bench::runBench(state, cbrtInputs(), [](auto x) { return dfm::cbrt(x); });
}
void BM_fastm_cbrt_accurate(benchmark::State& state) {
  bench::runBench(
      state, cbrtInputs(), [](auto x) { return dfm::cbrt<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_sin(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::sin(x); });
}
void BM_fastm_sin_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::sin<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_cos(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::cos(x); });
}
void BM_fastm_cos_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::cos<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_tan(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::tan(x); });
}
void BM_fastm_tan_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::tan<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_exp(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::exp(x); });
}
void BM_fastm_exp_bounds(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::exp<float, BoundsTraits>(x); });
}
void BM_fastm_exp_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::exp<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_exp2(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::exp2(x); });
}
void BM_fastm_exp2_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::exp2<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_exp10(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::exp10(x); });
}
void BM_fastm_exp10_accurate(benchmark::State& state) {
  bench::runBench(
      state, sinInputs(), [](auto x) { return dfm::exp10<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_log(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return dfm::log(x); });
}
void BM_fastm_log_accurate(benchmark::State& state) {
  bench::runBench(
      state, logInputs(), [](auto x) { return dfm::log<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_log2(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return dfm::log2(x); });
}
void BM_fastm_log2_accurate(benchmark::State& state) {
  bench::runBench(
      state, logInputs(), [](auto x) { return dfm::log2<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_log10(benchmark::State& state) {
  bench::runBench(state, logInputs(), [](auto x) { return dfm::log10(x); });
}
void BM_fastm_log10_accurate(benchmark::State& state) {
  bench::runBench(
      state, logInputs(), [](auto x) { return dfm::log10<float, dfm::MaxAccuracyTraits>(x); });
}
void BM_fastm_expm1(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::expm1(x); });
}
void BM_fastm_log1p(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::log1p(std::fabs(x)); });
}
void BM_fastm_tanh(benchmark::State& state) {
  bench::runBench(state, tanhInputs(), [](auto x) { return dfm::tanh(x); });
}

void BM_fastm_atan2(benchmark::State& state) {
  bench::runBench2(
      state, cbrtInputs(), sinInputs(), [](auto y, auto x) { return dfm::atan2(y, x); });
}
void BM_fastm_atan2_bounds(benchmark::State& state) {
  bench::runBench2(state, cbrtInputs(), sinInputs(), [](auto y, auto x) {
    return dfm::atan2<float, BoundsTraits>(y, x);
  });
}
void BM_fastm_hypot(benchmark::State& state) {
  bench::runBench2(
      state, hypotInputs(), sinInputs(), [](auto x, auto y) { return dfm::hypot(x, y); });
}
void BM_fastm_hypot_bounds(benchmark::State& state) {
  bench::runBench2(state, hypotInputs(), sinInputs(), [](auto x, auto y) {
    return dfm::hypot<float, dfm::MaxAccuracyTraits>(x, y);
  });
}
void BM_naive_hypot(benchmark::State& state) {
  bench::runBench2(
      state, hypotInputs(), sinInputs(), [](auto x, auto y) { return sqrtf(fmaf(x, x, y * y)); });
}
void BM_fastm_pow(benchmark::State& state) {
  bench::runBench2(
      state, powBaseInputs(), powExpInputs(), [](auto b, auto e) { return dfm::pow(b, e); });
}
void BM_fastm_pow_accurate(benchmark::State& state) {
  bench::runBench2(state, powBaseInputs(), powExpInputs(), [](auto b, auto e) {
    return dfm::pow<float, dfm::MaxAccuracyTraits>(b, e);
  });
}

// frexp and ldexp use non-standard loop patterns — kept hand-written.
void BM_fastm_frexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  int exp;
  int64_t expSum = 0;
  for (auto _ : state) {
    (void)_;
    sum += dfm::frexp(inputs[idx], &exp);
    expSum += exp;
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << " " << expSum << std::endl;
}

void BM_fastm_ldexp(benchmark::State& state) {
  const auto& inputs = sinInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto _ : state) {
    (void)_;
    sum += dfm::ldexp(inputs[idx], idx & 7);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// --- sincos / sincospi benchmarks ---

void BM_fastm_sin_plus_cos(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::sin(x) + dfm::cos(x); });
}
void BM_fastm_sincos(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) {
    float s, c;
    dfm::sincos(x, &s, &c);
    return s + c;
  });
}
void BM_fastm_sinpi(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::sinpi(x); });
}
void BM_fastm_cospi(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) { return dfm::cospi(x); });
}
void BM_fastm_sincospi(benchmark::State& state) {
  bench::runBench(state, sinInputs(), [](auto x) {
    float s, c;
    dfm::sincospi(x, &s, &c);
    return s + c;
  });
}

// --- Batch benchmarks: explicit SIMD via SseFloat (4-wide SSE) ---

#if defined(__SSE4_1__)
#include <dispenso/fast_math/float_traits_x86.h>

static void BM_batch_sinf(benchmark::State& state) {
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto _ : state) {
    (void)_;
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
  for (auto _ : state) {
    (void)_;
    for (size_t i = 0; i < kNumInputs; ++i) {
      outputs[i] = dfm::sin(inputs[i]);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

static void BM_batch_sin_sse(benchmark::State& state) {
  using namespace dispenso::fast_math;
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto _ : state) {
    (void)_;
    for (size_t i = 0; i < kNumInputs; i += 4) {
      SseFloat x = _mm_loadu_ps(&inputs[i]);
      SseFloat r = sin<SseFloat>(x);
      _mm_storeu_ps(&outputs[i], r.v);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

static void BM_batch_cos_scalar(benchmark::State& state) {
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto _ : state) {
    (void)_;
    for (size_t i = 0; i < kNumInputs; ++i) {
      outputs[i] = dfm::cos(inputs[i]);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

static void BM_batch_cos_sse(benchmark::State& state) {
  using namespace dispenso::fast_math;
  const auto& inputs = sinInputs();
  alignas(16) float outputs[kNumInputs];
  for (auto _ : state) {
    (void)_;
    for (size_t i = 0; i < kNumInputs; i += 4) {
      SseFloat x = _mm_loadu_ps(&inputs[i]);
      SseFloat r = cos<SseFloat>(x);
      _mm_storeu_ps(&outputs[i], r.v);
    }
    benchmark::DoNotOptimize(outputs);
  }
  state.SetItemsProcessed(state.iterations() * kNumInputs);
}

BENCHMARK(BM_batch_sinf);
BENCHMARK(BM_batch_sin_scalar);
BENCHMARK(BM_batch_sin_sse);
BENCHMARK(BM_batch_cos_scalar);
BENCHMARK(BM_batch_cos_sse);
#endif // __SSE4_1__

// --- Registrations ---

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

BENCHMARK(BM_hypot);
BENCHMARK(BM_fastm_hypot);
BENCHMARK(BM_naive_hypot);
BENCHMARK(BM_fastm_hypot_bounds);

BENCHMARK(BM_sin_plus_cos);
BENCHMARK(BM_fastm_sin_plus_cos);
BENCHMARK(BM_fastm_sincos);
BENCHMARK(BM_fastm_sinpi);
BENCHMARK(BM_fastm_cospi);
BENCHMARK(BM_fastm_sincospi);

BENCHMARK(BM_pow);
BENCHMARK(BM_fastm_pow);
BENCHMARK(BM_fastm_pow_accurate);

BENCHMARK(BM_expm1);
BENCHMARK(BM_fastm_expm1);
BENCHMARK(BM_log1p);
BENCHMARK(BM_fastm_log1p);
BENCHMARK(BM_tanh);
BENCHMARK(BM_fastm_tanh);
