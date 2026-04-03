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

#if defined(__aarch64__)

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED_VAR myLocalForLoopVar __attribute__((unused))
#else
#define UNUSED_VAR myLocalForLoopVar
#endif

namespace dfm = dispenso::fast_math;

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;

// Helper to sum lanes and prevent optimization.
static void consumeSum(float32x4_t sum) {
  alignas(16) float buf[4];
  vst1q_f32(buf, sum);
  float total = buf[0] + buf[1] + buf[2] + buf[3];
  std::cout << total << std::endl;
}

// --- sin ---

const std::vector<float32x4_t>& sinNeonInputs() {
  static std::vector<float32x4_t> inputs = []() {
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<float32x4_t> inp;
    float f = static_cast<float>(-M_PI / 2.0);
    for (size_t i = 0; i < kNumInputs; ++i) {
      float buf[4] = {f, f + delta, f + 2 * delta, f + 3 * delta};
      inp.emplace_back(vld1q_f32(buf));
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

void BM_sin_neon(benchmark::State& state) {
  const auto& inputs = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::sin(inputs[idx]));
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

void BM_cos_neon(benchmark::State& state) {
  const auto& inputs = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::cos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- exp ---

const std::vector<float32x4_t>& expNeonInputs() {
  static std::vector<float32x4_t> inputs = []() {
    float delta = 20.0f / kNumInputs;
    std::vector<float32x4_t> inp;
    float f = -10.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      float buf[4] = {f, f + delta, f + 2 * delta, f + 3 * delta};
      inp.emplace_back(vld1q_f32(buf));
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

void BM_exp_neon(benchmark::State& state) {
  const auto& inputs = expNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::exp(inputs[idx]));
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

const std::vector<float32x4_t>& logNeonInputs() {
  static std::vector<float32x4_t> inputs = []() {
    float delta = 10000.0f / kNumInputs;
    std::vector<float32x4_t> inp;
    float f = 0.001f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      float buf[4] = {f, f + delta, f + 2 * delta, f + 3 * delta};
      inp.emplace_back(vld1q_f32(buf));
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

void BM_log_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::log(inputs[idx]));
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

void BM_cbrt_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::cbrt(inputs[idx]));
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

void BM_atan_neon(benchmark::State& state) {
  const auto& inputs = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::atan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- acos ---

const std::vector<float32x4_t>& acosNeonInputs() {
  static std::vector<float32x4_t> inputs = []() {
    float delta = 2.0f / kNumInputs;
    std::vector<float32x4_t> inp;
    float f = -0.999f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      float buf[4] = {f, f + delta, f + 2 * delta, f + 3 * delta};
      inp.emplace_back(vld1q_f32(buf));
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

void BM_acos_neon(benchmark::State& state) {
  const auto& inputs = acosNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::acos(inputs[idx]));
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

void BM_tan_neon(benchmark::State& state) {
  const auto& inputs = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::tan(inputs[idx]));
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

void BM_asin_neon(benchmark::State& state) {
  const auto& inputs = acosNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::asin(inputs[idx]));
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

void BM_atan2_neon(benchmark::State& state) {
  const auto& inputs = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = vaddq_f32(sum, dfm::atan2(inputs[idx], inputs[idx2]));
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

void BM_exp2_neon(benchmark::State& state) {
  const auto& inputs = expNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::exp2(inputs[idx]));
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

void BM_exp10_neon(benchmark::State& state) {
  const auto& inputs = expNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::exp10(inputs[idx]));
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

void BM_log2_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::log2(inputs[idx]));
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

void BM_log10_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::log10(inputs[idx]));
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

void BM_frexp_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    dfm::IntType_t<float32x4_t> e;
    sum = vaddq_f32(sum, dfm::frexp(inputs[idx], &e));
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

void BM_ldexp_neon(benchmark::State& state) {
  const auto& inputs = logNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::ldexp(inputs[idx], vdupq_n_s32(3)));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

// --- hypot ---

const std::vector<float32x4_t>& hypotNeonInputs() {
  static std::vector<float32x4_t> inputs = []() {
    float delta = 200000.0f / kNumInputs;
    std::vector<float32x4_t> inp;
    float f = -100000.0f;
    for (size_t i = 0; i < kNumInputs; ++i) {
      float buf[4] = {f, f + delta, f + 2 * delta, f + 3 * delta};
      inp.emplace_back(vld1q_f32(buf));
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

void BM_hypot_neon(benchmark::State& state) {
  const auto& inputs = hypotNeonInputs();
  const auto& inputs2 = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum = vaddq_f32(sum, dfm::hypot(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_hypot_neon_bounds(benchmark::State& state) {
  const auto& inputs = hypotNeonInputs();
  const auto& inputs2 = sinNeonInputs();
  size_t idx = 0;
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (auto UNUSED_VAR : state) {
    sum =
        vaddq_f32(sum, dfm::hypot<float32x4_t, dfm::MaxAccuracyTraits>(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * 4);
  consumeSum(sum);
}

void BM_hypot_libc(benchmark::State& state) {
  const auto& inputs = hypotScalarInputs();
  const auto& inputs2 = sinScalarInputs();
  size_t idx = 0;
  float sum = 0.0f;
  for (auto UNUSED_VAR : state) {
    sum += ::hypotf(inputs[idx], inputs2[idx]);
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations());
  std::cout << sum << std::endl;
}

// Registrations.
BENCHMARK(BM_sin_libc);
BENCHMARK(BM_sin_scalar);
BENCHMARK(BM_sin_neon);

BENCHMARK(BM_cos_scalar);
BENCHMARK(BM_cos_neon);

BENCHMARK(BM_tan_scalar);
BENCHMARK(BM_tan_neon);

BENCHMARK(BM_exp_libc);
BENCHMARK(BM_exp_scalar);
BENCHMARK(BM_exp_neon);

BENCHMARK(BM_exp2_scalar);
BENCHMARK(BM_exp2_neon);

BENCHMARK(BM_exp10_scalar);
BENCHMARK(BM_exp10_neon);

BENCHMARK(BM_log_libc);
BENCHMARK(BM_log_scalar);
BENCHMARK(BM_log_neon);

BENCHMARK(BM_log2_scalar);
BENCHMARK(BM_log2_neon);

BENCHMARK(BM_log10_scalar);
BENCHMARK(BM_log10_neon);

BENCHMARK(BM_cbrt_scalar);
BENCHMARK(BM_cbrt_neon);

BENCHMARK(BM_atan_scalar);
BENCHMARK(BM_atan_neon);

BENCHMARK(BM_acos_scalar);
BENCHMARK(BM_acos_neon);

BENCHMARK(BM_asin_scalar);
BENCHMARK(BM_asin_neon);

BENCHMARK(BM_atan2_scalar);
BENCHMARK(BM_atan2_neon);

BENCHMARK(BM_frexp_scalar);
BENCHMARK(BM_frexp_neon);

BENCHMARK(BM_ldexp_scalar);
BENCHMARK(BM_ldexp_neon);

BENCHMARK(BM_hypot_libc);
BENCHMARK(BM_hypot_scalar);
BENCHMARK(BM_hypot_neon);
BENCHMARK(BM_hypot_neon_bounds);

#else // !defined(__aarch64__)

int main() {
  std::cout << "AArch64 NEON not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // defined(__aarch64__)
