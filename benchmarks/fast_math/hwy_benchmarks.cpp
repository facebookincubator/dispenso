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

#if __has_include("hwy/highway.h")
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#if defined(__GNUC__) || defined(__clang__)
#define UNUSED_VAR myLocalForLoopVar __attribute__((unused))
#else
#define UNUSED_VAR myLocalForLoopVar
#endif

namespace dfm = dispenso::fast_math;
namespace hn = hwy::HWY_NAMESPACE;
using HwyFloat = dfm::HwyFloat;
using HwyFloatTag = dfm::HwyFloatTag;
using HwyVecF = hn::Vec<HwyFloatTag>;

constexpr size_t kNumInputs = 4096;
constexpr size_t kInputsMask = 4095;

// Helper to sum lanes and prevent optimization.
static void consumeSum(HwyVecF sum) {
  const HwyFloatTag d;
  constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
  HWY_ALIGN float buf[kMaxLanes];
  hn::StoreU(sum, d, buf);
  float total = 0.0f;
  const size_t N = hn::Lanes(d);
  for (size_t i = 0; i < N; ++i) {
    total += buf[i];
  }
  std::cout << total << std::endl;
}

// --- sin ---

const std::vector<HwyVecF>& sinHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = static_cast<float>(M_PI / kNumInputs);
    std::vector<HwyVecF> inp;
    float f = static_cast<float>(-M_PI / 2.0);
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
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

void BM_sin_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::sin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_cos_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::cos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- exp ---

const std::vector<HwyVecF>& expHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = 20.0f / kNumInputs;
    std::vector<HwyVecF> inp;
    float f = -10.0f;
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
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

void BM_exp_hwy(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::exp(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

const std::vector<HwyVecF>& logHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = 10000.0f / kNumInputs;
    std::vector<HwyVecF> inp;
    float f = 0.001f;
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
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

void BM_log_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::log(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_cbrt_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::cbrt(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_atan_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::atan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- acos ---

const std::vector<HwyVecF>& acosHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = 2.0f / kNumInputs;
    std::vector<HwyVecF> inp;
    float f = -0.999f;
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
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

void BM_acos_hwy(benchmark::State& state) {
  const auto& inputs = acosHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::acos(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- Accurate and bounds variants ---

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

void BM_sin_hwy_accurate(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::sin<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_cos_hwy_accurate(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::cos<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_exp_hwy_accurate(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::exp<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_exp_hwy_bounds(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::exp<HwyVecF, BoundsTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_log_hwy_accurate(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::log<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_cbrt_hwy_accurate(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::cbrt<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_tan_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::tan(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_asin_hwy(benchmark::State& state) {
  const auto& inputs = acosHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::asin(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_atan2_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = hn::Add(sum, dfm::atan2(inputs[idx], inputs[idx2]));
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_exp2_hwy(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::exp2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_exp10_hwy(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::exp10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_log2_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::log2(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_log10_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::log10(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_frexp_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    dfm::IntType_t<HwyVecF> e;
    sum = hn::Add(sum, dfm::frexp(inputs[idx], &e));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
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

void BM_ldexp_hwy(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  const hn::RebindToSigned<HwyFloatTag> di;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::ldexp(inputs[idx], hn::Set(di, 3)));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- hypot ---

const std::vector<HwyVecF>& hypotHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = 200000.0f / kNumInputs;
    std::vector<HwyVecF> inp;
    float f = -100000.0f;
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
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

void BM_hypot_hwy(benchmark::State& state) {
  const auto& inputs = hypotHwyInputs();
  const auto& inputs2 = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::hypot(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_hypot_hwy_bounds(benchmark::State& state) {
  const auto& inputs = hypotHwyInputs();
  const auto& inputs2 = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::hypot<HwyVecF, dfm::MaxAccuracyTraits>(inputs[idx], inputs2[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- Highway contrib/math comparison benchmarks ---

void BM_sin_hwy_contrib(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Sin(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_cos_hwy_contrib(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Cos(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_exp_hwy_contrib(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Exp(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_exp2_hwy_contrib(benchmark::State& state) {
  const auto& inputs = expHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Exp2(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_log_hwy_contrib(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Log(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_log2_hwy_contrib(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Log2(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_log10_hwy_contrib(benchmark::State& state) {
  const auto& inputs = logHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Log10(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_atan_hwy_contrib(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Atan(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_acos_hwy_contrib(benchmark::State& state) {
  const auto& inputs = acosHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Acos(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_asin_hwy_contrib(benchmark::State& state) {
  const auto& inputs = acosHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, hn::Asin(d, inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

void BM_atan2_hwy_contrib(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    size_t idx2 = (idx + 1) & kInputsMask;
    sum = hn::Add(sum, hn::Atan2(d, inputs[idx], inputs[idx2]));
    idx = (idx + 2) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// Registrations.
BENCHMARK(BM_sin_libc);
BENCHMARK(BM_sin_scalar);
BENCHMARK(BM_sin_hwy);
BENCHMARK(BM_sin_hwy_accurate);
BENCHMARK(BM_sin_hwy_contrib);

BENCHMARK(BM_cos_scalar);
BENCHMARK(BM_cos_hwy);
BENCHMARK(BM_cos_hwy_accurate);
BENCHMARK(BM_cos_hwy_contrib);

BENCHMARK(BM_tan_scalar);
BENCHMARK(BM_tan_hwy);

BENCHMARK(BM_exp_libc);
BENCHMARK(BM_exp_scalar);
BENCHMARK(BM_exp_hwy);
BENCHMARK(BM_exp_hwy_bounds);
BENCHMARK(BM_exp_hwy_accurate);
BENCHMARK(BM_exp_hwy_contrib);

BENCHMARK(BM_exp2_scalar);
BENCHMARK(BM_exp2_hwy);
BENCHMARK(BM_exp2_hwy_contrib);

BENCHMARK(BM_exp10_scalar);
BENCHMARK(BM_exp10_hwy);

BENCHMARK(BM_log_libc);
BENCHMARK(BM_log_scalar);
BENCHMARK(BM_log_hwy);
BENCHMARK(BM_log_hwy_accurate);
BENCHMARK(BM_log_hwy_contrib);

BENCHMARK(BM_log2_scalar);
BENCHMARK(BM_log2_hwy);
BENCHMARK(BM_log2_hwy_contrib);

BENCHMARK(BM_log10_scalar);
BENCHMARK(BM_log10_hwy);
BENCHMARK(BM_log10_hwy_contrib);

BENCHMARK(BM_cbrt_scalar);
BENCHMARK(BM_cbrt_hwy);
BENCHMARK(BM_cbrt_hwy_accurate);

BENCHMARK(BM_atan_scalar);
BENCHMARK(BM_atan_hwy);
BENCHMARK(BM_atan_hwy_contrib);

BENCHMARK(BM_acos_scalar);
BENCHMARK(BM_acos_hwy);
BENCHMARK(BM_acos_hwy_contrib);

BENCHMARK(BM_asin_scalar);
BENCHMARK(BM_asin_hwy);
BENCHMARK(BM_asin_hwy_contrib);

BENCHMARK(BM_atan2_scalar);
BENCHMARK(BM_atan2_hwy);
BENCHMARK(BM_atan2_hwy_contrib);

BENCHMARK(BM_frexp_scalar);
BENCHMARK(BM_frexp_hwy);

BENCHMARK(BM_ldexp_scalar);
BENCHMARK(BM_ldexp_hwy);

BENCHMARK(BM_hypot_scalar);
BENCHMARK(BM_hypot_hwy);
BENCHMARK(BM_hypot_hwy_bounds);

// --- expm1 ---

void BM_expm1_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::expm1(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- log1p ---

void BM_log1p_hwy(benchmark::State& state) {
  const auto& inputs = sinHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    HwyVecF ax = hn::Abs(inputs[idx]);
    sum = hn::Add(sum, dfm::log1p(ax));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

// --- tanh ---

const std::vector<HwyVecF>& tanhHwyInputs() {
  static std::vector<HwyVecF> inputs = []() {
    const HwyFloatTag d;
    const size_t N = hn::Lanes(d);
    float delta = 10.0f / kNumInputs;
    std::vector<HwyVecF> inp;
    float f = -5.0f;
    constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);
    HWY_ALIGN float buf[kMaxLanes];
    for (size_t i = 0; i < kNumInputs; ++i) {
      for (size_t j = 0; j < N; ++j) {
        buf[j] = f + static_cast<float>(j) * delta;
      }
      inp.emplace_back(hn::Load(d, buf));
      f += static_cast<float>(N) * delta;
    }
    return inp;
  }();
  return inputs;
}

void BM_tanh_hwy(benchmark::State& state) {
  const auto& inputs = tanhHwyInputs();
  size_t idx = 0;
  const HwyFloatTag d;
  HwyVecF sum = hn::Zero(d);
  for (auto UNUSED_VAR : state) {
    sum = hn::Add(sum, dfm::tanh(inputs[idx]));
    idx = (idx + 1) & kInputsMask;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(hn::Lanes(HwyFloatTag())));
  consumeSum(sum);
}

BENCHMARK(BM_expm1_hwy);
BENCHMARK(BM_log1p_hwy);
BENCHMARK(BM_tanh_hwy);

#else // !__has_include("hwy/highway.h")

int main() {
  std::cout << "Highway not available, skipping benchmarks." << std::endl;
  return 0;
}

#endif // __has_include("hwy/highway.h")
