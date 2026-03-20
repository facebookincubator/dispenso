/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

auto cos_accurate = dispenso::fast_math::cos<float, dispenso::fast_math::MaxAccuracyTraits>;

// MacOS seems to have a slightly busted cos and cosf, but cosl seems to agree with other platforms.
#if defined(__APPLE__)
float gt_cos(float x) {
  return ::cosl(x);
}
#else
auto gt_cos = ::cosf;
#endif //__APPLE__

TEST(Cos, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
  auto cos = dispenso::fast_math::cos<float, dispenso::fast_math::MaxAccuracyTraits>;

  EXPECT_NE(cos(-kInf), cos(-kInf));
  EXPECT_NE(cos(kInf), cos(kInf));
  EXPECT_NE(cos(kNaN), cos(kNaN));
  EXPECT_EQ(cos(0.0f), 1.0f);
}

constexpr uint32_t kCosAccurateUlps = 1;
constexpr uint32_t kCosAccurateUlpsLarge = 1;
constexpr uint32_t kCosAccurateUlpsVeryLarge = 2;

constexpr uint32_t kCosUlps = 1;
constexpr uint32_t kCosUlpsLarge = 1;
constexpr uint32_t kCosUlpsVeryLarge = 2;

TEST(Cos, RangePi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -kPi, kPi);

  EXPECT_LE(result, kCosAccurateUlps);
}

TEST(Cos, Range128Pi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -128 * kPi, 128 * kPi);

  EXPECT_LE(result, kCosAccurateUlpsLarge);
}

TEST(Cos, Range1MPi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -(1 << 20) * kPi, -128 * kPi);
  EXPECT_LE(result, kCosAccurateUlpsVeryLarge);
  result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, 128 * kPi, (1 << 20) * kPi);

  EXPECT_LE(result, kCosAccurateUlpsVeryLarge);
}

TEST(CosLessAccurate, RangePi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, -kPi, kPi);

  EXPECT_LE(result, kCosUlps);
}

TEST(CosLessAccurate, Range128Pi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, -128 * kPi, -kPi);

  EXPECT_LE(result, kCosUlpsLarge);

  result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, kPi, 128 * kPi);

  EXPECT_LE(result, kCosUlpsLarge);
}

TEST(CosLessAccurate, Range32768Pi) {
  auto result = dispenso::fast_math::evalAccuracy(
      gt_cos, dispenso::fast_math::cos<float>, -32768 * kPi, -128 * kPi);

  EXPECT_LE(result, kCosUlpsVeryLarge);

  result = dispenso::fast_math::evalAccuracy(
      gt_cos, dispenso::fast_math::cos<float>, 128 * kPi, 32768 * kPi);

  EXPECT_LE(result, kCosUlpsVeryLarge);
}
