/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

auto sin_accurate = dispenso::fast_math::sin<float, dispenso::fast_math::MaxAccuracyTraits>;
// MacOS seems to have a slightly busted sin and sinf, but sinl seems to agree with other platforms.
#if defined(__APPLE__)
float gt_sin(float x) {
  return ::sinl(x);
}
#else
auto gt_sin = ::sinf;
#endif //__APPLE__

TEST(Sin, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
  auto sin = dispenso::fast_math::sin<float, dispenso::fast_math::MaxAccuracyTraits>;

  EXPECT_NE(sin(-kInf), sin(-kInf));
  EXPECT_NE(sin(kInf), sin(kInf));
  EXPECT_NE(sin(kNaN), sin(kNaN));
  EXPECT_EQ(sin(0.0f), 0.0f);
}

constexpr uint32_t kSinAccurateUlps = 1;
constexpr uint32_t kSinAccurateUlpsVeryLarge = 2;

constexpr uint32_t kSinUlps = 1;
constexpr uint32_t kSinUlpsLarge = 2;

TEST(Sin, RangePi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_sin, sin_accurate, -kPi, kPi);

  EXPECT_LE(result, kSinAccurateUlps);
}

TEST(Sin, Range128Pi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_sin, sin_accurate, -128 * kPi, 128 * kPi);

  EXPECT_LE(result, kSinAccurateUlps);
}

TEST(Sin, Range1MPi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_sin, sin_accurate, -(1 << 20) * kPi, -128 * kPi);
  EXPECT_LE(result, kSinAccurateUlpsVeryLarge);
  result = dispenso::fast_math::evalAccuracy(gt_sin, sin_accurate, 128 * kPi, (1 << 20) * kPi);

  EXPECT_LE(result, kSinAccurateUlpsVeryLarge);
}

TEST(SinLessAccurate, RangePi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_sin, dispenso::fast_math::sin<float>, -kPi, kPi);

  EXPECT_LE(result, kSinUlps);
}

TEST(SinLessAccurate, Range128Pi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_sin, dispenso::fast_math::sin<float>, -128 * kPi, -kPi);

  EXPECT_LE(result, kSinUlps);

  result =
      dispenso::fast_math::evalAccuracy(gt_sin, dispenso::fast_math::sin<float>, kPi, 128 * kPi);

  EXPECT_LE(result, kSinUlps);
}

TEST(SinLessAccurate, Range32768Pi) {
  auto result = dispenso::fast_math::evalAccuracy(
      gt_sin, dispenso::fast_math::sin<float>, -32768 * kPi, -128 * kPi);

  EXPECT_LE(result, kSinUlpsLarge);

  result = dispenso::fast_math::evalAccuracy(
      gt_sin, dispenso::fast_math::sin<float>, 128 * kPi, 32768 * kPi);

  EXPECT_LE(result, kSinUlpsLarge);
}
