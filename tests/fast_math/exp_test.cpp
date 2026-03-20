/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

struct BoundsOnlyTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

auto exp_accurate = dispenso::fast_math::exp<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Exp, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp_accurate(0.0f), 1.0f);
  EXPECT_EQ(exp_accurate(-kInf), 0.0f);
  EXPECT_EQ(exp_accurate(kInf), kInf);
  EXPECT_NE(exp_accurate(kNaN), exp_accurate(kNaN));
}

TEST(Exp, RangeSmall) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, -127.0f, -40.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp, RangeSmallish) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, -40.0f, 20.0f);
  EXPECT_LE(res, 1u);
}
TEST(Exp, RangeMedium) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, 20.0f, 40.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp, RangeLarge) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, 40.0f, 128.0f);
  EXPECT_LE(res, 3u);
}

auto exp_bounds = dispenso::fast_math::exp<float, BoundsOnlyTraits>;

TEST(ExpLessAccurateWBounds, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp_bounds(0.0f), 1.0f);
  EXPECT_EQ(exp_bounds(-kInf), 0.0f);
  EXPECT_EQ(exp_bounds(kInf), kInf);
  EXPECT_NE(exp_bounds(kNaN), exp_bounds(kNaN));
  EXPECT_LT(exp_bounds(-100.0f), 1e-38f);
}

TEST(ExpLessAccurateWBounds, Range_m100_100) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_bounds, -100.0f, 100.0f);
  EXPECT_LE(res, 5u);
}

TEST(ExpLessAccurate, Range_m88_88) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::expf, dispenso::fast_math::exp<float>, -88.0f, 88.0f);
  EXPECT_LE(res, 5u);
}
