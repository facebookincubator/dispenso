/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

float groundTruth(float input) {
  return ::powf(10.0f, input);
}

auto exp10_accurate = dispenso::fast_math::exp10<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Exp10, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp10_accurate(0.0f), 1.0f);
  EXPECT_EQ(exp10_accurate(-kInf), 0.0f);
  EXPECT_EQ(exp10_accurate(kInf), kInf);
  EXPECT_NE(exp10_accurate(kNaN), exp10_accurate(kNaN));
}

TEST(Exp10, Range) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, exp10_accurate, -40.0f, 40.0f);
  EXPECT_LE(res, 3u);
}

TEST(Exp10LessAccurate, RangeMedium) {
  uint32_t res = dispenso::fast_math::evalAccuracy(
      groundTruth, dispenso::fast_math::exp10<float>, -37.0f, 38.0f);
  EXPECT_LE(res, 3u);
}
