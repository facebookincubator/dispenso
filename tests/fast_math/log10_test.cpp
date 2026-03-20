/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

auto log10_accurate = dispenso::fast_math::log10<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Log10, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log10_accurate(0.0f), -kInf);
  EXPECT_NE(log10_accurate(-1.0f), log10_accurate(-1.0f));
  EXPECT_EQ(log10_accurate(kInf), kInf);
  EXPECT_NE(log10_accurate(kNaN), log10_accurate(kNaN));
}

TEST(Log10Accurate, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::log10f, log10_accurate, 0.0f, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10Accurate, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::log10f, log10_accurate, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10, RangeNeg) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::log10f, dispenso::fast_math::log10<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10, RangePos) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::log10f, dispenso::fast_math::log10<float>, 1.0f);
  EXPECT_LE(res, 3u);
}
