/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

auto log_accurate = dispenso::fast_math::log<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Log, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log_accurate(0.0f), -kInf);
  EXPECT_NE(log_accurate(-1.0f), log_accurate(-1.0f));
  EXPECT_EQ(log_accurate(kInf), kInf);
  EXPECT_NE(log_accurate(kNaN), log_accurate(kNaN));
}

TEST(LogAccurate, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, log_accurate, 0.0f, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(LogAccurate, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, log_accurate, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(Log, RangeNeg) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::logf, dispenso::fast_math::log<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(Log, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, dispenso::fast_math::log<float>, 1.0f);
  EXPECT_LE(res, 2u);
}
