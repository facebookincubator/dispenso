/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <limits>

#include "eval.h"

#include <gtest/gtest.h>

// We have reached asymptote for float32 at this value.
constexpr float kFarEnough = 20000000.0f;

TEST(Atan, SpecialVals) {
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  auto res = dispenso::fast_math::atan(-kFarEnough);
  EXPECT_EQ(res, -kPi_2);
  res = dispenso::fast_math::atan(kFarEnough);
  EXPECT_EQ(res, kPi_2);
  res = dispenso::fast_math::atan(0.0f);
  EXPECT_EQ(res, 0.0f);
  // NaN and Inf
  EXPECT_NE(dispenso::fast_math::atan(kNaN), dispenso::fast_math::atan(kNaN));
  EXPECT_EQ(dispenso::fast_math::atan(kInf), kPi_2);
  EXPECT_EQ(dispenso::fast_math::atan(-kInf), -kPi_2);
}

TEST(Atan, Range) {
  uint32_t ulps = dispenso::fast_math::evalAccuracy(
      atanf, dispenso::fast_math::atan<float>, -kFarEnough, kFarEnough);

  EXPECT_LE(ulps, 3);
}
