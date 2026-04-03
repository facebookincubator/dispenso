/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Asin, OutOfRange) {
  auto res = dispenso::fast_math::asin(1.00001f);
  EXPECT_NE(res, res);
  res = dispenso::fast_math::asin(-1.00001f);
  EXPECT_NE(res, res);
}

TEST(Asin, SpecialVals) {
  auto res = dispenso::fast_math::asin(-1.0f);
  EXPECT_EQ(res, -kPi_2);
  res = dispenso::fast_math::asin(1.0f);
  EXPECT_EQ(res, kPi_2);
  res = dispenso::fast_math::asin(0.0f);
  EXPECT_EQ(res, 0.0f);
}

TEST(Asin, Range) {
  uint32_t ulps =
      dispenso::fast_math::evalAccuracy(asinf, dispenso::fast_math::asin<float>, -1.0f, 1.0f);
  EXPECT_LE(ulps, 2);
}
