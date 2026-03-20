/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Ldexp, SpecialVals) {
  auto res = dispenso::fast_math::ldexp(-std::numeric_limits<float>::infinity(), 1);
  EXPECT_EQ(res, -std::numeric_limits<float>::infinity());
  res = dispenso::fast_math::ldexp(std::numeric_limits<float>::infinity(), 2);
  EXPECT_EQ(res, std::numeric_limits<float>::infinity());

  res = dispenso::fast_math::ldexp(std::numeric_limits<float>::quiet_NaN(), 7);
  EXPECT_NE(res, res);

  int exp;
  float tval = -0.49999997f;

  float a = dispenso::fast_math::frexp(tval, &exp);
  res = dispenso::fast_math::ldexp(a, exp);
  EXPECT_EQ(res, tval);
}

TEST(Ldexp, Range) {
  auto test = [](float f) {
    int exp;
    float a = dispenso::fast_math::frexp(f, &exp);
    float res = dispenso::fast_math::ldexp(a, exp);
    EXPECT_EQ(res, f);
  };

  dispenso::fast_math::evalForEach(
      -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), test);
}
