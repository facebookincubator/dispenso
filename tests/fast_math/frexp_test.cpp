/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Frexp, SpecialVals) {
  int exp;
  auto res = dispenso::fast_math::frexp(-std::numeric_limits<float>::infinity(), &exp);
  EXPECT_EQ(res, -std::numeric_limits<float>::infinity());
  res = dispenso::fast_math::frexp(std::numeric_limits<float>::infinity(), &exp);
  EXPECT_EQ(res, std::numeric_limits<float>::infinity());

  res = dispenso::fast_math::frexp(std::numeric_limits<float>::quiet_NaN(), &exp);
  EXPECT_NE(res, res);
}

TEST(Frexp, RangeNeg) {
  dispenso::fast_math::evalForEach(-std::numeric_limits<float>::max(), 0.0f, [](float f) {
    int exp;
    int expT;
    float a = ::frexpf(f, &exp);
    float aT = dispenso::fast_math::frexp(f, &expT);
    EXPECT_EQ(exp, expT) << "frexpf(" << f << ",&exp)";
    EXPECT_EQ(a, aT) << "frexpf(" << f << ",&exp)";
  });
}

TEST(Frexp, RangePos) {
  dispenso::fast_math::evalForEach(0.0f, std::numeric_limits<float>::max(), [](float f) {
    int exp;
    int expT;
    float a = ::frexpf(f, &exp);
    float aT = dispenso::fast_math::frexp(f, &expT);
    EXPECT_EQ(exp, expT) << "frexpf(" << f << ",&exp)";
    EXPECT_EQ(a, aT) << "frexpf(" << f << ",&exp)";
  });
}
