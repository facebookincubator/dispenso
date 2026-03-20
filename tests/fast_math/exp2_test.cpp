/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Exp2, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  auto exp2 = dispenso::fast_math::exp2<float, dispenso::fast_math::MaxAccuracyTraits>;

  EXPECT_EQ(exp2(0.0f), 1.0f);
  EXPECT_EQ(exp2(-kInf), 0.0f);
  EXPECT_EQ(exp2(kInf), kInf);
  EXPECT_NE(exp2(kNaN), exp2(kNaN));
}

#ifdef _WIN32
auto gtfunc = [](float f) { return static_cast<float>(::exp2l(f)); };
#else
auto gtfunc = ::exp2f;
#endif //_WIN32

TEST(Exp2, RangeSmall) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, -127.0f, -1.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp2, RangeMedium) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, -1.0f, 0.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp2, RangeLarge) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, 0.0f, 128.0f);
  EXPECT_LE(res, 1u);
}
