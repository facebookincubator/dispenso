/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Tan, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_NE(dispenso::fast_math::tan(-kInf), dispenso::fast_math::tan(-kInf));
  EXPECT_NE(dispenso::fast_math::tan(kInf), dispenso::fast_math::tan(kInf));
  EXPECT_NE(dispenso::fast_math::tan(kNaN), dispenso::fast_math::tan(kNaN));
  EXPECT_EQ(dispenso::fast_math::tan(0.0f), 0.0f);
}

constexpr uint32_t kTanAccurateUlps = 3;
constexpr uint32_t kTanAccurateUlpsLg = 4;

TEST(TanLessAccurate, Range8Pi) {
  auto result =
      dispenso::fast_math::evalAccuracy(::tanf, dispenso::fast_math::tan<float>, -8 * kPi, 8 * kPi);

  EXPECT_LE(result, kTanAccurateUlps);
}

TEST(TanLessAccurate, Range1KPi) {
  auto result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, -(1 << 10) * kPi, -8 * kPi);
  EXPECT_LE(result, kTanAccurateUlps);

  result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, 8 * kPi, (1 << 10) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);
}

TEST(TanLessAccurate, Range32KPi) {
  auto result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, -(1 << 15) * kPi, -(1 << 10) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);

  result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, (1 << 10) * kPi, (1 << 15) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);
}

auto tan_accurate = dispenso::fast_math::tan<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(TanAccurate, Range2MPi) {
  auto result =
      dispenso::fast_math::evalAccuracy(::tanf, tan_accurate, -(1 << 20) * kPi, -(1 << 15) * kPi);
  EXPECT_LE(result, kTanAccurateUlpsLg);

  result =
      dispenso::fast_math::evalAccuracy(::tanf, tan_accurate, (1 << 15) * kPi, (1 << 20) * kPi);
  EXPECT_LE(result, kTanAccurateUlpsLg);
}
