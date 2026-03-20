/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

#include <iomanip>
#include <limits>

auto cbrt_acc = dispenso::fast_math::cbrt<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Cbrt, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(cbrt_acc(0.0f), 0.0f) << std::setprecision(16) << cbrt_acc(0.0f);
  EXPECT_EQ(cbrt_acc(-kInf), -kInf);
  EXPECT_EQ(cbrt_acc(kInf), kInf);
  EXPECT_NE(cbrt_acc(kNaN), cbrt_acc(kNaN));
}

constexpr uint32_t kCbrtOuterUlps = 13; // 12 bits on Linux, MacOs, 13 with MSVC
constexpr float kCbrtMaxErrorSmallDomain = 4.2e-9f; // 4e-9f Linux, MacOs;

TEST(Cbrt, RangeNeg) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::epsilon());

  EXPECT_LE(result, kCbrtOuterUlps);
}
TEST(Cbrt, RangePos) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::max());

  EXPECT_LE(result, kCbrtOuterUlps);
}
TEST(Cbrt, RangeSmall) {
  float resAbs = dispenso::fast_math::evalAccuracyAbs(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      -std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::epsilon());

  EXPECT_LE(resAbs, kCbrtMaxErrorSmallDomain);
}

constexpr uint32_t kCbrtOuterUlpsAcc = 3;
constexpr float kCbrtMaxErrorSmallDomainAcc = 1e-9f;

TEST(CbrtAccurate, RangeNeg) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf, cbrt_acc, -std::numeric_limits<float>::max(), -std::numeric_limits<float>::epsilon());

  EXPECT_LE(result, kCbrtOuterUlpsAcc);
}
TEST(CbrtAccurate, RangePos) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf, cbrt_acc, std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max());

  EXPECT_LE(result, kCbrtOuterUlpsAcc);
}
TEST(CbrtAccurate, RangeSmall) {
  float resAbs = dispenso::fast_math::evalAccuracyAbs(
      cbrtf,
      cbrt_acc,
      -std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::epsilon());

  EXPECT_LE(resAbs, kCbrtMaxErrorSmallDomainAcc);
}
