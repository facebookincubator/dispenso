/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

auto log2_w_bounds = dispenso::fast_math::log2<float, dispenso::fast_math::MaxAccuracyTraits>;

#ifdef _WIN32
float groundTruth(float x) {
  return static_cast<float>(::log2l(x));
}
#else
auto groundTruth = ::log2f;
#endif // _WIN32

TEST(Log2, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log2_w_bounds(0.0f), -kInf);
  EXPECT_NE(log2_w_bounds(-1.0f), log2_w_bounds(-1.0f));
  EXPECT_EQ(log2_w_bounds(kInf), kInf);
  EXPECT_NE(log2_w_bounds(kNaN), log2_w_bounds(kNaN));

  EXPECT_EQ(log2_w_bounds(1e-45f), groundTruth(1e-45f));
}

constexpr int kNegUlps = 1;
constexpr int kPosUlps = 1;

TEST(Log2WBounds, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, log2_w_bounds, 0.0f, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kNegUlps));
}

TEST(Log2WBounds, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, log2_w_bounds, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kPosUlps));
}

TEST(Log2, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(
      groundTruth, dispenso::fast_math::log2<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kNegUlps));
}

TEST(Log2, RangePos) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(groundTruth, dispenso::fast_math::log2<float>, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kPosUlps));
}
