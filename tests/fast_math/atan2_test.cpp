/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <random>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Atan2, SpecialVals) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  auto res = dispenso::fast_math::atan2(0.0f, -1.0f);
  EXPECT_EQ(res, kPi);
  res = dispenso::fast_math::atan2(-0.0f, -1.0f);
  EXPECT_EQ(res, -kPi);

  res = dispenso::fast_math::atan2(0.0f, 1.0f);
  EXPECT_EQ(res, static_cast<float>(0.0f));
  res = dispenso::fast_math::atan2(-0.0f, 1.0f);
  EXPECT_EQ(
      dispenso::fast_math::bit_cast<uint32_t>(res), dispenso::fast_math::bit_cast<uint32_t>(-0.0f));

  res = dispenso::fast_math::atan2(-1.0f, 0.0f);
  EXPECT_EQ(res, -kPi_2);
  res = dispenso::fast_math::atan2(1.0f, 0.0f);
  EXPECT_EQ(res, kPi_2);
  res = dispenso::fast_math::atan2(0.0f, -0.0f);
  EXPECT_EQ(res, kPi);
  res = dispenso::fast_math::atan2(-0.0f, -0.0f);
  EXPECT_EQ(res, -kPi);
  res = dispenso::fast_math::atan2(0.0f, 0.0f);
  EXPECT_EQ(res, 0.0f);
  res = dispenso::fast_math::atan2(-0.0f, 0.0f);
  EXPECT_EQ(res, 0.0f);

  res = dispenso::fast_math::atan2(kNaN, 1.0f);
  EXPECT_NE(res, res);
  res = dispenso::fast_math::atan2(1.0f, kNaN);
  EXPECT_NE(res, res);
  res = dispenso::fast_math::atan2(kNaN, kNaN);
  EXPECT_NE(res, res);

  res = dispenso::fast_math::atan2(1.0f, -kInf);
  EXPECT_EQ(res, kPi);
  res = dispenso::fast_math::atan2(-1.0f, -kInf);
  EXPECT_EQ(res, -kPi);

  res = dispenso::fast_math::atan2(1.0f, kInf);
  EXPECT_EQ(res, static_cast<float>(0.0f));
  res = dispenso::fast_math::atan2(-1.0f, kInf);
  EXPECT_EQ(res, static_cast<float>(0.0f));

  res = dispenso::fast_math::atan2(kInf, 1.0f);
  EXPECT_EQ(res, kPi_2);
}

struct BoundsTraits {
  static constexpr bool kBoundsValues = true;
  static constexpr bool kMaxAccuracy = false; // This isn't actually used
};

TEST(Atan2WBounds, SpecialVals) {
  constexpr float kInf = std::numeric_limits<float>::infinity();

  auto atan2_bounds = dispenso::fast_math::atan2<float, BoundsTraits>;

  auto res = atan2_bounds(kInf, -1.0f);
  EXPECT_EQ(res, -kPi_2);

  res = atan2_bounds(kInf, -kInf);
  EXPECT_FLOAT_EQ(res, 3.0f * kPi_4);
  res = atan2_bounds(-kInf, -kInf);
  EXPECT_FLOAT_EQ(res, -3.0f * kPi_4);

  res = atan2_bounds(kInf, kInf);
  EXPECT_FLOAT_EQ(res, kPi_4);
  res = atan2_bounds(-kInf, kInf);
  EXPECT_FLOAT_EQ(res, -kPi_4);
}

TEST(Atan2, RangeNearZero) {
  constexpr int32_t kMax = 2048;

  float quantizer = 0.5f;
  quantizer *= quantizer;
  quantizer *= quantizer;
  quantizer *= quantizer;
  quantizer *= 0.25f;

  uint32_t maxUlps = 0;
  for (int32_t y = -kMax; y <= kMax; ++y) {
    for (int32_t x = -kMax; x <= kMax; ++x) {
      float yf = static_cast<float>(y) * quantizer;
      float xf = static_cast<float>(x) * quantizer;
      maxUlps = std::max(
          maxUlps,
          dispenso::fast_math::float_distance(
              ::atan2f(yf, xf), dispenso::fast_math::atan2(yf, xf)));
    }
  }
  EXPECT_LE(maxUlps, 3);
}

TEST(Atan, RandomSamples) {
  constexpr size_t kNumSamples = 1 << 24;

  std::mt19937 gen(77); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(-1e5f, 1e5f);

  uint32_t maxUlps = 0;
  for (size_t i = 0; i < kNumSamples; ++i) {
    // get random sample
    float yf = dis(gen);
    float xf = dis(gen);
    float gt = ::atan2f(yf, xf);
    float apx = dispenso::fast_math::atan2(yf, xf);
    uint32_t ulps = dispenso::fast_math::float_distance(gt, apx);
    maxUlps = std::max(maxUlps, ulps);
  }

  EXPECT_LE(maxUlps, 3);
}
