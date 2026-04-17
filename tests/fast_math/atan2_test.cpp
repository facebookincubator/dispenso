/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <random>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;
using namespace dispenso::fast_math::testing;

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

// Wrappers for traits variants — macro instantiates func<Flt>.
template <typename Flt>
Flt atan2_max(Flt y, Flt x) {
  return dfm::atan2<Flt, dfm::MaxAccuracyTraits>(y, x);
}

// Default atan2 handles all edge cases (NaN, Inf, zero-sign).
constexpr uint32_t kAtan2Ulps = 3;
// clang-format off
static const float kAtan2SpecialY[] = {
    1.0f, -1.0f, 0.0f, 1.0f,                                           // basic quadrants
    0.0f, -0.0f, 1.0f, -1.0f,                                          // axis values
    0.0f, -0.0f, 0.0f, -0.0f,                                          // zero-zero combos
    1e6f, 1.0f, 0.001f,                                                 // varied magnitudes
    std::numeric_limits<float>::quiet_NaN(), 1.0f,                      // NaN
    std::numeric_limits<float>::quiet_NaN(),
    1.0f, -1.0f, std::numeric_limits<float>::infinity(),                // inf cases
    -std::numeric_limits<float>::infinity()};
static const float kAtan2SpecialX[] = {
    1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.0f, -0.0f,
    1.0f, 1e6f, 0.001f,
    1.0f, std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
    1.0f, 1.0f};
// clang-format on
FAST_MATH_SPECIAL_TESTS_2ARG(
    Atan2DefaultSpecial,
    ::atan2f,
    dfm::atan2,
    kAtan2SpecialY,
    kAtan2SpecialX,
    kAtan2Ulps)

// MaxAccuracy additionally handles both-inf quadrant cases.
// clang-format off
static const float kAtan2MaxAccY[] = {
    1.0f, -1.0f, 0.0f, 1.0f,                                           // basic quadrants
    0.0f, -0.0f, 1.0f, -1.0f,                                          // axis values
    0.0f, -0.0f, 0.0f, -0.0f,                                          // zero-zero combos
    std::numeric_limits<float>::quiet_NaN(), 1.0f,                      // NaN
    std::numeric_limits<float>::quiet_NaN(),
    1.0f, -1.0f,                                                        // one-inf
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity(),                              // inf-inf quadrants
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
static const float kAtan2MaxAccX[] = {
    1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.0f, -0.0f,
    1.0f, std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
    1.0f, 1.0f,
    std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
// clang-format on
FAST_MATH_SPECIAL_TESTS_2ARG(
    Atan2MaxAccSpecial,
    ::atan2f,
    atan2_max,
    kAtan2MaxAccY,
    kAtan2MaxAccX,
    kAtan2Ulps)
