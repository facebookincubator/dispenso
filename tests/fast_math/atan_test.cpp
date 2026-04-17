/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <limits>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;
using namespace dispenso::fast_math::testing;

// We have reached asymptote for float32 at this value.
constexpr float kFarEnough = 20000000.0f;

TEST(Atan, SpecialVals) {
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  auto res = dfm::atan(-kFarEnough);
  EXPECT_EQ(res, -kPi_2);
  res = dfm::atan(kFarEnough);
  EXPECT_EQ(res, kPi_2);
  res = dfm::atan(0.0f);
  EXPECT_EQ(res, 0.0f);
  // NaN and Inf
  EXPECT_NE(dfm::atan(kNaN), dfm::atan(kNaN));
  EXPECT_EQ(dfm::atan(kInf), kPi_2);
  EXPECT_EQ(dfm::atan(-kInf), -kPi_2);
}

TEST(Atan, Range) {
  uint32_t ulps = dfm::evalAccuracy(atanf, dfm::atan<float>, -kFarEnough, kFarEnough);
  EXPECT_LE(ulps, 3);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kAtanMaxUlps = 3;
FAST_MATH_ACCURACY_TESTS(AtanAll, ::atanf, dfm::atan, -kFarEnough, kFarEnough, kAtanMaxUlps)

// Special values tested across all SIMD backends.
static const float kAtanSpecials[] = {
    0.0f,
    -0.0f,
    -kFarEnough,
    kFarEnough,
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    2.0f,
    -2.0f,
    10.0f,
    -10.0f,
    100.0f,
    1e10f,
    -1e10f,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(AtanSpecial, ::atanf, dfm::atan, kAtanSpecials, kAtanMaxUlps)
