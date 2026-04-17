/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

using namespace dispenso::fast_math::testing;

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
  EXPECT_LE(ulps, 3);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
namespace dfm = dispenso::fast_math;
constexpr uint32_t kAsinMaxUlps = 3;
FAST_MATH_ACCURACY_TESTS(AsinAll, ::asinf, dfm::asin, -1.0f, 1.0f, kAsinMaxUlps)

// Special values tested across all SIMD backends (including out-of-range → NaN).
static const float kAsinSpecials[] = {
    0.0f,
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    0.25f,
    -0.25f,
    0.9f,
    0.99999f,
    1.00001f,
    -1.00001f,
    2.0f,
    -5.0f,
    10.0f,
    -10.0f,
    100.0f,
    -100.0f};
FAST_MATH_SPECIAL_TESTS(AsinSpecial, ::asinf, dfm::asin, kAsinSpecials, kAsinMaxUlps)
