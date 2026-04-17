/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;
using namespace dispenso::fast_math::testing;

static float gt_log1p(float x) {
  return static_cast<float>(std::log1p(static_cast<double>(x)));
}

TEST(Log1p, SpecialValues) {
  EXPECT_EQ(dfm::log1p(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(dfm::log1p(1.0f), gt_log1p(1.0f));
  EXPECT_FLOAT_EQ(dfm::log1p(-0.5f), gt_log1p(-0.5f));
}

TEST(Log1p, NearZero) {
  // The primary use case: x very close to 0 where log(1+x) cancels.
  float xs[] = {1e-7f, -1e-7f, 1e-5f, -1e-5f, 1e-3f, -1e-3f, 0.01f, -0.01f, 0.1f, -0.1f};
  for (float x : xs) {
    float expected = gt_log1p(x);
    float result = dfm::log1p(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 2u) << "log1p(" << x << "): expected=" << expected << " got=" << result;
  }
}

TEST(Log1p, RangeSmall) {
  uint32_t ulps = dfm::evalAccuracy(gt_log1p, dfm::log1p<float>, -0.5f, 0.5f);
  EXPECT_LE(ulps, 2u);
}

TEST(Log1p, RangeMedium) {
  uint32_t ulps = dfm::evalAccuracy(gt_log1p, dfm::log1p<float>, -0.99f, 100.0f);
  EXPECT_LE(ulps, 2u);
}

TEST(Log1p, RangeLarge) {
  uint32_t ulps = dfm::evalAccuracy(gt_log1p, dfm::log1p<float>, -0.99f, 1e10f);
  EXPECT_LE(ulps, 2u);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kLog1pMaxUlps = 2;
FAST_MATH_ACCURACY_TESTS(Log1pAll, gt_log1p, dfm::log1p, -0.99f, 1e10f, kLog1pMaxUlps)

// Special values tested across all SIMD backends.
// log1p doesn't handle Inf — only test in-domain values.
static const float kLog1pSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -0.5f,
    0.5f,
    10.0f,
    100.0f,
    0.001f,
    -0.001f,
    1e-4f,
    -1e-4f,
    1e-7f,
    -1e-7f,
    0.1f,
    -0.1f,
    1e-6f,
    1e-3f};
FAST_MATH_SPECIAL_TESTS(Log1pSpecial, gt_log1p, dfm::log1p, kLog1pSpecials, kLog1pMaxUlps)
