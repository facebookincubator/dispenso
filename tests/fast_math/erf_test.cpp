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

static float gt_erf(float x) {
  return static_cast<float>(std::erf(static_cast<double>(x)));
}

constexpr uint32_t kErfMaxUlps = 2;

TEST(Erf, SpecialValues) {
  EXPECT_EQ(dfm::erf(0.0f), 0.0f);
  EXPECT_EQ(dfm::erf(-0.0f), -0.0f);
  EXPECT_TRUE(std::signbit(dfm::erf(-0.0f)));
  EXPECT_FLOAT_EQ(dfm::erf(1.0f), gt_erf(1.0f));
  EXPECT_FLOAT_EQ(dfm::erf(-1.0f), gt_erf(-1.0f));
  EXPECT_EQ(dfm::erf(std::numeric_limits<float>::infinity()), 1.0f);
  EXPECT_EQ(dfm::erf(-std::numeric_limits<float>::infinity()), -1.0f);
}

TEST(Erf, NaN) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(dfm::erf(nan)));
}

TEST(Erf, Saturation) {
  // For |x| >= 3.92, erf rounds to ±1 in float.
  EXPECT_EQ(dfm::erf(4.0f), 1.0f);
  EXPECT_EQ(dfm::erf(-4.0f), -1.0f);
  EXPECT_EQ(dfm::erf(10.0f), 1.0f);
  EXPECT_EQ(dfm::erf(-10.0f), -1.0f);
  EXPECT_EQ(dfm::erf(100.0f), 1.0f);
  EXPECT_EQ(dfm::erf(-100.0f), -1.0f);
}

TEST(Erf, NearZero) {
  // erf(x) ≈ (2/√π)·x for small x. Verify no cancellation.
  float xs[] = {1e-7f, -1e-7f, 1e-5f, -1e-5f, 1e-3f, -1e-3f, 0.01f, -0.01f, 0.1f, -0.1f};
  for (float x : xs) {
    float expected = gt_erf(x);
    float result = dfm::erf(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, kErfMaxUlps) << "erf(" << x << "): expected=" << expected << " got=" << result;
  }
}

// Exhaustive accuracy tests — scalar + all available SIMD backends, same threshold.
FAST_MATH_ACCURACY_TESTS(Erf, gt_erf, dfm::erf, -4.0f, 4.0f, kErfMaxUlps)
