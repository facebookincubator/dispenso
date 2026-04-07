/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;

static float gt_expm1(float x) {
  return static_cast<float>(std::expm1(static_cast<double>(x)));
}

TEST(Expm1, SpecialValues) {
  EXPECT_EQ(dfm::expm1(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(dfm::expm1(1.0f), gt_expm1(1.0f));
  EXPECT_FLOAT_EQ(dfm::expm1(-1.0f), gt_expm1(-1.0f));
}

TEST(Expm1, NearZero) {
  // The primary use case: x very close to 0 where exp(x)-1 cancels.
  float xs[] = {1e-7f, -1e-7f, 1e-5f, -1e-5f, 1e-3f, -1e-3f, 0.01f, -0.01f, 0.1f, -0.1f};
  for (float x : xs) {
    float expected = gt_expm1(x);
    float result = dfm::expm1(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 2u) << "expm1(" << x << "): expected=" << expected << " got=" << result;
  }
}

TEST(Expm1, RangeSmall) {
  uint32_t ulps = dfm::evalAccuracy(gt_expm1, dfm::expm1<float>, -0.5f, 0.5f);
  EXPECT_LE(ulps, 2u);
}

TEST(Expm1, RangeMedium) {
  uint32_t ulps = dfm::evalAccuracy(gt_expm1, dfm::expm1<float>, -10.0f, 10.0f);
  EXPECT_LE(ulps, 2u);
}

TEST(Expm1, RangeLarge) {
  uint32_t ulps = dfm::evalAccuracy(gt_expm1, dfm::expm1<float>, -88.0f, 88.0f);
  EXPECT_LE(ulps, 2u);
}
