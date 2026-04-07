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

static float gt_tanh(float x) {
  return static_cast<float>(std::tanh(static_cast<double>(x)));
}

TEST(Tanh, SpecialValues) {
  EXPECT_EQ(dfm::tanh(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(dfm::tanh(1.0f), gt_tanh(1.0f));
  EXPECT_FLOAT_EQ(dfm::tanh(-1.0f), gt_tanh(-1.0f));
}

TEST(Tanh, Saturation) {
  // For |x| >= ~9, tanh rounds to ±1 in float.
  EXPECT_EQ(dfm::tanh(10.0f), 1.0f);
  EXPECT_EQ(dfm::tanh(-10.0f), -1.0f);
  EXPECT_EQ(dfm::tanh(100.0f), 1.0f);
  EXPECT_EQ(dfm::tanh(-100.0f), -1.0f);
}

TEST(Tanh, NearZero) {
  // tanh(x) ≈ x for small x. Verify no cancellation.
  float xs[] = {1e-7f, -1e-7f, 1e-5f, -1e-5f, 1e-3f, -1e-3f, 0.01f, -0.01f, 0.1f, -0.1f};
  for (float x : xs) {
    float expected = gt_tanh(x);
    float result = dfm::tanh(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 2u) << "tanh(" << x << "): expected=" << expected << " got=" << result;
  }
}

TEST(Tanh, RangeSmall) {
  uint32_t ulps = dfm::evalAccuracy(gt_tanh, dfm::tanh<float>, -1.0f, 1.0f);
  EXPECT_LE(ulps, 2u);
}

TEST(Tanh, RangeFull) {
  uint32_t ulps = dfm::evalAccuracy(gt_tanh, dfm::tanh<float>, -10.0f, 10.0f);
  EXPECT_LE(ulps, 2u);
}
