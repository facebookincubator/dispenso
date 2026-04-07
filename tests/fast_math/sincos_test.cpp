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

TEST(SinCos, BasicValues) {
  float s, c;
  dfm::sincos(0.0f, &s, &c);
  EXPECT_FLOAT_EQ(s, 0.0f);
  EXPECT_FLOAT_EQ(c, 1.0f);

  dfm::sincos(kPi_2, &s, &c);
  EXPECT_NEAR(s, 1.0f, 1e-6f);
  EXPECT_NEAR(c, 0.0f, 1e-6f);

  dfm::sincos(kPi, &s, &c);
  EXPECT_NEAR(s, 0.0f, 1e-6f);
  EXPECT_NEAR(c, -1.0f, 1e-6f);

  dfm::sincos(-kPi_2, &s, &c);
  EXPECT_NEAR(s, -1.0f, 1e-6f);
  EXPECT_NEAR(c, 0.0f, 1e-6f);
}

TEST(SinCos, MatchesSeparateCalls) {
  constexpr int kSteps = 1024;
  constexpr float kRange = 128.0f * kPi;
  constexpr float kDelta = 2.0f * kRange / kSteps;

  for (int i = 0; i < kSteps; ++i) {
    float x = -kRange + static_cast<float>(i) * kDelta;
    float s, c;
    dfm::sincos(x, &s, &c);
    float expected_s = dfm::sin(x);
    float expected_c = dfm::cos(x);
    EXPECT_EQ(s, expected_s) << "sin mismatch at x=" << x;
    EXPECT_EQ(c, expected_c) << "cos mismatch at x=" << x;
  }
}

TEST(SinCos, AccuracyVsLibc) {
  constexpr int kSteps = 4096;
  constexpr float kRange = 128.0f * kPi;
  constexpr float kDelta = 2.0f * kRange / kSteps;

  for (int i = 0; i < kSteps; ++i) {
    float x = -kRange + static_cast<float>(i) * kDelta;
    float s, c;
    dfm::sincos(x, &s, &c);

    float gt_s = static_cast<float>(::sin(static_cast<double>(x)));
    float gt_c = static_cast<float>(::cos(static_cast<double>(x)));

    uint32_t sin_dist = dfm::float_distance(gt_s, s);
    uint32_t cos_dist = dfm::float_distance(gt_c, c);

    EXPECT_LE(sin_dist, 2u) << "sin at x=" << x << " expected=" << gt_s << " actual=" << s;
    EXPECT_LE(cos_dist, 2u) << "cos at x=" << x << " expected=" << gt_c << " actual=" << c;
  }
}

TEST(SinCos, MaxAccuracy) {
  constexpr int kSteps = 4096;
  constexpr float kRange = 128.0f * kPi;
  constexpr float kDelta = 2.0f * kRange / kSteps;

  for (int i = 0; i < kSteps; ++i) {
    float x = -kRange + static_cast<float>(i) * kDelta;
    float s, c;
    dfm::sincos<float, dfm::MaxAccuracyTraits>(x, &s, &c);

    float gt_s = static_cast<float>(::sin(static_cast<double>(x)));
    float gt_c = static_cast<float>(::cos(static_cast<double>(x)));

    EXPECT_LE(dfm::float_distance(gt_s, s), 2u) << "sin at x=" << x;
    EXPECT_LE(dfm::float_distance(gt_c, c), 2u) << "cos at x=" << x;
  }
}
