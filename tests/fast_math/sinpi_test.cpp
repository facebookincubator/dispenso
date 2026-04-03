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

// Ground truth for sinpi/cospi: use double-precision sinpi where available,
// otherwise compute via exact range reduction in double + sin/cos.
static float gt_sinpi(float x) {
  double xd = static_cast<double>(x);
  // Exact range reduction in double: much more precise than sin(pi * x).
  double j = std::round(2.0 * xd);
  double r = xd - j * 0.5;
  double t = r * M_PI;
  int qi = static_cast<int>(j) & 3;
  double sv = std::sin(t);
  double cv = std::cos(t);
  double result;
  switch (qi) {
    case 0:
      result = sv;
      break;
    case 1:
      result = cv;
      break;
    case 2:
      result = -sv;
      break;
    case 3:
      result = -cv;
      break;
    default:
      result = 0.0;
      break;
  }
  return static_cast<float>(result);
}

static float gt_cospi(float x) {
  double xd = static_cast<double>(x);
  double j = std::round(2.0 * xd);
  double r = xd - j * 0.5;
  double t = r * M_PI;
  int qi = (static_cast<int>(j) + 1) & 3;
  double sv = std::sin(t);
  double cv = std::cos(t);
  double result;
  switch (qi) {
    case 0:
      result = sv;
      break;
    case 1:
      result = cv;
      break;
    case 2:
      result = -sv;
      break;
    case 3:
      result = -cv;
      break;
    default:
      result = 0.0;
      break;
  }
  return static_cast<float>(result);
}

// --- sinpi ---

TEST(SinPi, ExactValues) {
  // sinpi(n) = 0 for all integers
  EXPECT_EQ(dfm::sinpi(0.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(1.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(-1.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(2.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(100.0f), 0.0f);

  // sinpi(0.5) = 1, sinpi(-0.5) = -1
  EXPECT_FLOAT_EQ(dfm::sinpi(0.5f), 1.0f);
  EXPECT_FLOAT_EQ(dfm::sinpi(-0.5f), -1.0f);
  EXPECT_FLOAT_EQ(dfm::sinpi(1.5f), -1.0f);
  EXPECT_FLOAT_EQ(dfm::sinpi(-1.5f), 1.0f);
}

TEST(SinPi, AccuracyVsLibc) {
  constexpr int kSteps = 4096;
  for (int i = 0; i < kSteps; ++i) {
    float x = -10.0f + 20.0f * static_cast<float>(i) / kSteps;
    float result = dfm::sinpi(x);
    float expected = gt_sinpi(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 2u) << "sinpi(" << x << ") = " << result << " expected " << expected;
  }
}

// --- cospi ---

TEST(CosPi, ExactValues) {
  // cospi(n) = ±1 for integers
  EXPECT_FLOAT_EQ(dfm::cospi(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(dfm::cospi(1.0f), -1.0f);
  EXPECT_FLOAT_EQ(dfm::cospi(-1.0f), -1.0f);
  EXPECT_FLOAT_EQ(dfm::cospi(2.0f), 1.0f);

  // cospi(0.5) = 0
  EXPECT_NEAR(dfm::cospi(0.5f), 0.0f, 1e-6f);
  EXPECT_NEAR(dfm::cospi(-0.5f), 0.0f, 1e-6f);
  EXPECT_NEAR(dfm::cospi(1.5f), 0.0f, 1e-6f);
}

TEST(CosPi, AccuracyVsLibc) {
  constexpr int kSteps = 4096;
  for (int i = 0; i < kSteps; ++i) {
    float x = -10.0f + 20.0f * static_cast<float>(i) / kSteps;
    float result = dfm::cospi(x);
    float expected = gt_cospi(x);
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 2u) << "cospi(" << x << ") = " << result << " expected " << expected;
  }
}

// --- sincospi ---

TEST(SinCosPi, MatchesSeparateCalls) {
  constexpr int kSteps = 1024;
  for (int i = 0; i < kSteps; ++i) {
    float x = -10.0f + 20.0f * static_cast<float>(i) / kSteps;
    float s, c;
    dfm::sincospi(x, &s, &c);
    EXPECT_EQ(s, dfm::sinpi(x)) << "sin mismatch at x=" << x;
    EXPECT_EQ(c, dfm::cospi(x)) << "cos mismatch at x=" << x;
  }
}

TEST(SinCosPi, AccuracyVsLibc) {
  constexpr int kSteps = 4096;
  for (int i = 0; i < kSteps; ++i) {
    float x = -10.0f + 20.0f * static_cast<float>(i) / kSteps;
    float s, c;
    dfm::sincospi(x, &s, &c);
    float gt_s = gt_sinpi(x);
    float gt_c = gt_cospi(x);
    EXPECT_LE(dfm::float_distance(gt_s, s), 2u) << "sinpi at x=" << x;
    EXPECT_LE(dfm::float_distance(gt_c, c), 2u) << "cospi at x=" << x;
  }
}

TEST(SinPi, LargeArguments) {
  // For |x| >= 2^23, every float is an integer, sinpi = 0.
  EXPECT_EQ(dfm::sinpi(8388608.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(16777216.0f), 0.0f);
  EXPECT_EQ(dfm::sinpi(-8388608.0f), 0.0f);
}

TEST(CosPi, LargeArguments) {
  // For |x| >= 2^23, every float is an integer, cospi = ±1.
  float c = dfm::cospi(8388608.0f);
  EXPECT_TRUE(c == 1.0f || c == -1.0f);
  c = dfm::cospi(16777216.0f);
  EXPECT_TRUE(c == 1.0f || c == -1.0f);
}
