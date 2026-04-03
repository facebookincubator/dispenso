/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <cmath>
#include <limits>
#include <random>

#include "eval.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;

// Ground truth: double-precision sqrt(x*x + y*y) cast to float.
static float hypotRef(float x, float y) {
  double xd = static_cast<double>(x);
  double yd = static_cast<double>(y);
  return static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
}

TEST(Hypot, SpecialValues) {
  EXPECT_EQ(dfm::hypot(0.0f, 0.0f), 0.0f);
  EXPECT_EQ(dfm::hypot(3.0f, 0.0f), 3.0f);
  EXPECT_EQ(dfm::hypot(0.0f, 4.0f), 4.0f);
  EXPECT_EQ(dfm::hypot(-3.0f, 0.0f), 3.0f);
  EXPECT_EQ(dfm::hypot(0.0f, -4.0f), 4.0f);
  EXPECT_FLOAT_EQ(dfm::hypot(3.0f, 4.0f), 5.0f);
  EXPECT_FLOAT_EQ(dfm::hypot(1.0f, 1.0f), std::sqrt(2.0f));
}

TEST(Hypot, GridNearZero) {
  constexpr int32_t kMax = 2048;

  float quantizer = 0.5f;
  quantizer *= quantizer;
  quantizer *= quantizer;
  quantizer *= quantizer;
  quantizer *= 0.25f;

  uint32_t maxUlps = 0;
  for (int32_t yi = -kMax; yi <= kMax; ++yi) {
    for (int32_t xi = -kMax; xi <= kMax; ++xi) {
      float x = static_cast<float>(xi) * quantizer;
      float y = static_cast<float>(yi) * quantizer;
      maxUlps = std::max(maxUlps, dfm::float_distance(hypotRef(x, y), dfm::hypot(x, y)));
    }
  }
  EXPECT_LE(maxUlps, 1u);
}

TEST(Hypot, RandomNormal) {
  constexpr size_t kNumSamples = 1 << 24;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1e5f, 1e5f);

  uint32_t maxUlps = 0;
  for (size_t i = 0; i < kNumSamples; ++i) {
    float x = dis(gen);
    float y = dis(gen);
    maxUlps = std::max(maxUlps, dfm::float_distance(hypotRef(x, y), dfm::hypot(x, y)));
  }
  EXPECT_LE(maxUlps, 1u);
}

TEST(Hypot, RandomWide) {
  constexpr size_t kNumSamples = 1 << 24;
  std::mt19937 gen(77);
  std::uniform_real_distribution<float> dis(-1e30f, 1e30f);

  uint32_t maxUlps = 0;
  for (size_t i = 0; i < kNumSamples; ++i) {
    float x = dis(gen);
    float y = dis(gen);
    maxUlps = std::max(maxUlps, dfm::float_distance(hypotRef(x, y), dfm::hypot(x, y)));
  }
  EXPECT_LE(maxUlps, 1u);
}

TEST(Hypot, RandomTiny) {
  constexpr size_t kNumSamples = 1 << 24;
  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dis(-1e-30f, 1e-30f);

  uint32_t maxUlps = 0;
  for (size_t i = 0; i < kNumSamples; ++i) {
    float x = dis(gen);
    float y = dis(gen);
    maxUlps = std::max(maxUlps, dfm::float_distance(hypotRef(x, y), dfm::hypot(x, y)));
  }
  EXPECT_LE(maxUlps, 1u);
}

TEST(Hypot, Symmetry) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1e10f, 1e10f);

  for (size_t i = 0; i < 100000; ++i) {
    float x = dis(gen);
    float y = dis(gen);
    EXPECT_EQ(dfm::hypot(x, y), dfm::hypot(y, x));
    EXPECT_EQ(dfm::hypot(x, y), dfm::hypot(-x, y));
    EXPECT_EQ(dfm::hypot(x, y), dfm::hypot(x, -y));
    EXPECT_EQ(dfm::hypot(x, y), dfm::hypot(-x, -y));
  }
}

TEST(Hypot, DiagonalSweep) {
  uint32_t maxUlps = 0;
  for (float x = 1e-20f; x < 1e20f; x *= 1.5f) {
    maxUlps = std::max(maxUlps, dfm::float_distance(hypotRef(x, x), dfm::hypot(x, x)));
  }
  EXPECT_LE(maxUlps, 1u);
}

// --- hypot with MaxAccuracyTraits (IEEE boundary conditions) ---

template <typename... Args>
static float hypotBounds(Args... args) {
  return dfm::hypot<float, dfm::MaxAccuracyTraits>(args...);
}

TEST(HypotBounds, InfFinite) {
  float inf = std::numeric_limits<float>::infinity();
  // hypot(±inf, finite) = +inf
  EXPECT_EQ(hypotBounds(inf, 3.0f), inf);
  EXPECT_EQ(hypotBounds(-inf, 3.0f), inf);
  EXPECT_EQ(hypotBounds(3.0f, inf), inf);
  EXPECT_EQ(hypotBounds(3.0f, -inf), inf);
  EXPECT_EQ(hypotBounds(inf, 0.0f), inf);
  EXPECT_EQ(hypotBounds(0.0f, -inf), inf);
}

TEST(HypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  // IEEE 754: hypot(±inf, NaN) = +inf (inf wins over NaN)
  float r1 = hypotBounds(inf, nan);
  float r2 = hypotBounds(-inf, nan);
  float r3 = hypotBounds(nan, inf);
  float r4 = hypotBounds(nan, -inf);
  EXPECT_TRUE(std::isinf(r1) && r1 > 0) << "hypot(inf, nan) = " << r1;
  EXPECT_TRUE(std::isinf(r2) && r2 > 0) << "hypot(-inf, nan) = " << r2;
  EXPECT_TRUE(std::isinf(r3) && r3 > 0) << "hypot(nan, inf) = " << r3;
  EXPECT_TRUE(std::isinf(r4) && r4 > 0) << "hypot(nan, -inf) = " << r4;
}

TEST(HypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  // hypot(NaN, finite) = NaN (when neither is inf)
  EXPECT_TRUE(std::isnan(hypotBounds(nan, 3.0f)));
  EXPECT_TRUE(std::isnan(hypotBounds(3.0f, nan)));
  EXPECT_TRUE(std::isnan(hypotBounds(nan, nan)));
}
