/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;

// Ground truth: double-precision pow, rounded to float.
static float gt_pow(float x, float y) {
  return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

// --- Accuracy tests ---

TEST(Pow, SpecialValues) {
  auto pow_f = dfm::pow<float>;
  EXPECT_EQ(pow_f(2.0f, 10.0f), 1024.0f);
  EXPECT_EQ(pow_f(0.5f, 2.0f), 0.25f);
  EXPECT_EQ(pow_f(4.0f, 0.5f), 2.0f);
  EXPECT_NEAR(pow_f(8.0f, 1.0f / 3.0f), 2.0f, 1e-6f);
  EXPECT_NEAR(pow_f(10.0f, -1.0f), 0.1f, 1e-7f);
}

TEST(Pow, ExactIntegerPowers) {
  // pow(2, n) for small n should be exact or near-exact.
  for (int32_t n = 1; n <= 23; ++n) {
    float expected = static_cast<float>(1 << n);
    float result = dfm::pow(2.0f, static_cast<float>(n));
    uint32_t dist = dfm::float_distance(expected, result);
    EXPECT_LE(dist, 1u) << "pow(2, " << n << "): expected=" << expected << " got=" << result;
  }
  // pow(3, n) for small n.
  float p3 = 1.0f;
  for (int32_t n = 1; n <= 14; ++n) {
    p3 *= 3.0f;
    float result = dfm::pow(3.0f, static_cast<float>(n));
    uint32_t dist = dfm::float_distance(p3, result);
    EXPECT_LE(dist, 2u) << "pow(3, " << n << "): expected=" << p3 << " got=" << result;
  }
}

TEST(Pow, RandomModerate) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> x_dist(0.01f, 100.0f);
  std::uniform_real_distribution<float> y_dist(-8.0f, 8.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::pow(x, y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  // Default path now uses double-precision core: <1 ULP for float output.
  EXPECT_LE(maxUlps, 4u) << "RandomModerate max ULP";
}

TEST(Pow, RandomWide) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> x_dist(1e-10f, 1e10f);
  std::uniform_real_distribution<float> y_dist(-4.0f, 4.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::pow(x, y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  EXPECT_LE(maxUlps, 4u) << "RandomWide max ULP";
}

TEST(Pow, RandomSmallExp) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> x_dist(0.5f, 2.0f);
  std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::pow(x, y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  EXPECT_LE(maxUlps, 3u) << "RandomSmallExp max ULP";
}

// --- Negative base tests ---

TEST(Pow, NegativeBaseIntegerExp) {
  // pow(neg, even) → positive, pow(neg, odd) → negative.
  float bases[] = {-0.5f, -1.0f, -2.0f, -100.0f};
  float exps[] = {2.0f, 3.0f, 4.0f, 5.0f, -1.0f, -2.0f, -3.0f};

  for (float x : bases) {
    for (float y : exps) {
      float expected = gt_pow(x, y);
      float result = dfm::pow<float, dfm::MaxAccuracyTraits>(x, y);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(result)) << "pow(" << x << ", " << y << ")";
      } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(result)) << "pow(" << x << ", " << y << ")";
        EXPECT_EQ(std::signbit(expected), std::signbit(result))
            << "pow(" << x << ", " << y << ") sign mismatch";
      } else {
        uint32_t dist = dfm::float_distance(expected, result);
        EXPECT_LE(dist, 3u) << "pow(" << x << ", " << y << "): expected=" << expected
                            << " got=" << result;
        // Check sign correctness.
        EXPECT_EQ(std::signbit(expected), std::signbit(result))
            << "pow(" << x << ", " << y << ") sign mismatch";
      }
    }
  }
}

TEST(Pow, NegativeBaseNonIntegerExp) {
  // pow(negative, non-integer) → NaN.
  float bases[] = {-1.0f, -2.0f, -0.5f};
  float exps[] = {0.5f, 1.5f, -0.5f, 2.7f};

  for (float x : bases) {
    for (float y : exps) {
      float result = dfm::pow<float, dfm::MaxAccuracyTraits>(x, y);
      EXPECT_TRUE(std::isnan(result))
          << "pow(" << x << ", " << y << ") should be NaN, got " << result;
    }
  }
}

TEST(Pow, NegativeBaseRandomInteger) {
  std::mt19937 rng(999);
  std::uniform_real_distribution<float> x_dist(-100.0f, -0.01f);

  uint32_t maxUlps = 0;
  for (int32_t y_int = -50; y_int <= 50; ++y_int) {
    float y = static_cast<float>(y_int);
    for (int32_t i = 0; i < 1024; ++i) {
      float x = x_dist(rng);
      float expected = gt_pow(x, y);
      float result = dfm::pow<float, dfm::MaxAccuracyTraits>(x, y);
      if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
        continue;
      }
      uint32_t dist = dfm::float_distance(expected, result);
      maxUlps = std::max(maxUlps, dist);
    }
  }
  // MaxAccuracy with negative base goes through exp2(y * log2(|x|)) + sign XOR.
  // For y up to 50 and large |x|, error can reach ~10 ULP.
  EXPECT_LE(maxUlps, 16u) << "NegativeBaseRandomInteger max ULP";
}

// --- IEEE 754 boundary conditions (MaxAccuracyTraits) ---

TEST(PowBounds, YZero) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  // pow(x, 0) = 1 for all x.
  float xs[] = {0.0f, -0.0f, 1.0f, -1.0f, kInf, -kInf, kNaN, 42.0f, -42.0f};
  for (float x : xs) {
    EXPECT_EQ(pow_b(x, 0.0f), 1.0f) << "pow(" << x << ", 0) should be 1";
  }
}

TEST(PowBounds, XOne) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  // pow(1, y) = 1 for all y, even NaN.
  float ys[] = {0.0f, 1.0f, -1.0f, kInf, -kInf, kNaN, 42.0f};
  for (float y : ys) {
    EXPECT_EQ(pow_b(1.0f, y), 1.0f) << "pow(1, " << y << ") should be 1";
  }
}

TEST(PowBounds, NegOneInf) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(pow_b(-1.0f, kInf), 1.0f) << "pow(-1, +inf) should be 1";
  EXPECT_EQ(pow_b(-1.0f, -kInf), 1.0f) << "pow(-1, -inf) should be 1";
}

TEST(PowBounds, ZeroPosExp) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;

  EXPECT_EQ(pow_b(0.0f, 1.0f), 0.0f);
  EXPECT_FALSE(std::signbit(pow_b(0.0f, 1.0f)));
  EXPECT_EQ(pow_b(0.0f, 2.0f), 0.0f);
  EXPECT_EQ(pow_b(0.0f, 0.5f), 0.0f);

  EXPECT_EQ(pow_b(-0.0f, 1.0f), -0.0f);
  EXPECT_TRUE(std::signbit(pow_b(-0.0f, 1.0f)));
  EXPECT_EQ(pow_b(-0.0f, 3.0f), -0.0f);
  EXPECT_TRUE(std::signbit(pow_b(-0.0f, 3.0f)));
  EXPECT_EQ(pow_b(-0.0f, 2.0f), 0.0f);
  EXPECT_FALSE(std::signbit(pow_b(-0.0f, 2.0f)));
  EXPECT_EQ(pow_b(-0.0f, 0.5f), 0.0f);
}

TEST(PowBounds, ZeroNegExp) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(pow_b(0.0f, -1.0f), kInf);
  EXPECT_EQ(pow_b(0.0f, -2.0f), kInf);
  EXPECT_EQ(pow_b(-0.0f, -1.0f), -kInf);
  EXPECT_EQ(pow_b(-0.0f, -3.0f), -kInf);
  EXPECT_EQ(pow_b(-0.0f, -2.0f), kInf);
}

TEST(PowBounds, InfPosExp) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(pow_b(kInf, 1.0f), kInf);
  EXPECT_EQ(pow_b(kInf, 2.0f), kInf);
  EXPECT_EQ(pow_b(-kInf, 1.0f), -kInf);
  EXPECT_EQ(pow_b(-kInf, 3.0f), -kInf);
  EXPECT_EQ(pow_b(-kInf, 2.0f), kInf);
}

TEST(PowBounds, InfNegExp) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(pow_b(kInf, -1.0f), 0.0f);
  EXPECT_FALSE(std::signbit(pow_b(kInf, -1.0f)));
  EXPECT_EQ(pow_b(kInf, -2.0f), 0.0f);

  EXPECT_EQ(pow_b(-kInf, -1.0f), -0.0f);
  EXPECT_TRUE(std::signbit(pow_b(-kInf, -1.0f)));
  EXPECT_EQ(pow_b(-kInf, -3.0f), -0.0f);
  EXPECT_TRUE(std::signbit(pow_b(-kInf, -3.0f)));
  EXPECT_EQ(pow_b(-kInf, -2.0f), 0.0f);
  EXPECT_FALSE(std::signbit(pow_b(-kInf, -2.0f)));
}

TEST(PowBounds, AbsXInf) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(pow_b(0.5f, -kInf), kInf);
  EXPECT_EQ(pow_b(2.0f, -kInf), 0.0f);
  EXPECT_EQ(pow_b(0.5f, kInf), 0.0f);
  EXPECT_EQ(pow_b(2.0f, kInf), kInf);
}

TEST(PowBounds, NaNPropagation) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_TRUE(std::isnan(pow_b(kNaN, 2.0f)));
  EXPECT_TRUE(std::isnan(pow_b(2.0f, kNaN)));
  EXPECT_TRUE(std::isnan(pow_b(kNaN, kNaN)));
  EXPECT_TRUE(std::isnan(pow_b(-2.0f, 1.5f)));
}

TEST(PowBounds, Subnormal) {
  auto pow_b = dfm::pow<float, dfm::MaxAccuracyTraits>;
  // Subnormal bases: fp32 subnormals are in [1.4e-45, 1.2e-38].
  float subnormals[] = {1.0e-40f, 1.0e-42f, 1.0e-44f, std::numeric_limits<float>::denorm_min()};
  float exps[] = {2.0f, 0.5f, -1.0f, 3.0f};

  for (float x : subnormals) {
    for (float y : exps) {
      float expected = gt_pow(x, y);
      float result = pow_b(x, y);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(result)) << "pow(" << x << ", " << y << ")";
      } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(result))
            << "pow(" << x << ", " << y << ") expected inf, got " << result;
      } else if (expected == 0.0f) {
        EXPECT_EQ(result, 0.0f) << "pow(" << x << ", " << y << ") expected 0, got " << result;
      } else {
        uint32_t dist = dfm::float_distance(expected, result);
        EXPECT_LE(dist, 4u) << "pow(" << x << ", " << y << "): expected=" << expected
                            << " got=" << result;
      }
    }
  }
}

// --- MaxAccuracy random tests ---

TEST(PowAccurate, RandomModerate) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> x_dist(0.01f, 100.0f);
  std::uniform_real_distribution<float> y_dist(-8.0f, 8.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::pow<float, dfm::MaxAccuracyTraits>(x, y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  EXPECT_LE(maxUlps, 3u) << "MaxAccuracy RandomModerate max ULP";
}

TEST(PowDouble, RandomModerate) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> x_dist(0.01f, 100.0f);
  std::uniform_real_distribution<float> y_dist(-8.0f, 8.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::detail::pow_double_core(std::fabs(x), y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  EXPECT_LE(maxUlps, 4u) << "PowDouble RandomModerate max ULP";
}

TEST(PowDouble, RandomWide) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> x_dist(1e-10f, 1e10f);
  std::uniform_real_distribution<float> y_dist(-4.0f, 4.0f);

  uint32_t maxUlps = 0;
  constexpr int32_t kN = 1 << 20;
  for (int32_t i = 0; i < kN; ++i) {
    float x = x_dist(rng);
    float y = y_dist(rng);
    float expected = gt_pow(x, y);
    float result = dfm::detail::pow_double_core(std::fabs(x), y);
    if (std::isnan(expected) || std::isinf(expected) || expected == 0.0f) {
      continue;
    }
    uint32_t dist = dfm::float_distance(expected, result);
    maxUlps = std::max(maxUlps, dist);
  }
  EXPECT_LE(maxUlps, 4u) << "PowDouble RandomWide max ULP";
}
