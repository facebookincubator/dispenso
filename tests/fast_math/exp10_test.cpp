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

float groundTruth(float input) {
  return ::powf(10.0f, input);
}

auto exp10_accurate = dispenso::fast_math::exp10<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Exp10, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp10_accurate(0.0f), 1.0f);
  EXPECT_EQ(exp10_accurate(-kInf), 0.0f);
  EXPECT_EQ(exp10_accurate(kInf), kInf);
  EXPECT_NE(exp10_accurate(kNaN), exp10_accurate(kNaN));
}

TEST(Exp10, Range) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, exp10_accurate, -40.0f, 40.0f);
  EXPECT_LE(res, 2u);
}

TEST(Exp10LessAccurate, RangeMedium) {
  uint32_t res = dispenso::fast_math::evalAccuracy(
      groundTruth, dispenso::fast_math::exp10<float>, -37.0f, 38.0f);
  EXPECT_LE(res, 2u);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt exp10_max(Flt x) {
  return dfm::exp10<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kExp10MaxUlps = 2 + kMsvcUlpSlack;
FAST_MATH_ACCURACY_TESTS(Exp10MaxAccAll, groundTruth, exp10_max, -40.0f, 40.0f, kExp10MaxUlps)
FAST_MATH_ACCURACY_TESTS(Exp10DefaultAll, groundTruth, dfm::exp10, -37.0f, 38.0f, kExp10MaxUlps)

// Special values tested across all SIMD backends.
// MaxAccuracy handles NaN/Inf.
static const float kExp10MaxAccSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    2.0f,
    -2.0f,
    38.0f,
    -38.0f,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(
    Exp10MaxAccSpecial,
    groundTruth,
    exp10_max,
    kExp10MaxAccSpecials,
    kExp10MaxUlps)

// Default traits only works for in-range floats.
static const float kExp10DefaultSpecials[] =
    {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 10.0f, -10.0f, 0.5f, -0.5f};
FAST_MATH_SPECIAL_TESTS(
    Exp10DefaultSpecial,
    groundTruth,
    dfm::exp10,
    kExp10DefaultSpecials,
    kExp10MaxUlps)
