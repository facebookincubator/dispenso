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

auto log10_accurate = dispenso::fast_math::log10<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Log10, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log10_accurate(0.0f), -kInf);
  EXPECT_NE(log10_accurate(-1.0f), log10_accurate(-1.0f));
  EXPECT_EQ(log10_accurate(kInf), kInf);
  EXPECT_NE(log10_accurate(kNaN), log10_accurate(kNaN));
}

TEST(Log10Accurate, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::log10f, log10_accurate, 0.0f, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10Accurate, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::log10f, log10_accurate, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10, RangeNeg) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::log10f, dispenso::fast_math::log10<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, 3u);
}

TEST(Log10, RangePos) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::log10f, dispenso::fast_math::log10<float>, 1.0f);
  EXPECT_LE(res, 3u);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt log10_max(Flt x) {
  return dfm::log10<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kLog10MaxUlps = 3;
FAST_MATH_ACCURACY_TESTS(
    Log10MaxAccAll,
    ::log10f,
    log10_max,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLog10MaxUlps)
FAST_MATH_ACCURACY_TESTS(
    Log10DefaultAll,
    ::log10f,
    dfm::log10,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLog10MaxUlps)

// Special values tested across all SIMD backends.
// MaxAccuracy handles out-of-domain inputs (0, negatives, NaN, Inf).
static const float kLog10MaxAccSpecials[] = {
    1.0f,
    0.5f,
    10.0f,
    100.0f,
    1000.0f,
    0.1f,
    0.01f,
    0.0f,
    -0.0f,
    -1.0f,
    -100.0f,
    std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(
    Log10MaxAccSpecial,
    ::log10f,
    log10_max,
    kLog10MaxAccSpecials,
    kLog10MaxUlps)

// Default traits only works for positive normal floats.
static const float kLog10DefaultSpecials[] =
    {1.0f, 0.5f, 10.0f, 100.0f, 1000.0f, 0.1f, 0.01f, 1e-10f, 10000.0f};
FAST_MATH_SPECIAL_TESTS(
    Log10DefaultSpecial,
    ::log10f,
    dfm::log10,
    kLog10DefaultSpecials,
    kLog10MaxUlps)
