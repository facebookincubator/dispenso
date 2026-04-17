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

auto log_accurate = dispenso::fast_math::log<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Log, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log_accurate(0.0f), -kInf);
  EXPECT_NE(log_accurate(-1.0f), log_accurate(-1.0f));
  EXPECT_EQ(log_accurate(kInf), kInf);
  EXPECT_NE(log_accurate(kNaN), log_accurate(kNaN));
}

TEST(LogAccurate, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, log_accurate, 0.0f, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(LogAccurate, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, log_accurate, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(Log, RangeNeg) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::logf, dispenso::fast_math::log<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, 2u);
}

TEST(Log, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::logf, dispenso::fast_math::log<float>, 1.0f);
  EXPECT_LE(res, 2u);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt log_max(Flt x) {
  return dfm::log<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kLogMaxUlps = 2;
FAST_MATH_ACCURACY_TESTS(
    LogMaxAccAll,
    ::logf,
    log_max,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLogMaxUlps)
FAST_MATH_ACCURACY_TESTS(
    LogDefaultAll,
    ::logf,
    dfm::log,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLogMaxUlps)

// Special values tested across all SIMD backends.
// MaxAccuracy handles out-of-domain inputs (0, negatives, NaN, Inf).
static const float kLogMaxAccSpecials[] = {
    1.0f,
    0.5f,
    2.0f,
    10.0f,
    100.0f,
    0.01f,
    1e-6f,
    1e10f,
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
FAST_MATH_SPECIAL_TESTS(LogMaxAccSpecial, ::logf, log_max, kLogMaxAccSpecials, kLogMaxUlps)

// Default traits only works for positive normal floats.
static const float kLogDefaultSpecials[] =
    {1.0f, 0.5f, 2.0f, 10.0f, 100.0f, 0.01f, 1000.0f, 1e-10f, 1e6f};
FAST_MATH_SPECIAL_TESTS(LogDefaultSpecial, ::logf, dfm::log, kLogDefaultSpecials, kLogMaxUlps)
