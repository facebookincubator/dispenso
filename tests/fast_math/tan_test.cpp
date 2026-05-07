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

TEST(Tan, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_NE(dispenso::fast_math::tan(-kInf), dispenso::fast_math::tan(-kInf));
  EXPECT_NE(dispenso::fast_math::tan(kInf), dispenso::fast_math::tan(kInf));
  EXPECT_NE(dispenso::fast_math::tan(kNaN), dispenso::fast_math::tan(kNaN));
  EXPECT_EQ(dispenso::fast_math::tan(0.0f), 0.0f);
}

constexpr uint32_t kTanAccurateUlps = 3 + kMsvcUlpSlack;
constexpr uint32_t kTanAccurateUlpsLg = 4 + kMsvcUlpSlack;

TEST(TanLessAccurate, Range8Pi) {
  auto result =
      dispenso::fast_math::evalAccuracy(::tanf, dispenso::fast_math::tan<float>, -8 * kPi, 8 * kPi);

  EXPECT_LE(result, kTanAccurateUlps);
}

TEST(TanLessAccurate, Range1KPi) {
  auto result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, -(1 << 10) * kPi, -8 * kPi);
  EXPECT_LE(result, kTanAccurateUlps);

  result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, 8 * kPi, (1 << 10) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);
}

TEST(TanLessAccurate, Range32KPi) {
  auto result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, -(1 << 15) * kPi, -(1 << 10) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);

  result = dispenso::fast_math::evalAccuracy(
      ::tanf, dispenso::fast_math::tan<float>, (1 << 10) * kPi, (1 << 15) * kPi);
  EXPECT_LE(result, kTanAccurateUlps);
}

auto tan_accurate = dispenso::fast_math::tan<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(TanAccurate, Range2MPi) {
  auto result =
      dispenso::fast_math::evalAccuracy(::tanf, tan_accurate, -(1 << 20) * kPi, -(1 << 15) * kPi);
  EXPECT_LE(result, kTanAccurateUlpsLg);

  result =
      dispenso::fast_math::evalAccuracy(::tanf, tan_accurate, (1 << 15) * kPi, (1 << 20) * kPi);
  EXPECT_LE(result, kTanAccurateUlpsLg);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt tan_max(Flt x) {
  return dfm::tan<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
FAST_MATH_ACCURACY_TESTS(
    TanDefaultAll,
    ::tanf,
    dfm::tan,
    -(1 << 15) * kPi,
    (1 << 15) * kPi,
    kTanAccurateUlps)
FAST_MATH_ACCURACY_TESTS(
    TanMaxAccAll,
    ::tanf,
    tan_max,
    -(1 << 20) * kPi,
    (1 << 20) * kPi,
    kTanAccurateUlpsLg)

// Special values tested across all SIMD backends.
static const float kTanSpecials[] = {
    0.0f,
    -0.0f,
    kPi_4,
    -kPi_4,
    1.0f,
    -1.0f,
    50.0f,
    -50.0f,
    200.0f,
    -200.0f,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(TanDefaultSpecial, ::tanf, dfm::tan, kTanSpecials, kTanAccurateUlps)
FAST_MATH_SPECIAL_TESTS(TanMaxAccSpecial, ::tanf, tan_max, kTanSpecials, kTanAccurateUlpsLg)
