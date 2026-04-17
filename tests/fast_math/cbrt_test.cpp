/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

#include <iomanip>
#include <limits>

namespace dfm = dispenso::fast_math;
using namespace dispenso::fast_math::testing;

auto cbrt_acc = dispenso::fast_math::cbrt<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Cbrt, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(cbrt_acc(0.0f), 0.0f) << std::setprecision(16) << cbrt_acc(0.0f);
  EXPECT_EQ(cbrt_acc(-kInf), -kInf);
  EXPECT_EQ(cbrt_acc(kInf), kInf);
  EXPECT_NE(cbrt_acc(kNaN), cbrt_acc(kNaN));
}

constexpr uint32_t kCbrtOuterUlps = 13; // 12 bits on Linux, MacOs, 13 with MSVC

TEST(Cbrt, RangeNeg) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::epsilon());

  EXPECT_LE(result, kCbrtOuterUlps);
}
TEST(Cbrt, RangePos) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::max());

  EXPECT_LE(result, kCbrtOuterUlps);
}
TEST(Cbrt, RangeSmall) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      dispenso::fast_math::cbrt<float>,
      -std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::epsilon());

  // cbrt handles denormals via rescaling — uniform ULP accuracy near zero.
  EXPECT_LE(result, kCbrtOuterUlps);
}

constexpr uint32_t kCbrtOuterUlpsAcc = 3;

TEST(CbrtAccurate, RangeNeg) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf, cbrt_acc, -std::numeric_limits<float>::max(), -std::numeric_limits<float>::epsilon());

  EXPECT_LE(result, kCbrtOuterUlpsAcc);
}
TEST(CbrtAccurate, RangePos) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf, cbrt_acc, std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max());

  EXPECT_LE(result, kCbrtOuterUlpsAcc);
}
TEST(CbrtAccurate, RangeSmall) {
  auto result = dispenso::fast_math::evalAccuracy(
      cbrtf,
      cbrt_acc,
      -std::numeric_limits<float>::epsilon(),
      std::numeric_limits<float>::epsilon());

  EXPECT_LE(result, kCbrtOuterUlpsAcc);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt cbrt_max(Flt x) {
  return dfm::cbrt<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
FAST_MATH_ACCURACY_TESTS(
    CbrtDefaultAll,
    ::cbrtf,
    dfm::cbrt,
    -std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    kCbrtOuterUlps)
FAST_MATH_ACCURACY_TESTS(
    CbrtMaxAccAll,
    ::cbrtf,
    cbrt_max,
    -std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    kCbrtOuterUlpsAcc)

// Special values tested across all SIMD backends.
// MaxAccuracy handles NaN/Inf.
static const float kCbrtMaxAccSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    8.0f,
    -8.0f,
    27.0f,
    -27.0f,
    64.0f,
    125.0f,
    0.125f,
    0.001f,
    1e-20f,
    std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(
    CbrtMaxAccSpecial,
    ::cbrtf,
    cbrt_max,
    kCbrtMaxAccSpecials,
    kCbrtOuterUlpsAcc)

// Default traits don't handle NaN/Inf — only test in-domain values.
static const float kCbrtDefaultSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    8.0f,
    -8.0f,
    27.0f,
    -27.0f,
    64.0f,
    125.0f,
    0.125f,
    0.001f,
    1000.0f,
    -1000.0f};
FAST_MATH_SPECIAL_TESTS(
    CbrtDefaultSpecial,
    ::cbrtf,
    dfm::cbrt,
    kCbrtDefaultSpecials,
    kCbrtOuterUlps)
