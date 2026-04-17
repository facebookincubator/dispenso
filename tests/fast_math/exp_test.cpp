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

struct BoundsOnlyTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

auto exp_accurate = dispenso::fast_math::exp<float, dispenso::fast_math::MaxAccuracyTraits>;

TEST(Exp, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp_accurate(0.0f), 1.0f);
  EXPECT_EQ(exp_accurate(-kInf), 0.0f);
  EXPECT_EQ(exp_accurate(kInf), kInf);
  EXPECT_NE(exp_accurate(kNaN), exp_accurate(kNaN));
}

TEST(Exp, RangeSmall) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, -127.0f, -40.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp, RangeSmallish) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, -40.0f, 20.0f);
  EXPECT_LE(res, 1u);
}
TEST(Exp, RangeMedium) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, 20.0f, 40.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp, RangeLarge) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_accurate, 40.0f, 128.0f);
  EXPECT_LE(res, 1u);
}

auto exp_bounds = dispenso::fast_math::exp<float, BoundsOnlyTraits>;

TEST(ExpLessAccurateWBounds, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(exp_bounds(0.0f), 1.0f);
  EXPECT_EQ(exp_bounds(-kInf), 0.0f);
  EXPECT_EQ(exp_bounds(kInf), kInf);
  EXPECT_NE(exp_bounds(kNaN), exp_bounds(kNaN));
  EXPECT_LT(exp_bounds(-100.0f), 1e-38f);
}

TEST(ExpLessAccurateWBounds, Range_m100_100) {
  uint32_t res = dispenso::fast_math::evalAccuracy(::expf, exp_bounds, -100.0f, 100.0f);
  EXPECT_LE(res, 1u);
}

TEST(ExpLessAccurate, Range_m88_88) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(::expf, dispenso::fast_math::exp<float>, -88.0f, 88.0f);
  EXPECT_LE(res, 3u);
}

// Wrappers for traits variants — macro instantiates func<Flt>.
template <typename Flt>
Flt exp_max_fn(Flt x) {
  return dfm::exp<Flt, dfm::MaxAccuracyTraits>(x);
}

template <typename Flt>
Flt exp_bounds_fn(Flt x) {
  return dfm::exp<Flt, BoundsOnlyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kExpMaxAccMaxUlps = 1;
constexpr uint32_t kExpBoundsMaxUlps = 1;
constexpr uint32_t kExpDefaultMaxUlps = 3;
FAST_MATH_ACCURACY_TESTS(ExpMaxAccAll, ::expf, exp_max_fn, -127.0f, 128.0f, kExpMaxAccMaxUlps)
FAST_MATH_ACCURACY_TESTS(ExpBoundsAll, ::expf, exp_bounds_fn, -100.0f, 100.0f, kExpBoundsMaxUlps)
FAST_MATH_ACCURACY_TESTS(ExpDefaultAll, ::expf, dfm::exp, -88.0f, 88.0f, kExpDefaultMaxUlps)

// Special values tested across all SIMD backends.
// MaxAccuracy handles NaN/Inf.
static const float kExpMaxAccSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    5.0f,
    -5.0f,
    10.0f,
    -10.0f,
    88.7f,
    -88.7f,
    1e-10f,
    std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(ExpMaxAccSpecial, ::expf, exp_max_fn, kExpMaxAccSpecials, kExpMaxAccMaxUlps)

// BoundsOnly handles NaN/Inf and clamps out-of-range inputs.
static const float kExpBoundsSpecials[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    5.0f,
    -5.0f,
    89.0f,
    -100.0f,
    200.0f,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(
    ExpBoundsSpecial,
    ::expf,
    exp_bounds_fn,
    kExpBoundsSpecials,
    kExpBoundsMaxUlps)
