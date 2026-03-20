/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/util.h>

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;

// ---- bit_cast ----

TEST(BitCast, FloatToUint) {
  EXPECT_EQ(dfm::bit_cast<uint32_t>(0.0f), 0u);
  EXPECT_EQ(dfm::bit_cast<uint32_t>(-0.0f), 0x80000000u);
  EXPECT_EQ(dfm::bit_cast<uint32_t>(1.0f), 0x3f800000u);
  EXPECT_EQ(dfm::bit_cast<uint32_t>(-1.0f), 0xbf800000u);
}

TEST(BitCast, UintToFloat) {
  EXPECT_EQ(dfm::bit_cast<float>(0x3f800000u), 1.0f);
  EXPECT_EQ(dfm::bit_cast<float>(0x40000000u), 2.0f);
}

TEST(BitCast, RoundTrip) {
  float vals[] = {0.0f, 1.0f, -1.0f, 3.14f, -2.718f, std::numeric_limits<float>::infinity()};
  for (float v : vals) {
    EXPECT_EQ(dfm::bit_cast<float>(dfm::bit_cast<uint32_t>(v)), v);
    EXPECT_EQ(dfm::bit_cast<float>(dfm::bit_cast<int32_t>(v)), v);
  }
}

// ---- float_distance ----

TEST(FloatDistance, IdenticalValues) {
  EXPECT_EQ(dfm::float_distance(1.0f, 1.0f), 0u);
  EXPECT_EQ(dfm::float_distance(0.0f, 0.0f), 0u);
  EXPECT_EQ(dfm::float_distance(-1.0f, -1.0f), 0u);
}

TEST(FloatDistance, OneUlp) {
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f);
  EXPECT_EQ(dfm::float_distance(a, b), 1u);
  EXPECT_EQ(dfm::float_distance(b, a), 1u);
}

TEST(FloatDistance, SymmetricForPositive) {
  float a = 1.0f;
  float b = 1.5f;
  EXPECT_EQ(dfm::float_distance(a, b), dfm::float_distance(b, a));
}

TEST(FloatDistance, DenormalsAreZero) {
  // Denormals should be treated as zero
  float denorm = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(dfm::float_distance(denorm, 0.0f), 0u);
}

// ---- signof ----

TEST(Signof, PositiveValues) {
  EXPECT_EQ(dfm::signof(1.0f), 1.0f);
  EXPECT_EQ(dfm::signof(42.0f), 1.0f);
  EXPECT_EQ(dfm::signof(0.001f), 1.0f);
}

TEST(Signof, NegativeValues) {
  EXPECT_EQ(dfm::signof(-1.0f), -1.0f);
  EXPECT_EQ(dfm::signof(-42.0f), -1.0f);
  EXPECT_EQ(dfm::signof(-0.001f), -1.0f);
}

TEST(Signof, Zero) {
  // +0 should give +1, -0 should give -1
  EXPECT_EQ(dfm::signof(0.0f), 1.0f);
  EXPECT_EQ(dfm::signof(-0.0f), -1.0f);
}

// ---- signofi ----

TEST(Signofi, PositiveValues) {
  EXPECT_EQ((dfm::signofi<float>(1)), 1);
  EXPECT_EQ((dfm::signofi<float>(100)), 1);
}

TEST(Signofi, NegativeValues) {
  EXPECT_EQ((dfm::signofi<float>(-1)), -1);
  EXPECT_EQ((dfm::signofi<float>(-100)), -1);
}

TEST(Signofi, Zero) {
  EXPECT_EQ((dfm::signofi<float>(0)), 1);
}

// ---- nonnormal ----

TEST(Nonnormal, IntVersion) {
  // Inf: exponent all 1s, mantissa 0
  int32_t inf_bits = dfm::bit_cast<int32_t>(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(dfm::nonnormal<float>(inf_bits));

  // NaN: exponent all 1s, mantissa nonzero
  int32_t nan_bits = dfm::bit_cast<int32_t>(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(dfm::nonnormal<float>(nan_bits));

  // Normal number
  int32_t one_bits = dfm::bit_cast<int32_t>(1.0f);
  EXPECT_FALSE(dfm::nonnormal<float>(one_bits));

  int32_t neg_one_bits = dfm::bit_cast<int32_t>(-1.0f);
  EXPECT_FALSE(dfm::nonnormal<float>(neg_one_bits));
}

TEST(Nonnormal, FloatVersion) {
  EXPECT_TRUE(dfm::nonnormal<float>(std::numeric_limits<float>::infinity()));
  EXPECT_TRUE(dfm::nonnormal<float>(-std::numeric_limits<float>::infinity()));
  EXPECT_TRUE(dfm::nonnormal<float>(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_FALSE(dfm::nonnormal<float>(1.0f));
  EXPECT_FALSE(dfm::nonnormal<float>(-42.0f));
}

// ---- nonnormalOrZero ----

TEST(NonnormalOrZero, Normal) {
  int32_t bits = dfm::bit_cast<int32_t>(1.0f);
  EXPECT_FALSE(dfm::nonnormalOrZero<float>(bits));
  bits = dfm::bit_cast<int32_t>(-42.0f);
  EXPECT_FALSE(dfm::nonnormalOrZero<float>(bits));
}

TEST(NonnormalOrZero, Zero) {
  int32_t bits = dfm::bit_cast<int32_t>(0.0f);
  EXPECT_TRUE(dfm::nonnormalOrZero<float>(bits));
}

TEST(NonnormalOrZero, Inf) {
  int32_t bits = dfm::bit_cast<int32_t>(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(dfm::nonnormalOrZero<float>(bits));
}

TEST(NonnormalOrZero, NaN) {
  int32_t bits = dfm::bit_cast<int32_t>(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(dfm::nonnormalOrZero<float>(bits));
}

// ---- convert_to_int ----

TEST(ConvertToInt, BasicValues) {
  EXPECT_EQ(dfm::convert_to_int(0.0f), 0);
  EXPECT_EQ(dfm::convert_to_int(1.0f), 1);
  EXPECT_EQ(dfm::convert_to_int(-1.0f), -1);
  EXPECT_EQ(dfm::convert_to_int(2.0f), 2);
  EXPECT_EQ(dfm::convert_to_int(-2.0f), -2);
}

TEST(ConvertToInt, Rounding) {
  // SSE uses round-to-nearest-even (banker's rounding)
  EXPECT_EQ(dfm::convert_to_int(1.4f), 1);
  EXPECT_EQ(dfm::convert_to_int(1.6f), 2);
  EXPECT_EQ(dfm::convert_to_int(-1.4f), -1);
  EXPECT_EQ(dfm::convert_to_int(-1.6f), -2);
  // Banker's rounding: 0.5 rounds to nearest even
  EXPECT_EQ(dfm::convert_to_int(0.5f), 0); // 0.5 → 0 (even)
  EXPECT_EQ(dfm::convert_to_int(1.5f), 2); // 1.5 → 2 (even)
  EXPECT_EQ(dfm::convert_to_int(2.5f), 2); // 2.5 → 2 (even)
  EXPECT_EQ(dfm::convert_to_int(3.5f), 4); // 3.5 → 4 (even)
  EXPECT_EQ(dfm::convert_to_int(-0.5f), 0);
  EXPECT_EQ(dfm::convert_to_int(-1.5f), -2);
}

TEST(ConvertToInt, LargeValues) {
  EXPECT_EQ(dfm::convert_to_int(100.0f), 100);
  EXPECT_EQ(dfm::convert_to_int(-100.0f), -100);
  EXPECT_EQ(dfm::convert_to_int(1000.0f), 1000);
}

// ---- convert_to_int_clamped ----

TEST(ConvertToIntClamped, WithinRange) {
  EXPECT_EQ((dfm::convert_to_int_clamped<float, -10, 10>(5.0f)), 5);
  EXPECT_EQ((dfm::convert_to_int_clamped<float, -10, 10>(-5.0f)), -5);
  EXPECT_EQ((dfm::convert_to_int_clamped<float, -10, 10>(0.0f)), 0);
}

TEST(ConvertToIntClamped, AtBounds) {
  EXPECT_EQ((dfm::convert_to_int_clamped<float, -10, 10>(10.0f)), 10);
  EXPECT_EQ((dfm::convert_to_int_clamped<float, -10, 10>(-10.0f)), -10);
}

TEST(ConvertToIntClamped, ClampedAbove) {
  EXPECT_LE((dfm::convert_to_int_clamped<float, -10, 10>(100.0f)), 10);
}

TEST(ConvertToIntClamped, ClampedBelow) {
  EXPECT_GE((dfm::convert_to_int_clamped<float, -10, 10>(-100.0f)), -10);
}

// ---- floor_small ----

TEST(FloorSmall, Integers) {
  EXPECT_EQ(dfm::floor_small(0.0f), 0.0f);
  EXPECT_EQ(dfm::floor_small(1.0f), 1.0f);
  EXPECT_EQ(dfm::floor_small(-1.0f), -1.0f);
  EXPECT_EQ(dfm::floor_small(5.0f), 5.0f);
}

TEST(FloorSmall, PositiveFractions) {
  EXPECT_EQ(dfm::floor_small(1.5f), 1.0f);
  EXPECT_EQ(dfm::floor_small(1.9f), 1.0f);
  EXPECT_EQ(dfm::floor_small(1.1f), 1.0f);
  EXPECT_EQ(dfm::floor_small(0.5f), 0.0f);
}

TEST(FloorSmall, NegativeFractions) {
  EXPECT_EQ(dfm::floor_small(-0.5f), -1.0f);
  EXPECT_EQ(dfm::floor_small(-1.5f), -2.0f);
  EXPECT_EQ(dfm::floor_small(-1.1f), -2.0f);
  EXPECT_EQ(dfm::floor_small(-0.1f), -1.0f);
}

TEST(FloorSmall, MatchesStdFloor) {
  float vals[] = {
      0.0f,
      0.1f,
      0.5f,
      0.9f,
      1.0f,
      1.5f,
      2.7f,
      100.3f,
      -0.1f,
      -0.5f,
      -0.9f,
      -1.0f,
      -1.5f,
      -2.7f,
      -100.3f};
  for (float v : vals) {
    EXPECT_EQ(dfm::floor_small(v), std::floor(v)) << "Mismatch for " << v;
  }
}

// ---- min ----

TEST(Min, BasicValues) {
  EXPECT_EQ(dfm::min<float>(1.0f, 2.0f), 1.0f);
  EXPECT_EQ(dfm::min<float>(2.0f, 1.0f), 1.0f);
  EXPECT_EQ(dfm::min<float>(-1.0f, 1.0f), -1.0f);
  EXPECT_EQ(dfm::min<float>(5.0f, 5.0f), 5.0f);
}

TEST(Min, NanBehavior) {
  // If first arg is NaN and second is not, should return second
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(dfm::min<float>(nan, 1.0f), 1.0f);
}

// ---- clamp_allow_nan ----

TEST(ClampAllowNan, WithinRange) {
  EXPECT_EQ(dfm::clamp_allow_nan<float>(0.5f, 0.0f, 1.0f), 0.5f);
}

TEST(ClampAllowNan, BelowMin) {
  EXPECT_EQ(dfm::clamp_allow_nan<float>(-1.0f, 0.0f, 1.0f), 0.0f);
}

TEST(ClampAllowNan, AboveMax) {
  EXPECT_EQ(dfm::clamp_allow_nan<float>(2.0f, 0.0f, 1.0f), 1.0f);
}

TEST(ClampAllowNan, NanInput) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float result = dfm::clamp_allow_nan<float>(nan, 0.0f, 1.0f);
  // "allow_nan" refers to handling NaN in the bounds, not clamping NaN input.
  // With NaN input on SSE, NaN propagates through min/max chain.
  EXPECT_TRUE(std::isnan(result));
}

// ---- clamp_no_nan ----

TEST(ClampNoNan, WithinRange) {
  EXPECT_EQ(dfm::clamp_no_nan<float>(0.5f, 0.0f, 1.0f), 0.5f);
}

TEST(ClampNoNan, BelowMin) {
  EXPECT_EQ(dfm::clamp_no_nan<float>(-1.0f, 0.0f, 1.0f), 0.0f);
}

TEST(ClampNoNan, AboveMax) {
  EXPECT_EQ(dfm::clamp_no_nan<float>(2.0f, 0.0f, 1.0f), 1.0f);
}

TEST(ClampNoNan, AtBounds) {
  EXPECT_EQ(dfm::clamp_no_nan<float>(0.0f, 0.0f, 1.0f), 0.0f);
  EXPECT_EQ(dfm::clamp_no_nan<float>(1.0f, 0.0f, 1.0f), 1.0f);
}

// ---- gather ----

TEST(Gather, BasicLookup) {
  float table[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  EXPECT_EQ(dfm::gather<float>(table, 0), 10.0f);
  EXPECT_EQ(dfm::gather<float>(table, 2), 30.0f);
  EXPECT_EQ(dfm::gather<float>(table, 4), 50.0f);
}

// ---- nbool_as_one / bool_as_one ----

TEST(NboolAsOne, Float) {
  EXPECT_EQ((dfm::nbool_as_one<float, bool>(true)), 0.0f);
  EXPECT_EQ((dfm::nbool_as_one<float, bool>(false)), 1.0f);
}

TEST(NboolAsOne, Int) {
  EXPECT_EQ((dfm::nbool_as_one<int32_t, bool>(true)), 0);
  EXPECT_EQ((dfm::nbool_as_one<int32_t, bool>(false)), 1);
}

TEST(BoolAsOne, Float) {
  EXPECT_EQ((dfm::bool_as_one<float, bool>(true)), 1.0f);
  EXPECT_EQ((dfm::bool_as_one<float, bool>(false)), 0.0f);
}

TEST(BoolAsOne, Int) {
  EXPECT_EQ((dfm::bool_as_one<int32_t, bool>(true)), 1);
  EXPECT_EQ((dfm::bool_as_one<int32_t, bool>(false)), 0);
}

// ---- bool_as_mask ----

TEST(BoolAsMask, TrueGivesAllOnes) {
  int32_t mask = dfm::bool_as_mask<int32_t, bool>(true);
  EXPECT_EQ(mask, static_cast<int32_t>(0xFFFFFFFF));
}

TEST(BoolAsMask, FalseGivesZero) {
  int32_t mask = dfm::bool_as_mask<int32_t, bool>(false);
  EXPECT_EQ(mask, 0);
}

// ---- bool_apply_or_zero ----

TEST(BoolApplyOrZero, TrueReturnsValue) {
  EXPECT_EQ((dfm::bool_apply_or_zero<int32_t>(true, 42)), 42);
  EXPECT_EQ((dfm::bool_apply_or_zero<int32_t>(true, -7)), -7);
}

TEST(BoolApplyOrZero, FalseReturnsZero) {
  EXPECT_EQ((dfm::bool_apply_or_zero<int32_t>(false, 42)), 0);
  EXPECT_EQ((dfm::bool_apply_or_zero<int32_t>(false, -7)), 0);
}

TEST(BoolApplyOrZero, FloatTrue) {
  float result = dfm::bool_apply_or_zero<float>(true, 3.14f);
  EXPECT_EQ(result, 3.14f);
}

TEST(BoolApplyOrZero, FloatFalse) {
  float result = dfm::bool_apply_or_zero<float>(false, 3.14f);
  EXPECT_EQ(result, 0.0f);
}

// ---- int_div_by_3 ----

TEST(IntDivBy3, ExactMultiples) {
  EXPECT_EQ(dfm::int_div_by_3(0), 0);
  EXPECT_EQ(dfm::int_div_by_3(3), 1);
  EXPECT_EQ(dfm::int_div_by_3(6), 2);
  EXPECT_EQ(dfm::int_div_by_3(9), 3);
  EXPECT_EQ(dfm::int_div_by_3(30), 10);
  EXPECT_EQ(dfm::int_div_by_3(300), 100);
}

TEST(IntDivBy3, WithRemainder) {
  // Truncation toward zero like integer division
  EXPECT_EQ(dfm::int_div_by_3(1), 0);
  EXPECT_EQ(dfm::int_div_by_3(2), 0);
  EXPECT_EQ(dfm::int_div_by_3(4), 1);
  EXPECT_EQ(dfm::int_div_by_3(5), 1);
  EXPECT_EQ(dfm::int_div_by_3(7), 2);
  EXPECT_EQ(dfm::int_div_by_3(8), 2);
  EXPECT_EQ(dfm::int_div_by_3(10), 3);
}

TEST(IntDivBy3, LargeValues) {
  // Values used in cbrt magic constant computations
  for (int32_t i = 0; i < 10000; i += 7) {
    EXPECT_EQ(dfm::int_div_by_3(i), i / 3) << "Mismatch at " << i;
  }
}
