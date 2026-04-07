/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#if __has_include("hwy/highway.h")
#include "hwy/highway.h"

namespace dfm = dispenso::fast_math;
namespace hn = hwy::HWY_NAMESPACE;
using HwyFloat = dfm::HwyFloat;
using HwyInt32 = dfm::HwyInt32;
using HwyUint32 = dfm::HwyUint32;
using HwyFloatTag = dfm::HwyFloatTag;
using HwyInt32Tag = dfm::HwyInt32Tag;
using HwyUint32Tag = dfm::HwyUint32Tag;

// Raw Highway vector type aliases — used in tests to exercise SimdTypeFor forwarding.
using HwyVecF = hn::Vec<HwyFloatTag>;
using HwyVecI = hn::Vec<HwyInt32Tag>;
using HwyVecU = hn::Vec<HwyUint32Tag>;

// Lane count (constant per compile-time target).
static size_t N() {
  return hn::Lanes(HwyFloatTag{});
}

constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(float);

// Helpers: extract lane i from Highway vectors.
static float lane(HwyVecF v, size_t i) {
  HWY_ALIGN float buf[kMaxLanes];
  hn::StoreU(v, HwyFloatTag{}, buf);
  return buf[i];
}
static float lane(HwyFloat v, size_t i) {
  return lane(v.v, i);
}

static int32_t lane(HwyVecI v, size_t i) {
  HWY_ALIGN int32_t buf[kMaxLanes];
  hn::StoreU(v, HwyInt32Tag{}, buf);
  return buf[i];
}
static int32_t lane(HwyInt32 v, size_t i) {
  return lane(v.v, i);
}

static uint32_t lane(HwyVecU v, size_t i) {
  HWY_ALIGN uint32_t buf[kMaxLanes];
  hn::StoreU(v, HwyUint32Tag{}, buf);
  return buf[i];
}
static uint32_t lane(HwyUint32 v, size_t i) {
  return lane(v.v, i);
}

// Helper: create HwyVecF from an array of values (must have N() entries).
static HwyVecF loadF(const float* vals) {
  return hn::LoadU(HwyFloatTag{}, vals);
}

// Helper: create HwyVecI from an array of values.
static HwyVecI loadI(const int32_t* vals) {
  return hn::LoadU(HwyInt32Tag{}, vals);
}

// Helper: compare lane-by-lane against scalar function using raw Highway vector types.
template <typename ScalarFunc, typename HwyFunc>
static void
checkLaneByLane(ScalarFunc scalar_fn, HwyFunc hwy_fn, HwyVecF input, uint32_t max_ulps = 0) {
  HwyVecF result = hwy_fn(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = scalar_fn(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << ": expected NaN, got " << actual;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, max_ulps) << "Lane " << i << ": input=" << lane(input, i)
                                << " expected=" << expected << " actual=" << actual
                                << " ulps=" << dist;
    }
  }
}

// ---- HwyFloat basic arithmetic ----

TEST(HwyFloat, Broadcast) {
  HwyFloat v(3.0f);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(v, i), 3.0f);
  }
}

TEST(HwyFloat, Arithmetic) {
  // Use sequential values: 1, 2, 3, ...
  HWY_ALIGN float a_vals[kMaxLanes];
  HWY_ALIGN float b_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    a_vals[i] = static_cast<float>(i + 1);
    b_vals[i] = static_cast<float>(i + 5);
  }
  HwyFloat a = loadF(a_vals);
  HwyFloat b = loadF(b_vals);

  HwyFloat sum = a + b;
  HwyFloat diff = a - b;
  HwyFloat prod = a * b;
  HwyFloat quot = a / b;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(sum, i), a_vals[i] + b_vals[i]);
    EXPECT_EQ(lane(diff, i), a_vals[i] - b_vals[i]);
    EXPECT_EQ(lane(prod, i), a_vals[i] * b_vals[i]);
    EXPECT_NEAR(lane(quot, i), a_vals[i] / b_vals[i], 1e-6f);
  }
}

TEST(HwyFloat, Negation) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = static_cast<float>(i + 1);
  }
  HwyFloat v = loadF(vals);
  HwyFloat neg = -v;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(neg, i), -vals[i]);
  }

  // Test -0.0f: negation should flip the sign bit.
  HwyFloat z(0.0f);
  HwyFloat negz = -z;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::signbit(lane(negz, i)));
  }
}

TEST(HwyFloat, CompoundAssignment) {
  HwyFloat a(2.0f);
  HwyFloat b(3.0f);

  HwyFloat c = a;
  c += b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(c, i), 5.0f);
  }

  c = a;
  c -= b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(c, i), -1.0f);
  }

  c = a;
  c *= b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(c, i), 6.0f);
  }
}

// ---- HwyFloat comparisons ----

TEST(HwyFloat, Comparisons) {
  HwyFloat a(1.0f);
  HwyFloat b(2.0f);

  // a < b: all lanes should be true (all-ones mask).
  HwyFloat lt = a < b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, i)), 0u) << "Lane " << i;
  }

  // a > b: all lanes should be false.
  HwyFloat gt = a > b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(gt, i)), 0u) << "Lane " << i;
  }

  // a == a: all lanes should be true.
  HwyFloat eq = a == a;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(eq, i)), 0xFFFFFFFFu) << "Lane " << i;
  }

  // a != b: all lanes should be true.
  HwyFloat ne = a != b;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_NE(dfm::bit_cast<uint32_t>(lane(ne, i)), 0u) << "Lane " << i;
  }
}

TEST(HwyFloat, LogicalNot) {
  HwyFloat a(1.0f);
  HwyFloat b(2.0f);
  HwyFloat mask = a < b; // all true
  HwyFloat notmask = !mask;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(notmask, i)), 0u) << "Lane " << i;
  }
}

// ---- HwyInt32 ----

TEST(HwyInt32, ArithmeticAndShifts) {
  HWY_ALIGN int32_t a_vals[kMaxLanes];
  HWY_ALIGN int32_t b_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    a_vals[i] = static_cast<int32_t>(10 + i);
    b_vals[i] = static_cast<int32_t>(3 + i);
  }
  HwyInt32 a = loadI(a_vals);
  HwyInt32 b = loadI(b_vals);

  HwyInt32 sum = a + b;
  HwyInt32 diff = a - b;
  HwyInt32 prod = a * b;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(sum, i), a_vals[i] + b_vals[i]);
    EXPECT_EQ(lane(diff, i), a_vals[i] - b_vals[i]);
    EXPECT_EQ(lane(prod, i), a_vals[i] * b_vals[i]);
  }

  // Shifts.
  HwyInt32 shl = a << 2;
  HwyInt32 shr = a >> 1;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(shl, i), a_vals[i] << 2);
    EXPECT_EQ(lane(shr, i), a_vals[i] >> 1);
  }
}

TEST(HwyInt32, Negation) {
  HwyInt32 a(42);
  HwyInt32 neg = -a;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(neg, i), -42);
  }
}

TEST(HwyInt32, Bitwise) {
  HwyInt32 a(0xFF00FF00);
  HwyInt32 b(0x0F0F0F0F);

  HwyInt32 andResult = a & b;
  HwyInt32 orResult = a | b;
  HwyInt32 xorResult = a ^ b;
  HwyInt32 notResult = ~a;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(andResult, i), int32_t(0xFF00FF00 & 0x0F0F0F0F));
    EXPECT_EQ(lane(orResult, i), int32_t(0xFF00FF00 | 0x0F0F0F0F));
    EXPECT_EQ(lane(xorResult, i), int32_t(0xFF00FF00 ^ 0x0F0F0F0F));
    EXPECT_EQ(lane(notResult, i), ~int32_t(0xFF00FF00));
  }
}

TEST(HwyInt32, Comparisons) {
  HwyInt32 a(5);
  HwyInt32 b(10);

  HwyInt32 lt = a < b;
  HwyInt32 gt = a > b;
  HwyInt32 eq = a == a;
  HwyInt32 ne = a != b;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(lt, i), -1); // all-ones
    EXPECT_EQ(lane(gt, i), 0);
    EXPECT_EQ(lane(eq, i), -1);
    EXPECT_EQ(lane(ne, i), -1);
  }
}

// ---- HwyUint32 ----

TEST(HwyUint32, ArithmeticAndShifts) {
  HwyUint32 a(100u);
  HwyUint32 b(30u);

  HwyUint32 sum = a + b;
  HwyUint32 diff = a - b;
  HwyUint32 prod = a * b;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(sum, i), 130u);
    EXPECT_EQ(lane(diff, i), 70u);
    EXPECT_EQ(lane(prod, i), 3000u);
  }

  // Logical right shift.
  HwyUint32 shr = a >> 2;
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(shr, i), 25u);
  }
}

TEST(HwyUint32, UnsignedComparisons) {
  HwyUint32 a(0x80000001u); // large unsigned value
  HwyUint32 b(1u);

  HwyUint32 gt = a > b;
  HwyUint32 lt = a < b;

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(gt, i), 0xFFFFFFFFu) << "Lane " << i; // a > b unsigned
    EXPECT_EQ(lane(lt, i), 0u) << "Lane " << i;
  }
}

// ---- bit_cast ----

TEST(HwyBitCast, FloatIntRoundTrip) {
  HwyFloat f(1.0f);
  HwyInt32 i = dfm::bit_cast<HwyInt32>(f);
  HwyFloat f2 = dfm::bit_cast<HwyFloat>(i);

  for (size_t j = 0; j < N(); ++j) {
    EXPECT_EQ(lane(f2, j), 1.0f);
    EXPECT_EQ(lane(i, j), 0x3f800000);
  }
}

TEST(HwyBitCast, FloatUintRoundTrip) {
  HwyFloat f(-1.0f);
  HwyUint32 u = dfm::bit_cast<HwyUint32>(f);
  HwyFloat f2 = dfm::bit_cast<HwyFloat>(u);

  for (size_t j = 0; j < N(); ++j) {
    EXPECT_EQ(lane(f2, j), -1.0f);
    EXPECT_EQ(lane(u, j), 0xBF800000u);
  }
}

// ---- FloatTraits ----

TEST(HwyFloatTraits, Conditional) {
  using FT = dfm::FloatTraits<HwyFloat>;

  // Create a mask: alternating true/false per lane.
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = (i % 2 == 0) ? 1.0f : 2.0f;
  }
  HwyVecF input = loadF(vals);
  HwyFloat mask = input < HwyFloat(1.5f); // even lanes true, odd lanes false

  HwyFloat x(10.0f);
  HwyFloat y(20.0f);
  HwyFloat result = FT::conditional(mask, x, y);

  for (size_t i = 0; i < N(); ++i) {
    float expected = (i % 2 == 0) ? 10.0f : 20.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(HwyFloatTraits, ConditionalInt32) {
  using FT = dfm::FloatTraits<HwyFloat>;

  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = (i % 2 == 0) ? 1.0f : 2.0f;
  }
  HwyFloat mask = loadF(vals) < HwyFloat(1.5f);

  HwyInt32 x(100);
  HwyInt32 y(200);
  HwyInt32 result = FT::conditional(mask, x, y);

  for (size_t i = 0; i < N(); ++i) {
    int32_t expected = (i % 2 == 0) ? 100 : 200;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(HwyFloatTraits, ConditionalWithLaneWideMask) {
  using FT = dfm::FloatTraits<HwyFloat>;

  // Create an integer lane-wide mask (alternating -1/0).
  HWY_ALIGN int32_t mask_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    mask_vals[i] = (i % 2 == 0) ? -1 : 0;
  }
  HwyInt32 mask = loadI(mask_vals);

  HwyFloat x(10.0f);
  HwyFloat y(20.0f);
  HwyFloat result = FT::conditional(mask, x, y);

  for (size_t i = 0; i < N(); ++i) {
    float expected = (i % 2 == 0) ? 10.0f : 20.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(HwyFloatTraits, Apply) {
  using FT = dfm::FloatTraits<HwyFloat>;

  HwyFloat mask = HwyFloat(1.0f) < HwyFloat(2.0f); // all true
  HwyFloat x(42.0f);
  HwyFloat result = FT::apply(mask, x);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), 42.0f) << "Lane " << i;
  }

  // All false mask.
  HwyFloat falseMask = HwyFloat(2.0f) < HwyFloat(1.0f);
  HwyFloat result2 = FT::apply(falseMask, x);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result2, i), 0.0f) << "Lane " << i;
  }
}

TEST(HwyFloatTraits, FmaAndSqrt) {
  using FT = dfm::FloatTraits<HwyFloat>;

  HwyFloat a(2.0f);
  HwyFloat b(3.0f);
  HwyFloat c(4.0f);

  // fma: a*b + c = 2*3 + 4 = 10
  HwyFloat fma_result = FT::fma(a, b, c);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_NEAR(lane(fma_result, i), 10.0f, 1e-6f);
  }

  // sqrt(4.0) = 2.0
  HwyFloat sq = FT::sqrt(c);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(sq, i), 2.0f);
  }
}

TEST(HwyFloatTraits, MinMax) {
  using FT = dfm::FloatTraits<HwyFloat>;

  HWY_ALIGN float a_vals[kMaxLanes];
  HWY_ALIGN float b_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    a_vals[i] = static_cast<float>(i);
    b_vals[i] = static_cast<float>(N() - 1 - i);
  }
  HwyFloat a = loadF(a_vals);
  HwyFloat b = loadF(b_vals);

  HwyFloat mn = FT::min(a, b);
  HwyFloat mx = FT::max(a, b);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(mn, i), std::min(a_vals[i], b_vals[i])) << "Lane " << i;
    EXPECT_EQ(lane(mx, i), std::max(a_vals[i], b_vals[i])) << "Lane " << i;
  }
}

// ---- Util functions ----

TEST(HwyUtil, FloorSmall) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.7f, -1.7f, 2.0f, -0.5f, 3.3f, -3.3f, 0.0f, 100.9f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  HwyFloat result = dfm::floor_small(input);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), std::floor(vals[i])) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyUtil, ConvertToInt) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.5f, -1.5f, 2.5f, -2.5f, 3.5f, -3.5f, 0.4f, 100.6f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  HwyInt32 result = dfm::convert_to_int(input);

  for (size_t i = 0; i < N(); ++i) {
    int32_t expected = static_cast<int32_t>(std::nearbyint(vals[i]));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyUtil, ConvertToIntNaN) {
  HwyFloat nan_val(std::numeric_limits<float>::quiet_NaN());
  HwyInt32 result = dfm::convert_to_int(nan_val);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), 0) << "Lane " << i;
  }
}

TEST(HwyUtil, Gather) {
  float table[16];
  for (int j = 0; j < 16; ++j) {
    table[j] = static_cast<float>(j * 10);
  }

  HWY_ALIGN int32_t idx[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    idx[i] = static_cast<int32_t>(i % 16);
  }
  HwyInt32 index = loadI(idx);
  HwyFloat result = dfm::gather<HwyFloat>(table, index);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), table[idx[i]]) << "Lane " << i;
  }
}

TEST(HwyUtil, IntDivBy3) {
  HWY_ALIGN int32_t vals[kMaxLanes];
  int32_t test_vals[] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 16];
  }
  HwyInt32 input = loadI(vals);
  HwyInt32 result = dfm::int_div_by_3(input);

  for (size_t i = 0; i < N(); ++i) {
    int32_t expected = dfm::int_div_by_3(vals[i]);
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyUtil, Signof) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.0f, -1.0f, 100.0f, -0.5f, 0.0f, -0.0f, 42.0f, -42.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  HwyFloat result = dfm::signof(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = std::signbit(vals[i]) ? -1.0f : 1.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyUtil, Signofi) {
  HWY_ALIGN int32_t vals[kMaxLanes];
  int32_t test_vals[] = {1, -1, 0, -100, 50, -50, 0, 1000};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyInt32 input = loadI(vals);
  HwyInt32 result = dfm::signofi(input);

  for (size_t i = 0; i < N(); ++i) {
    int32_t expected = (vals[i] < 0) ? -1 : 1;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyUtil, Nonnormal) {
  HwyFloat inf_val(std::numeric_limits<float>::infinity());
  HwyFloat nan_val(std::numeric_limits<float>::quiet_NaN());
  HwyFloat normal_val(1.0f);

  HwyInt32 inf_nn = dfm::nonnormal(inf_val);
  HwyInt32 nan_nn = dfm::nonnormal(nan_val);
  HwyInt32 normal_nn = dfm::nonnormal(normal_val);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(inf_nn, i), -1) << "Inf should be nonnormal";
    EXPECT_EQ(lane(nan_nn, i), -1) << "NaN should be nonnormal";
    EXPECT_EQ(lane(normal_nn, i), 0) << "1.0 should not be nonnormal";
  }
}

TEST(HwyUtil, BoolAsOne) {
  HwyFloat true_mask = HwyFloat(1.0f) < HwyFloat(2.0f); // all true
  HwyFloat false_mask = HwyFloat(2.0f) < HwyFloat(1.0f); // all false

  HwyFloat true_one = dfm::bool_as_one<HwyFloat, HwyFloat>(true_mask);
  HwyFloat false_one = dfm::bool_as_one<HwyFloat, HwyFloat>(false_mask);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(true_one, i), 1.0f) << "Lane " << i;
    EXPECT_EQ(lane(false_one, i), 0.0f) << "Lane " << i;
  }
}

TEST(HwyUtil, NBoolAsOne) {
  HwyFloat true_mask = HwyFloat(1.0f) < HwyFloat(2.0f);
  HwyFloat false_mask = HwyFloat(2.0f) < HwyFloat(1.0f);

  HwyFloat true_nbo = dfm::nbool_as_one<HwyFloat, HwyFloat>(true_mask);
  HwyFloat false_nbo = dfm::nbool_as_one<HwyFloat, HwyFloat>(false_mask);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(true_nbo, i), 0.0f) << "Lane " << i;
    EXPECT_EQ(lane(false_nbo, i), 1.0f) << "Lane " << i;
  }
}

TEST(HwyUtil, BoolAsMask) {
  HwyFloat true_mask = HwyFloat(1.0f) < HwyFloat(2.0f);
  HwyInt32 int_mask = dfm::bool_as_mask<HwyInt32, HwyFloat>(true_mask);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(int_mask, i), -1) << "Lane " << i; // all-ones = -1
  }
}

TEST(HwyUtil, BoolApplyOrZero) {
  HWY_ALIGN float mask_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    mask_vals[i] = (i % 2 == 0) ? 1.0f : 2.0f;
  }
  HwyFloat input = loadF(mask_vals);
  HwyFloat mask = input < HwyFloat(1.5f); // even lanes true

  HwyFloat value(42.0f);
  HwyFloat result = dfm::bool_apply_or_zero(mask, value);

  for (size_t i = 0; i < N(); ++i) {
    float expected = (i % 2 == 0) ? 42.0f : 0.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(HwyUtil, ClampAllowNan) {
  HwyFloat x(3.0f);
  HwyFloat mn(1.0f);
  HwyFloat mx(2.0f);
  HwyFloat result = dfm::clamp_allow_nan(x, mn, mx);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), 2.0f) << "Lane " << i;
  }

  // NaN should propagate.
  HwyFloat nan_val(std::numeric_limits<float>::quiet_NaN());
  HwyFloat nan_result = dfm::clamp_allow_nan(nan_val, mn, mx);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::isnan(lane(nan_result, i))) << "Lane " << i;
  }
}

TEST(HwyUtil, ClampNoNan) {
  HwyFloat x(0.5f);
  HwyFloat mn(1.0f);
  HwyFloat mx(2.0f);
  HwyFloat result = dfm::clamp_no_nan(x, mn, mx);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), 1.0f) << "Lane " << i;
  }

  // NaN should be suppressed.
  HwyFloat nan_val(std::numeric_limits<float>::quiet_NaN());
  HwyFloat nan_result = dfm::clamp_no_nan(nan_val, mn, mx);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_FALSE(std::isnan(lane(nan_result, i))) << "Lane " << i;
    EXPECT_GE(lane(nan_result, i), 1.0f);
    EXPECT_LE(lane(nan_result, i), 2.0f);
  }
}

// ---- Transcendental functions ----

TEST(HwyTranscendentals, Sin) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::sin(x); },
      input,
      2);
}

TEST(HwyTranscendentals, Cos) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::cos(x); },
      input,
      2);
}

TEST(HwyTranscendentals, Tan) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::tan(x); },
      input,
      3);
}

TEST(HwyTranscendentals, Exp) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp(x); },
      input,
      5);
}

TEST(HwyTranscendentals, Log) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::log(x); },
      input,
      2);
}

TEST(HwyTranscendentals, Acos) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -0.9f + 1.8f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::acos(x); },
      input,
      4);
}

TEST(HwyTranscendentals, Asin) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -0.9f + 1.8f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::asin(x); },
      input,
      4);
}

TEST(HwyTranscendentals, Atan) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::atan(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::atan(x); },
      input,
      3);
}

TEST(HwyTranscendentals, Cbrt) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 100.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::cbrt(x); },
      input,
      12);
}

TEST(HwyTranscendentals, Exp2) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp2(x); },
      input,
      1);
}

TEST(HwyTranscendentals, Exp10) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -3.0f + 6.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp10(x); },
      input,
      3);
}

TEST(HwyTranscendentals, Log2) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::log2(x); },
      input,
      1);
}

TEST(HwyTranscendentals, Log10) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log10(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::log10(x); },
      input,
      3);
}

// ---- Sweep tests (more inputs) ----

TEST(HwyTranscendentals, SinSweep) {
  constexpr int kSteps = 256;
  const float delta = static_cast<float>(2.0 * M_PI / kSteps);
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = static_cast<float>(-M_PI) + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::sin(x); },
        input,
        2);
  }
}

TEST(HwyTranscendentals, ExpSweep) {
  constexpr int kSteps = 256;
  const float delta = 20.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -10.0f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::exp(x); },
        input,
        5);
  }
}

TEST(HwyTranscendentals, LogSweep) {
  constexpr int kSteps = 256;
  const float delta = 1000.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = 0.01f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::log(x); },
        input,
        2);
  }
}

// ---- frexp / ldexp ----

TEST(HwyTranscendentals, Frexp) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.0f, 2.0f, 0.5f, 100.0f, 0.01f, -3.0f, 1024.0f, 0.125f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);

  HwyInt32 exp_out;
  HwyFloat frac = dfm::frexp(input, &exp_out);

  for (size_t i = 0; i < N(); ++i) {
    int32_t scalar_exp;
    float scalar_frac = dfm::frexp(vals[i], &scalar_exp);
    EXPECT_EQ(lane(frac, i), scalar_frac) << "Lane " << i << " input=" << vals[i];
    EXPECT_EQ(lane(exp_out, i), scalar_exp) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyFrexp, SpecialValues) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      -inf,
      -0.0f,
      std::numeric_limits<float>::min(),
      1.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  HwyInt32 exp_out;
  HwyFloat frac = dfm::frexp(input, &exp_out);

  for (size_t i = 0; i < N(); ++i) {
    int32_t scalar_exp;
    float scalar_frac = dfm::frexp(vals[i], &scalar_exp);
    if (std::isnan(scalar_frac)) {
      EXPECT_TRUE(std::isnan(lane(frac, i))) << "Lane " << i << " input=" << vals[i];
    } else {
      EXPECT_EQ(lane(frac, i), scalar_frac) << "Lane " << i << " input=" << vals[i];
      EXPECT_EQ(lane(exp_out, i), scalar_exp) << "Lane " << i << " input=" << vals[i];
    }
  }
}

TEST(HwyTranscendentals, Ldexp) {
  HWY_ALIGN float frac_vals[kMaxLanes];
  HWY_ALIGN int32_t exp_vals[kMaxLanes];
  float test_fracs[] = {0.5f, 0.75f, 0.625f, 0.5f};
  int32_t test_exps[] = {2, 3, 1, 10};

  for (size_t i = 0; i < N(); ++i) {
    frac_vals[i] = test_fracs[i % 4];
    exp_vals[i] = test_exps[i % 4];
  }
  HwyFloat frac = loadF(frac_vals);
  HwyInt32 exp_in = loadI(exp_vals);

  HwyFloat result = dfm::ldexp(frac, exp_in);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::ldexp(frac_vals[i], exp_vals[i]);
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(HwyLdexp, SpecialValues) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float frac_vals[kMaxLanes];
  HWY_ALIGN int32_t exp_vals[kMaxLanes];
  float test_fracs[] = {inf, -inf, nan, 0.0f, -0.0f, 1.0f, -1.0f, 0.5f};
  int32_t test_exps[] = {2, 1, 7, 0, 3, 0, 5, 10};
  for (size_t i = 0; i < N(); ++i) {
    frac_vals[i] = test_fracs[i % 8];
    exp_vals[i] = test_exps[i % 8];
  }
  HwyFloat frac = loadF(frac_vals);
  HwyInt32 exp_in = loadI(exp_vals);
  HwyFloat result = dfm::ldexp(frac, exp_in);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::ldexp(frac_vals[i], exp_vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << frac_vals[i];
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << frac_vals[i];
    }
  }
}

// ---- atan2 ----

TEST(HwyTranscendentals, Atan2) {
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.0f};
  float test_xs[] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 2.0f, -2.0f, 0.1f};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);

  HwyFloat result = dfm::atan2(y, x);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << y_vals[i] << " x=" << x_vals[i]
                        << " expected=" << expected << " actual=" << actual;
  }
}

// ---- Edge cases ----

TEST(HwyEdgeCases, NaN) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  HwyFloat nan_vec(nan);

  auto sin_result = dfm::sin(nan_vec);
  auto exp_result = dfm::exp(nan_vec);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::isnan(lane(sin_result, i))) << "sin(NaN) lane " << i;
    EXPECT_TRUE(std::isnan(lane(exp_result, i))) << "exp(NaN) lane " << i;
  }
}

// Note: No Infinity edge case test. fast_math functions use DefaultAccuracyTraits
// (kBoundsValues=false) which does not bound-check inf/NaN. Results for non-normal
// inputs are undefined. To get defined inf/NaN behavior, use MaxAccuracyTraits.

TEST(HwyEdgeCases, MixedValues) {
  // Different value per lane to verify no cross-lane contamination.
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f * static_cast<float>(i + 1);
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::sin(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = ::sinf(vals[i]);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 2u) << "Lane " << i << ": input=" << vals[i] << " expected=" << expected
                        << " actual=" << actual;
  }
}

// ---- Accuracy and bounds trait variants ----

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// -- sin/cos accurate --

TEST(HwySinAccurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::sin<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

TEST(HwySinAccurate, Sweep) {
  constexpr int kSteps = 256;
  const float delta = static_cast<float>(2.0 * M_PI / kSteps);
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = static_cast<float>(-M_PI) + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::sin<HwyVecF, dfm::MaxAccuracyTraits>(x); },
        input,
        2);
  }
}

TEST(HwyCosAccurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::cos<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

// -- exp accurate and bounds --

TEST(HwyExpAccurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      5);
}

TEST(HwyExpAccurate, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 20.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -10.0f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::exp<HwyVecF, dfm::MaxAccuracyTraits>(x); },
        input,
        5);
  }
}

TEST(HwyExpBounds, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp<HwyVecF, BoundsTraits>(x); },
      input,
      5);
}

TEST(HwyExpBounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {nan, inf, -inf, -100.0f, 89.0f, 100.0f, -100.0f, 0.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp<HwyVecF, BoundsTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp<float, BoundsTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// -- exp2 bounds --

TEST(HwyExp2Bounds, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp2<HwyVecF, BoundsTraits>(x); },
      input,
      1);
}

TEST(HwyExp2Bounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {nan, 128.0f, -150.0f, 0.0f, 1.0f, -1.0f, 127.0f, -126.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp2<HwyVecF, BoundsTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp2<float, BoundsTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// -- exp10 bounds --

TEST(HwyExp10Bounds, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = -3.0f + 6.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::exp10<HwyVecF, BoundsTraits>(x); },
      input,
      3);
}

TEST(HwyExp10Bounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {nan, 39.0f, -39.0f, 0.0f, 1.0f, -1.0f, 38.0f, -38.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp10<HwyVecF, BoundsTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp10<float, BoundsTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// -- log accurate --

TEST(HwyLogAccurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 10.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::log<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

TEST(HwyLogAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      0.5f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(HwyLogAccurate, NegativeInputs) {
  // log(negative) and log(-inf) must return NaN, matching scalar behavior.
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-1.0f, -100.0f, -inf, -0.0f, -0.5f, -1e10f, -1e-30f, -42.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

// -- log2 accurate --

TEST(HwyLog2Accurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 100.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::log2<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      1);
}

TEST(HwyLog2Accurate, NegativeInputs) {
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-1.0f, -100.0f, -inf, -0.0f, -0.5f, -1e10f, -1e-30f, -42.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log2<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log2<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

TEST(HwyLog10Accurate, NegativeInputs) {
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-1.0f, -100.0f, -inf, -0.0f, -0.5f, -1e10f, -1e-30f, -42.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log10<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log10<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

// -- cbrt accurate --

TEST(HwyCbrtAccurate, LaneByLane) {
  HWY_ALIGN float vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = 0.1f + 100.0f * static_cast<float>(i) / static_cast<float>(N());
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::cbrt<HwyVecF, dfm::MaxAccuracyTraits>(x); },
      input,
      3);
}

TEST(HwyCbrtAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f, inf, nan, std::numeric_limits<float>::denorm_min(), 1.0f, 1000.0f, 0.001f, 100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::cbrt<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// -- atan2 bounds --

TEST(HwyAtan2Bounds, LaneByLane) {
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.0f};
  float test_xs[] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 2.0f, -2.0f, 0.1f};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);
  auto result = dfm::atan2<HwyVecF, dfm::MaxAccuracyTraits>(y, x);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i;
  }
}

TEST(HwyAtan2Bounds, InfCases) {
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {inf, -inf, inf, -inf, 1.0f, -1.0f, inf, 0.0f};
  float test_xs[] = {inf, inf, -inf, -inf, inf, -inf, 1.0f, inf};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);
  auto result = dfm::atan2<HwyVecF, dfm::MaxAccuracyTraits>(y, x);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << y_vals[i] << " x=" << x_vals[i]
                        << " expected=" << expected << " actual=" << actual;
  }
}

// ---- Mixed special values: different edge cases in different lanes ----
// These tests catch bugs in per-lane conditional/masking logic.

TEST(HwySin, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {kNaN, kInf, -kInf, 1.0f, -1.0f, 0.0f, 100.0f, -100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::sin(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = ::sinf(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 2u) << "Lane " << i << " input=" << vals[i] << " expected=" << expected
                          << " actual=" << actual;
    }
  }
}

TEST(HwyCos, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {0.0f, kNaN, kInf, -kInf, 1.0f, -1.0f, 100.0f, -100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::cos(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = ::cosf(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 2u) << "Lane " << i << " input=" << vals[i] << " expected=" << expected
                          << " actual=" << actual;
    }
  }
}

TEST(HwyExp, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  // Bounds-checked: NaN, +Inf→+Inf, -Inf→0, overflow, underflow, normal
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {kNaN, kInf, -kInf, 89.0f, -100.0f, 0.0f, -0.0f, 1.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp<HwyVecF, BoundsTraits>(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp<float, BoundsTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i << " input=" << vals[i];
      EXPECT_EQ(std::signbit(expected), std::signbit(actual));
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << " input=" << vals[i] << " expected=" << expected
                          << " actual=" << actual;
    }
  }
}

TEST(HwyLog, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  // Note: log(negative) returns NaN for scalar but is not guaranteed for SIMD —
  // the bounds-check logic doesn't explicitly detect negative inputs.
  // Test only the guaranteed edge cases: 0 → -Inf, +Inf → +Inf, NaN → NaN, denorm.
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f,
      std::numeric_limits<float>::denorm_min(),
      kInf,
      kNaN,
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log<HwyVecF, dfm::MaxAccuracyTraits>(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i << " input=" << vals[i];
      EXPECT_EQ(std::signbit(expected), std::signbit(actual));
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << " input=" << vals[i] << " expected=" << expected
                          << " actual=" << actual;
    }
  }
}

TEST(HwyCbrt, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min(), 8.0f, 27.0f, 0.001f, 1e20f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::cbrt<HwyVecF, dfm::MaxAccuracyTraits>(input);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << " input=" << vals[i] << " expected=" << expected
                          << " actual=" << actual;
    }
  }
}

// ---- Domain boundary tests ----

TEST(HwyAsin, DomainBoundaries) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-1.0f, 1.0f, 0.0f, 0.99999f, -0.99999f, 0.5f, -0.5f, 0.0001f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::asin(x); },
      input,
      4);
}

TEST(HwyAcos, DomainBoundaries) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-1.0f, 1.0f, 0.0f, -0.5f, 0.5f, 0.99999f, -0.99999f, 0.0001f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::acos(x); },
      input,
      4);
}

TEST(HwyAtan2, NegativeZero) {
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 1.0f, -1.0f};
  float test_xs[] = {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);
  auto result = dfm::atan2(y, x);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << y_vals[i] << " x=" << x_vals[i] << " expected=" << expected
        << " actual=" << actual;
  }
}

TEST(HwyAtan2, ZeroZero) {
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {0.0f, -0.0f, 0.0f, -0.0f, 1.0f, -1.0f, 0.0f, -0.0f};
  float test_xs[] = {0.0f, 0.0f, -0.0f, -0.0f, 1.0f, -1.0f, 1.0f, -1.0f};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);
  auto result = dfm::atan2(y, x);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << y_vals[i] << " x=" << x_vals[i] << " expected=" << expected
        << " actual=" << actual;
  }
}

TEST(HwyAtan2, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_ys[] = {kNaN, 1.0f, kInf, -kInf, 1.0f, -1.0f, kInf, 0.0f};
  float test_xs[] = {1.0f, kNaN, kInf, -kInf, kInf, -kInf, 1.0f, kInf};

  for (size_t i = 0; i < N(); ++i) {
    y_vals[i] = test_ys[i % 8];
    x_vals[i] = test_xs[i % 8];
  }
  HwyVecF y = loadF(y_vals);
  HwyVecF x = loadF(x_vals);
  auto result = dfm::atan2(y, x);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan2(y_vals[i], x_vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << y_vals[i] << " x=" << x_vals[i]
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(HwyTan, MixedSpecialValues) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {0.0f, nan, inf, -inf, 1.0f, -1.0f, 0.5f, -0.5f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::tan(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::tan(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << vals[i];
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(HwyAtan, SpecialValues) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {0.0f, nan, inf, -inf, 20000000.0f, -20000000.0f, 1e10f, -1e10f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::atan(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::atan(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << vals[i];
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(HwyAsin, OutOfRange) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::asin(input);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyAcos, OutOfRange) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::acos(input);
  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << vals[i];
  }
}

TEST(HwyLog2Accurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      0.5f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log2<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log2<float, dfm::MaxAccuracyTraits>(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(HwyLog10Accurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      10.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::log10<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::log10<float, dfm::MaxAccuracyTraits>(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(HwyCbrtAccurate, NegativeInf) {
  float inf = std::numeric_limits<float>::infinity();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {-inf, -8.0f, -27.0f, -1.0f, -0.001f, -1e10f, -1e-20f, -125.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::cbrt<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(vals[i]);
    if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 0u) << "Lane " << i << " input=" << vals[i];
    }
  }
}

TEST(HwyExp2Bounds, InfInputs) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {inf, -inf, nan, 0.0f, 128.0f, -150.0f, 1.0f, -1.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp2<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp2<float, dfm::MaxAccuracyTraits>(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(HwyExp10Bounds, InfInputs) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {inf, -inf, nan, 0.0f, 39.0f, -39.0f, 1.0f, -1.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  auto result = dfm::exp10<HwyVecF, dfm::MaxAccuracyTraits>(input);
  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::exp10<float, dfm::MaxAccuracyTraits>(vals[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// ---- Broader sweep tests for functions with thin coverage ----

TEST(HwyAtan, Sweep) {
  constexpr int kSteps = 512;
  const float delta = 20.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -10.0f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::atan(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::atan(x); },
        input,
        3);
  }
}

TEST(HwyAsin, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 1.98f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -0.99f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::asin(x); },
        input,
        4);
  }
}

TEST(HwyAcos, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 1.98f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -0.99f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::acos(x); },
        input,
        4);
  }
}

TEST(HwyCbrt, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 100.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = 0.01f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::cbrt(x); },
        input,
        12);
  }
}

TEST(HwyExp2, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 20.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -10.0f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::exp2(x); },
        input,
        1);
  }
}

TEST(HwyExp10, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 10.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -5.0f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::exp10(x); },
        input,
        3);
  }
}

TEST(HwyLog2, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 1000.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = 0.01f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::log2(x); },
        input,
        1);
  }
}

TEST(HwyLog10, Sweep) {
  constexpr int kSteps = 256;
  const float delta = 1000.0f / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = 0.01f + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log10(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::log10(x); },
        input,
        3);
  }
}

TEST(HwyTan, Sweep) {
  constexpr int kSteps = 256;
  const float range = static_cast<float>(M_PI / 2 - 0.1);
  const float delta = 2.0f * range / kSteps;
  const size_t n = N();
  HWY_ALIGN float vals[kMaxLanes];

  for (int step = 0; step < kSteps; step += static_cast<int>(n)) {
    for (size_t i = 0; i < n && (step + static_cast<int>(i)) < kSteps; ++i) {
      vals[i] = -range + static_cast<float>(step + i) * delta;
    }
    HwyVecF input = loadF(vals);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
        [](HwyVecF x) { return dfm::tan(x); },
        input,
        3);
  }
}

TEST(HwyAtan2, Sweep) {
  const size_t n = N();
  HWY_ALIGN float y_vals[kMaxLanes];
  HWY_ALIGN float x_vals[kMaxLanes];

  // Sweep all four quadrants with varying magnitudes.
  float test_ys[] = {-5.0f, -2.0f, -1.0f, -0.1f, 0.1f, 1.0f, 2.0f, 5.0f};
  float test_xs[] = {-5.0f, -1.0f, 0.5f, 2.0f, -2.0f, -0.5f, 1.0f, 5.0f};

  for (float yscale = 0.01f; yscale <= 100.0f; yscale *= 10.0f) {
    for (size_t i = 0; i < n; ++i) {
      y_vals[i] = test_ys[i % 8] * yscale;
      x_vals[i] = test_xs[i % 8] * yscale;
    }
    HwyFloat y = loadF(y_vals);
    HwyFloat x = loadF(x_vals);
    auto result = dfm::atan2(y, x);

    for (size_t i = 0; i < n; ++i) {
      float expected = static_cast<float>(
          ::atan2(static_cast<double>(y_vals[i]), static_cast<double>(x_vals[i])));
      float actual = lane(result, i);
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 3u) << "Lane " << i << " y=" << y_vals[i] << " x=" << x_vals[i]
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

// ---- Large magnitude trig tests ----

TEST(HwySin, LargeMagnitude) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {100.0f, -100.0f, 1000.0f, -1000.0f, 10000.0f, -10000.0f, 50.0f, -50.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::sin(x); },
      input,
      2);
}

TEST(HwyCos, LargeMagnitude) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {100.0f, -100.0f, 1000.0f, -1000.0f, 10000.0f, -10000.0f, 50.0f, -50.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::cos(x); },
      input,
      2);
}

TEST(HwyTan, LargeMagnitude) {
  HWY_ALIGN float vals[kMaxLanes];
  float test_vals[] = {50.0f, -50.0f, 200.0f, -200.0f, 500.0f, -500.0f, 30.0f, -30.0f};
  for (size_t i = 0; i < N(); ++i) {
    vals[i] = test_vals[i % 8];
  }
  HwyVecF input = loadF(vals);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
      [](HwyVecF x) { return dfm::tan(x); },
      input,
      3);
}

// --- hypot ---

TEST(HwyTranscendentals, Hypot) {
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {3.0f, 0.0f, 1e30f, -5.0f, 1.0f, 1e-20f, -1e15f, 7.0f};
  float test_ys[] = {4.0f, 1.0f, 1e30f, 12.0f, 1.0f, 1e-20f, 1e15f, 24.0f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::hypot(x, y);

  for (size_t i = 0; i < N(); ++i) {
    double xd = static_cast<double>(x_vals[i]);
    double yd = static_cast<double>(y_vals[i]);
    float expected = static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 2u) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i]
                        << " expected=" << expected << " actual=" << actual;
  }
}

// -- hypot bounds --

TEST(HwyHypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float test_xs[] = {inf, -inf, nan, nan, inf, 3.0f, -inf, nan};
  float test_ys[] = {3.0f, nan, inf, -inf, inf, inf, 0.0f, -inf};
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);
  HwyFloat result = dfm::hypot<HwyVecF, dfm::MaxAccuracyTraits>(x, y);
  for (size_t i = 0; i < N(); ++i) {
    float actual = lane(result, i);
    EXPECT_TRUE(std::isinf(actual) && actual > 0)
        << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i] << " result=" << actual;
  }
}

TEST(HwyHypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float test_xs[] = {nan, 3.0f, nan, nan, 0.0f, nan, -1.0f, nan};
  float test_ys[] = {3.0f, nan, 0.0f, nan, nan, -5.0f, nan, nan};
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);
  HwyFloat result = dfm::hypot<HwyVecF, dfm::MaxAccuracyTraits>(x, y);
  for (size_t i = 0; i < N(); ++i) {
    float actual = lane(result, i);
    EXPECT_TRUE(std::isnan(actual))
        << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i] << " result=" << actual;
  }
}

// ---- pow ----

static float gt_pow(float x, float y) {
  return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

TEST(HwyPow, Basic) {
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {2.0f, 3.0f, 10.0f, 0.5f, 1.0f, 100.0f, 0.1f, 7.0f};
  float test_ys[] = {3.0f, 2.0f, 0.5f, 4.0f, 100.0f, 0.25f, 3.0f, 1.5f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::pow(x, y);

  for (size_t i = 0; i < N(); ++i) {
    float expected = gt_pow(x_vals[i], y_vals[i]);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 4u) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i]
                        << " expected=" << expected << " actual=" << actual;
  }
}

TEST(HwyPow, NegativeBase) {
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {-2.0f, -3.0f, -1.0f, -10.0f, -0.5f, -4.0f, -2.0f, -5.0f};
  float test_ys[] = {3.0f, 2.0f, 5.0f, 4.0f, 3.0f, 2.0f, 4.0f, 1.0f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::pow(x, y);

  for (size_t i = 0; i < N(); ++i) {
    float expected = gt_pow(x_vals[i], y_vals[i]);
    float actual = lane(result, i);
    // Check sign correctness.
    EXPECT_EQ(std::signbit(expected), std::signbit(actual))
        << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i] << " expected=" << expected
        << " actual=" << actual;
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 3u) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i]
                        << " expected=" << expected << " actual=" << actual;
  }
}

TEST(HwyPow, NegBaseNonInt) {
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {-2.0f, -3.0f, -1.0f, -10.0f, -0.5f, -4.0f, -2.0f, -5.0f};
  float test_ys[] = {0.5f, 1.5f, 2.3f, 0.1f, 0.7f, 3.3f, -0.5f, 1.1f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::pow(x, y);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": pow(" << x_vals[i] << ", " << y_vals[i] << ") should be NaN, got "
        << lane(result, i);
  }
}

TEST(HwyPow, YZero) {
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {2.0f, -3.0f, 0.0f, 100.0f, -1.0f, 0.5f, 42.0f, -0.1f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = 0.0f;
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::pow(x, y);

  for (size_t i = 0; i < N(); ++i) {
    EXPECT_EQ(lane(result, i), 1.0f) << "Lane " << i << ": pow(" << x_vals[i] << ", 0) should be 1";
  }
}

TEST(HwyPow, ScalarExp) {
  HWY_ALIGN float x_vals[kMaxLanes];
  float test_xs[] = {2.0f, 3.0f, 0.5f, 10.0f, 1.0f, 4.0f, 0.1f, 7.0f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
  }
  HwyFloat x = loadF(x_vals);

  HwyFloat result = dfm::pow(x, 2.0f);

  for (size_t i = 0; i < N(); ++i) {
    float expected = gt_pow(x_vals[i], 2.0f);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 4u) << "Lane " << i << ": x=" << x_vals[i] << " expected=" << expected
                        << " actual=" << actual;
  }
}

TEST(HwyPowBounds, Specials) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  HWY_ALIGN float x_vals[kMaxLanes];
  HWY_ALIGN float y_vals[kMaxLanes];
  float test_xs[] = {0.0f, -0.0f, inf, nan, 1.0f, -1.0f, inf, -inf};
  float test_ys[] = {2.0f, 3.0f, 2.0f, 5.0f, nan, 2.0f, -1.0f, 3.0f};

  for (size_t i = 0; i < N(); ++i) {
    x_vals[i] = test_xs[i % 8];
    y_vals[i] = test_ys[i % 8];
  }
  HwyVecF x = loadF(x_vals);
  HwyVecF y = loadF(y_vals);

  HwyFloat result = dfm::pow<HwyVecF, dfm::MaxAccuracyTraits>(x, y);

  for (size_t i = 0; i < N(); ++i) {
    float expected = dfm::pow<float, dfm::MaxAccuracyTraits>(x_vals[i], y_vals[i]);
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i];
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i];
      EXPECT_EQ(std::signbit(expected), std::signbit(actual))
          << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i];
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": x=" << x_vals[i] << " y=" << y_vals[i]
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

#else // !__has_include("hwy/highway.h")

// Dummy test so the binary does something when Highway is not available.
TEST(HwyFloat, Unavailable) {
  GTEST_SKIP() << "Highway not available";
}

#endif // __has_include("hwy/highway.h")
