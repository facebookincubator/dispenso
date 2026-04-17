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

#else // !__has_include("hwy/highway.h")

// Dummy test so the binary does something when Highway is not available.
TEST(HwyFloat, Unavailable) {
  GTEST_SKIP() << "Highway not available";
}

#endif // __has_include("hwy/highway.h")
