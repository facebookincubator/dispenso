/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <limits>

#include <dispenso/fast_math/fast_math.h>
#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;
using NeonFloat = dfm::NeonFloat;
using NeonInt32 = dfm::NeonInt32;
using NeonUint32 = dfm::NeonUint32;

constexpr int kLanes = 4;

// --- Lane extractors ---

static float lane(float32x4_t v, int i) {
  alignas(16) float buf[kLanes];
  vst1q_f32(buf, v);
  return buf[i];
}
static float lane(NeonFloat v, int i) {
  return lane(v.v, i);
}

static int32_t lane(int32x4_t v, int i) {
  alignas(16) int32_t buf[kLanes];
  vst1q_s32(buf, v);
  return buf[i];
}
static int32_t lane(NeonInt32 v, int i) {
  return lane(v.v, i);
}

static uint32_t lane(uint32x4_t v, int i) {
  alignas(16) uint32_t buf[kLanes];
  vst1q_u32(buf, v);
  return buf[i];
}
static uint32_t lane(NeonUint32 v, int i) {
  return lane(v.v, i);
}

// --- Helpers ---

static float32x4_t make4(float a, float b, float c, float d) {
  float buf[4] = {a, b, c, d};
  return vld1q_f32(buf);
}

static int32x4_t makeInt4(int32_t a, int32_t b, int32_t c, int32_t d) {
  int32_t buf[4] = {a, b, c, d};
  return vld1q_s32(buf);
}

// Check that a SIMD function matches the scalar float version lane-by-lane.
template <typename SimdFn, typename ScalarFn>
static void checkLaneByLane(
    float32x4_t input,
    SimdFn simdFn,
    ScalarFn scalarFn,
    const char* name,
    uint32_t max_ulps = 0) {
  auto result = simdFn(input);
  for (int i = 0; i < kLanes; ++i) {
    float scalarResult = scalarFn(lane(input, i));
    float rLane = lane(result, i);
    if (std::isnan(scalarResult)) {
      EXPECT_TRUE(std::isnan(rLane)) << name << " lane " << i << ": expected NaN, got " << rLane;
    } else {
      uint32_t dist = dfm::float_distance(scalarResult, rLane);
      EXPECT_LE(dist, max_ulps) << name << " lane " << i << ": input=" << lane(input, i)
                                << " simd=" << rLane << " scalar=" << scalarResult
                                << " ulps=" << dist;
    }
  }
}

// ==================== Basic Arithmetic ====================

TEST(NeonFloat, Arithmetic) {
  NeonFloat a(3.0f), b(2.0f);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a + b, i), 5.0f);
    EXPECT_EQ(lane(a - b, i), 1.0f);
    EXPECT_EQ(lane(a * b, i), 6.0f);
    EXPECT_EQ(lane(a / b, i), 1.5f);
  }
}

TEST(NeonFloat, Negation) {
  NeonFloat a = make4(1.0f, -2.0f, 0.0f, 3.5f);
  NeonFloat neg = -a;
  EXPECT_EQ(lane(neg, 0), -1.0f);
  EXPECT_EQ(lane(neg, 1), 2.0f);
  EXPECT_FLOAT_EQ(lane(neg, 2), -0.0f);
  EXPECT_EQ(lane(neg, 3), -3.5f);
}

TEST(NeonFloat, CompoundAssignment) {
  NeonFloat a(10.0f), b(3.0f);
  a += b;
  for (int i = 0; i < kLanes; ++i)
    EXPECT_EQ(lane(a, i), 13.0f);
  a -= b;
  for (int i = 0; i < kLanes; ++i)
    EXPECT_EQ(lane(a, i), 10.0f);
  a *= b;
  for (int i = 0; i < kLanes; ++i)
    EXPECT_EQ(lane(a, i), 30.0f);
}

TEST(NeonInt32, Arithmetic) {
  NeonInt32 a(7), b(3);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a + b, i), 10);
    EXPECT_EQ(lane(a - b, i), 4);
    EXPECT_EQ(lane(a * b, i), 21);
  }
}

TEST(NeonInt32, Negation) {
  NeonInt32 a = makeInt4(5, -3, 0, 100);
  NeonInt32 neg = -a;
  EXPECT_EQ(lane(neg, 0), -5);
  EXPECT_EQ(lane(neg, 1), 3);
  EXPECT_EQ(lane(neg, 2), 0);
  EXPECT_EQ(lane(neg, 3), -100);
}

TEST(NeonInt32, Shifts) {
  NeonInt32 a(16);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a << 2, i), 64);
    EXPECT_EQ(lane(a >> 2, i), 4);
  }
  // Arithmetic right shift preserves sign.
  NeonInt32 neg(-16);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(neg >> 2, i), -4);
  }
}

TEST(NeonUint32, Shifts) {
  NeonUint32 a(16u);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a << 2, i), 64u);
    EXPECT_EQ(lane(a >> 2, i), 4u);
  }
  // Logical right shift (no sign extension).
  NeonUint32 big(0x80000000u);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(big >> 1, i), 0x40000000u);
  }
}

// ==================== Comparisons ====================

TEST(NeonFloat, Comparisons) {
  NeonFloat a = make4(1.0f, 2.0f, 3.0f, 4.0f);
  NeonFloat b = make4(4.0f, 3.0f, 3.0f, 1.0f);

  NeonFloat lt = a < b;
  EXPECT_NE(lane(NeonInt32(bit_cast<NeonInt32>(lt)), 0), 0); // 1 < 4
  EXPECT_NE(lane(NeonInt32(bit_cast<NeonInt32>(lt)), 1), 0); // 2 < 3
  EXPECT_EQ(lane(NeonInt32(bit_cast<NeonInt32>(lt)), 2), 0); // 3 == 3
  EXPECT_EQ(lane(NeonInt32(bit_cast<NeonInt32>(lt)), 3), 0); // 4 > 1
}

TEST(NeonInt32, Comparisons) {
  NeonInt32 a = makeInt4(1, 5, -3, 0);
  NeonInt32 b = makeInt4(2, 5, 0, -1);

  NeonInt32 lt = a < b;
  EXPECT_NE(lane(lt, 0), 0); // 1 < 2
  EXPECT_EQ(lane(lt, 1), 0); // 5 == 5
  EXPECT_NE(lane(lt, 2), 0); // -3 < 0
  EXPECT_EQ(lane(lt, 3), 0); // 0 > -1

  NeonInt32 eq = a == b;
  EXPECT_EQ(lane(eq, 0), 0);
  EXPECT_NE(lane(eq, 1), 0);
  EXPECT_EQ(lane(eq, 2), 0);
  EXPECT_EQ(lane(eq, 3), 0);
}

TEST(NeonUint32, UnsignedComparisons) {
  NeonUint32 a(0x80000001u); // Large unsigned
  NeonUint32 b(1u);
  // a > b in unsigned (but would be negative in signed).
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_NE(lane(NeonUint32(a > b), i), 0u);
    EXPECT_EQ(lane(NeonUint32(a < b), i), 0u);
  }
}

// ==================== bit_cast ====================

TEST(NeonBitCast, FloatIntRoundTrip) {
  NeonFloat f = make4(1.0f, -2.0f, 0.0f, 3.14f);
  NeonInt32 fi = dfm::bit_cast<NeonInt32>(f);
  NeonFloat back = dfm::bit_cast<NeonFloat>(fi);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(back, i), lane(f, i));
  }
}

TEST(NeonBitCast, FloatUintRoundTrip) {
  NeonFloat f = make4(1.0f, -2.0f, 0.0f, 3.14f);
  NeonUint32 fu = dfm::bit_cast<NeonUint32>(f);
  NeonFloat back = dfm::bit_cast<NeonFloat>(fu);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(back, i), lane(f, i));
  }
}

TEST(NeonBitCast, IntUintRoundTrip) {
  NeonInt32 a = makeInt4(1, -1, 0, 0x7FFFFFFF);
  NeonUint32 au = dfm::bit_cast<NeonUint32>(a);
  NeonInt32 back = dfm::bit_cast<NeonInt32>(au);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(back, i), lane(a, i));
  }
}

// ==================== FloatTraits ====================

TEST(NeonFloatTraits, ConditionalWithMask) {
  // Even lanes true, odd lanes false via float mask.
  uint32_t mask_bits[4] = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
  NeonFloat mask = vreinterpretq_f32_u32(vld1q_u32(mask_bits));
  NeonFloat x(10.0f);
  NeonFloat y(20.0f);
  NeonFloat result = dfm::FloatTraits<NeonFloat>::conditional(mask, x, y);
  EXPECT_EQ(lane(result, 0), 10.0f);
  EXPECT_EQ(lane(result, 1), 20.0f);
  EXPECT_EQ(lane(result, 2), 10.0f);
  EXPECT_EQ(lane(result, 3), 20.0f);
}

TEST(NeonFloatTraits, ConditionalFromComparison) {
  NeonFloat a = make4(1.0f, 5.0f, 10.0f, 15.0f);
  NeonFloat threshold(8.0f);
  auto mask = a > threshold;
  NeonFloat result = dfm::FloatTraits<NeonFloat>::conditional(mask, NeonFloat(100.0f), a);
  EXPECT_EQ(lane(result, 0), 1.0f);
  EXPECT_EQ(lane(result, 1), 5.0f);
  EXPECT_EQ(lane(result, 2), 100.0f);
  EXPECT_EQ(lane(result, 3), 100.0f);
}

TEST(NeonFloatTraits, ConditionalInt32WithMask) {
  uint32_t mask_bits[4] = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
  NeonFloat mask = vreinterpretq_f32_u32(vld1q_u32(mask_bits));
  NeonInt32 x(100);
  NeonInt32 y(200);
  NeonInt32 result = dfm::FloatTraits<NeonFloat>::conditional(mask, x, y);
  EXPECT_EQ(lane(result, 0), 100);
  EXPECT_EQ(lane(result, 1), 200);
  EXPECT_EQ(lane(result, 2), 100);
  EXPECT_EQ(lane(result, 3), 200);
}

TEST(NeonFloatTraits, ConditionalWithLaneWideMask) {
  // Test the NeonInt32 mask overload.
  NeonInt32 mask = makeInt4(-1, 0, -1, 0);
  NeonFloat x(10.0f);
  NeonFloat y(20.0f);
  NeonFloat result = dfm::FloatTraits<NeonFloat>::conditional(mask, x, y);
  EXPECT_EQ(lane(result, 0), 10.0f);
  EXPECT_EQ(lane(result, 1), 20.0f);
  EXPECT_EQ(lane(result, 2), 10.0f);
  EXPECT_EQ(lane(result, 3), 20.0f);
}

TEST(NeonFloatTraits, Apply) {
  uint32_t mask_bits[4] = {0, 0, 0xFFFFFFFF, 0xFFFFFFFF};
  NeonFloat mask = vreinterpretq_f32_u32(vld1q_u32(mask_bits));
  NeonFloat x(42.0f);
  NeonFloat result = dfm::FloatTraits<NeonFloat>::apply(mask, x);
  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_EQ(lane(result, 1), 0.0f);
  EXPECT_EQ(lane(result, 2), 42.0f);
  EXPECT_EQ(lane(result, 3), 42.0f);
}

TEST(NeonFloatTraits, Fma) {
  NeonFloat a(2.0f), b(3.0f), c(4.0f);
  NeonFloat result = dfm::FloatTraits<NeonFloat>::fma(a, b, c);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 10.0f) << "Lane " << i;
  }
}

TEST(NeonFloatTraits, Sqrt) {
  NeonFloat a = make4(4.0f, 9.0f, 16.0f, 25.0f);
  NeonFloat result = dfm::FloatTraits<NeonFloat>::sqrt(a);
  EXPECT_EQ(lane(result, 0), 2.0f);
  EXPECT_EQ(lane(result, 1), 3.0f);
  EXPECT_EQ(lane(result, 2), 4.0f);
  EXPECT_EQ(lane(result, 3), 5.0f);
}

TEST(NeonFloatTraits, MinMax) {
  NeonFloat a = make4(1.0f, 5.0f, 3.0f, 7.0f);
  NeonFloat b = make4(4.0f, 2.0f, 3.0f, 8.0f);
  NeonFloat mn = dfm::FloatTraits<NeonFloat>::min(a, b);
  NeonFloat mx = dfm::FloatTraits<NeonFloat>::max(a, b);
  EXPECT_EQ(lane(mn, 0), 1.0f);
  EXPECT_EQ(lane(mn, 1), 2.0f);
  EXPECT_EQ(lane(mn, 2), 3.0f);
  EXPECT_EQ(lane(mn, 3), 7.0f);
  EXPECT_EQ(lane(mx, 0), 4.0f);
  EXPECT_EQ(lane(mx, 1), 5.0f);
  EXPECT_EQ(lane(mx, 2), 3.0f);
  EXPECT_EQ(lane(mx, 3), 8.0f);
}

// ==================== Util Functions ====================

TEST(NeonUtil, FloorSmall) {
  float32x4_t a = make4(1.5f, -1.5f, 2.0f, -0.1f);
  auto result = dfm::floor_small(a);
  EXPECT_EQ(lane(result, 0), 1.0f);
  EXPECT_EQ(lane(result, 1), -2.0f);
  EXPECT_EQ(lane(result, 2), 2.0f);
  EXPECT_EQ(lane(result, 3), -1.0f);
}

TEST(NeonUtil, ConvertToInt) {
  float32x4_t a = make4(1.5f, -1.5f, 2.4f, 3.6f);
  auto result = dfm::convert_to_int(a);
  // Round-to-nearest-even: 1.5→2, -1.5→-2, 2.4→2, 3.6→4
  EXPECT_EQ(lane(result, 0), 2);
  EXPECT_EQ(lane(result, 1), -2);
  EXPECT_EQ(lane(result, 2), 2);
  EXPECT_EQ(lane(result, 3), 4);
}

TEST(NeonUtil, ConvertToIntNaN) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  float32x4_t a = make4(nan, inf, -inf, 5.0f);
  auto result = dfm::convert_to_int(a);
  EXPECT_EQ(lane(result, 0), 0); // NaN → 0
  EXPECT_EQ(lane(result, 1), 0); // Inf → 0
  EXPECT_EQ(lane(result, 2), 0); // -Inf → 0
  EXPECT_EQ(lane(result, 3), 5); // Normal
}

TEST(NeonUtil, Gather) {
  float table[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  int32x4_t index = makeInt4(3, 0, 7, 1);
  auto result = dfm::gather<NeonFloat>(table, index);
  EXPECT_EQ(lane(result, 0), 40.0f);
  EXPECT_EQ(lane(result, 1), 10.0f);
  EXPECT_EQ(lane(result, 2), 80.0f);
  EXPECT_EQ(lane(result, 3), 20.0f);
}

TEST(NeonUtil, IntDivBy3) {
  int32x4_t a = makeInt4(0, 3, 9, 99);
  auto result = dfm::int_div_by_3(a);
  int32_t inputs[] = {0, 3, 9, 99};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), inputs[i] / 3) << "Lane " << i << " input=" << inputs[i];
  }
}

TEST(NeonUtil, Signof) {
  float32x4_t a = make4(1.0f, -1.0f, 0.0f, -0.0f);
  auto result = dfm::signof(a);
  EXPECT_EQ(lane(result, 0), 1.0f);
  EXPECT_EQ(lane(result, 1), -1.0f);
  EXPECT_EQ(lane(result, 2), 1.0f);
  EXPECT_EQ(lane(result, 3), -1.0f);
}

TEST(NeonUtil, Signofi) {
  int32x4_t a = makeInt4(5, -3, 0, -100);
  auto result = dfm::signofi<NeonFloat>(a);
  EXPECT_EQ(lane(result, 0), 1);
  EXPECT_EQ(lane(result, 1), -1);
  EXPECT_EQ(lane(result, 2), 1);
  EXPECT_EQ(lane(result, 3), -1);
}

TEST(NeonUtil, Nonnormal) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t a = make4(1.0f, inf, nan, 0.0f);
  auto result = dfm::nonnormal(NeonFloat(a));
  EXPECT_EQ(lane(result, 0), 0); // Normal
  EXPECT_NE(lane(result, 1), 0); // Inf
  EXPECT_NE(lane(result, 2), 0); // NaN
  EXPECT_EQ(lane(result, 3), 0); // Zero
}

TEST(NeonUtil, NonnormalOrZero) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t a = make4(1.0f, inf, nan, 0.0f);
  NeonInt32 fi = dfm::bit_cast<NeonInt32>(NeonFloat(a));
  auto result = dfm::nonnormalOrZero<NeonFloat>(fi);
  EXPECT_EQ(lane(result, 0), 0); // Normal
  EXPECT_NE(lane(result, 1), 0); // Inf
  EXPECT_NE(lane(result, 2), 0); // NaN
  EXPECT_NE(lane(result, 3), 0); // Zero
}

TEST(NeonUtil, BoolAsOne) {
  NeonFloat mask = NeonFloat(1.0f) > NeonFloat(0.0f); // All true
  NeonFloat result = dfm::bool_as_one<NeonFloat, NeonFloat>(mask);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 1.0f);
  }

  NeonFloat false_mask = NeonFloat(0.0f) > NeonFloat(1.0f); // All false
  NeonFloat result2 = dfm::bool_as_one<NeonFloat, NeonFloat>(false_mask);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result2, i), 0.0f);
  }
}

TEST(NeonUtil, NboolAsOne) {
  NeonFloat mask = NeonFloat(1.0f) > NeonFloat(0.0f); // All true
  NeonFloat result = dfm::nbool_as_one<NeonFloat, NeonFloat>(mask);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 0.0f); // nbool: true → 0
  }

  NeonFloat false_mask = NeonFloat(0.0f) > NeonFloat(1.0f); // All false
  NeonFloat result2 = dfm::nbool_as_one<NeonFloat, NeonFloat>(false_mask);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result2, i), 1.0f); // nbool: false → 1
  }
}

TEST(NeonUtil, BoolAsMask) {
  NeonFloat mask = NeonFloat(1.0f) > NeonFloat(0.0f); // All true = 0xFFFFFFFF
  NeonInt32 result = dfm::bool_as_mask<NeonInt32, NeonFloat>(mask);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), -1);
  }
}

TEST(NeonUtil, BoolApplyOrZero) {
  NeonFloat mask = NeonFloat(make4(1.0f, 0.0f, 3.0f, 0.0f)) > NeonFloat(0.5f);
  NeonFloat val(42.0f);
  NeonFloat result = dfm::bool_apply_or_zero(mask, val);
  EXPECT_EQ(lane(result, 0), 42.0f);
  EXPECT_EQ(lane(result, 1), 0.0f);
  EXPECT_EQ(lane(result, 2), 42.0f);
  EXPECT_EQ(lane(result, 3), 0.0f);
}

TEST(NeonUtil, ClampAllowNan) {
  float32x4_t x = make4(-5.0f, 0.5f, 10.0f, 0.5f);
  auto result = dfm::clamp_allow_nan(NeonFloat(x), NeonFloat(0.0f), NeonFloat(1.0f));
  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_EQ(lane(result, 1), 0.5f);
  EXPECT_EQ(lane(result, 2), 1.0f);
  EXPECT_EQ(lane(result, 3), 0.5f);

  // NaN propagates.
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t nan_x = make4(nan, 0.5f, nan, 0.5f);
  auto nan_result = dfm::clamp_allow_nan(NeonFloat(nan_x), NeonFloat(0.0f), NeonFloat(1.0f));
  EXPECT_TRUE(std::isnan(lane(nan_result, 0)));
  EXPECT_EQ(lane(nan_result, 1), 0.5f);
  EXPECT_TRUE(std::isnan(lane(nan_result, 2)));
  EXPECT_EQ(lane(nan_result, 3), 0.5f);
}

TEST(NeonUtil, ClampNoNan) {
  float32x4_t x = make4(-5.0f, 0.5f, 10.0f, 0.5f);
  auto result = dfm::clamp_no_nan(NeonFloat(x), NeonFloat(0.0f), NeonFloat(1.0f));
  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_EQ(lane(result, 1), 0.5f);
  EXPECT_EQ(lane(result, 2), 1.0f);
  EXPECT_EQ(lane(result, 3), 0.5f);

  // NaN is suppressed — result must be in [mn, mx].
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t nan_x = make4(nan, 0.5f, nan, 0.5f);
  auto nan_result = dfm::clamp_no_nan(NeonFloat(nan_x), NeonFloat(0.0f), NeonFloat(1.0f));
  EXPECT_FALSE(std::isnan(lane(nan_result, 0)));
  EXPECT_GE(lane(nan_result, 0), 0.0f);
  EXPECT_LE(lane(nan_result, 0), 1.0f);
  EXPECT_EQ(lane(nan_result, 1), 0.5f);
  EXPECT_FALSE(std::isnan(lane(nan_result, 2)));
  EXPECT_EQ(lane(nan_result, 3), 0.5f);
}

// ==================== Transcendentals ====================

TEST(NeonTranscendentals, Sin) {
  float32x4_t input = make4(0.0f, 0.5f, 1.0f, -0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::sin(x); },
      [](float x) { return ::sinf(x); },
      "sin",
      2);
}

TEST(NeonTranscendentals, Cos) {
  float32x4_t input = make4(0.0f, 0.5f, 1.0f, -0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::cos(x); },
      [](float x) { return ::cosf(x); },
      "cos",
      2);
}

TEST(NeonTranscendentals, Tan) {
  float32x4_t input = make4(0.0f, 0.3f, 0.7f, -0.3f);
  checkLaneByLane(
      input, [](float32x4_t x) { return dfm::tan(x); }, [](float x) { return dfm::tan(x); }, "tan");
}

TEST(NeonTranscendentals, Exp) {
  float32x4_t input = make4(0.0f, 1.0f, -1.0f, 2.0f);
  checkLaneByLane(
      input, [](float32x4_t x) { return dfm::exp(x); }, [](float x) { return dfm::exp(x); }, "exp");
}

TEST(NeonTranscendentals, Exp2) {
  float32x4_t input = make4(0.0f, 1.0f, -1.0f, 3.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::exp2(x); },
      [](float x) { return dfm::exp2(x); },
      "exp2");
}

TEST(NeonTranscendentals, Exp10) {
  float32x4_t input = make4(0.0f, 1.0f, -1.0f, 2.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::exp10(x); },
      [](float x) { return dfm::exp10(x); },
      "exp10");
}

TEST(NeonTranscendentals, Log) {
  float32x4_t input = make4(1.0f, 2.0f, 10.0f, 0.5f);
  checkLaneByLane(
      input, [](float32x4_t x) { return dfm::log(x); }, [](float x) { return dfm::log(x); }, "log");
}

TEST(NeonTranscendentals, Log2) {
  float32x4_t input = make4(1.0f, 2.0f, 4.0f, 0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::log2(x); },
      [](float x) { return dfm::log2(x); },
      "log2");
}

TEST(NeonTranscendentals, Log10) {
  float32x4_t input = make4(1.0f, 10.0f, 100.0f, 0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::log10(x); },
      [](float x) { return dfm::log10(x); },
      "log10");
}

TEST(NeonTranscendentals, Acos) {
  float32x4_t input = make4(0.0f, 0.5f, -0.5f, 0.9f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::acos(x); },
      [](float x) { return dfm::acos(x); },
      "acos");
}

TEST(NeonTranscendentals, Asin) {
  float32x4_t input = make4(0.0f, 0.5f, -0.5f, 0.9f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::asin(x); },
      [](float x) { return dfm::asin(x); },
      "asin");
}

TEST(NeonTranscendentals, Atan) {
  float32x4_t input = make4(0.0f, 1.0f, -1.0f, 0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::atan(x); },
      [](float x) { return dfm::atan(x); },
      "atan");
}

TEST(NeonTranscendentals, Cbrt) {
  float32x4_t input = make4(1.0f, 8.0f, 27.0f, 100.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::cbrt(x); },
      [](float x) { return dfm::cbrt(x); },
      "cbrt");
}

// ==================== Sweep Tests ====================

TEST(NeonSweep, SinSweep) {
  constexpr int kSteps = 256;
  float delta = static_cast<float>(2.0 * M_PI / kSteps);
  float f = static_cast<float>(-M_PI);
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::sin(x); },
        [](float x) { return ::sinf(x); },
        "sin_sweep",
        2);
    f += kLanes * delta;
  }
}

TEST(NeonSweep, CosSweep) {
  constexpr int kSteps = 256;
  float delta = static_cast<float>(2.0 * M_PI / kSteps);
  float f = static_cast<float>(-M_PI);
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::cos(x); },
        [](float x) { return ::cosf(x); },
        "cos_sweep",
        2);
    f += kLanes * delta;
  }
}

TEST(NeonSweep, ExpSweep) {
  constexpr int kSteps = 256;
  float delta = 20.0f / kSteps;
  float f = -10.0f;
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::exp(x); },
        [](float x) { return dfm::exp(x); },
        "exp_sweep");
    f += kLanes * delta;
  }
}

TEST(NeonSweep, LogSweep) {
  constexpr int kSteps = 256;
  float delta = 10000.0f / kSteps;
  float f = 0.001f;
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::log(x); },
        [](float x) { return dfm::log(x); },
        "log_sweep");
    f += kLanes * delta;
  }
}

TEST(NeonSweep, AcosSweep) {
  constexpr int kSteps = 256;
  float delta = 1.998f / kSteps;
  float f = -0.999f;
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::acos(x); },
        [](float x) { return dfm::acos(x); },
        "acos_sweep");
    f += kLanes * delta;
  }
}

TEST(NeonSweep, AtanSweep) {
  constexpr int kSteps = 256;
  float delta = 20.0f / kSteps;
  float f = -10.0f;
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::atan(x); },
        [](float x) { return dfm::atan(x); },
        "atan_sweep");
    f += kLanes * delta;
  }
}

TEST(NeonSweep, CbrtSweep) {
  constexpr int kSteps = 256;
  float delta = 10000.0f / kSteps;
  float f = 0.001f;
  for (int step = 0; step < kSteps; step += kLanes) {
    float32x4_t input = make4(f, f + delta, f + 2 * delta, f + 3 * delta);
    checkLaneByLane(
        input,
        [](float32x4_t x) { return dfm::cbrt(x); },
        [](float x) { return dfm::cbrt(x); },
        "cbrt_sweep");
    f += kLanes * delta;
  }
}

// ==================== Frexp / Ldexp ====================

TEST(NeonTranscendentals, Frexp) {
  float32x4_t input = make4(1.0f, 4.0f, 0.5f, 100.0f);
  dfm::IntType_t<float32x4_t> e;
  float32x4_t frac = dfm::frexp(input, &e);
  for (int i = 0; i < kLanes; ++i) {
    int32_t se;
    float sf = dfm::frexp(lane(input, i), &se);
    EXPECT_EQ(lane(frac, i), sf) << "Frac lane " << i;
    EXPECT_EQ(lane(e, i), se) << "Exp lane " << i;
  }
}

TEST(NeonTranscendentals, Ldexp) {
  float32x4_t frac = make4(0.5f, 0.75f, 0.625f, -0.5f);
  int32x4_t exp = makeInt4(2, 3, 4, 1);
  auto result = dfm::ldexp(frac, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(lane(frac, i), lane(exp, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

// ==================== Atan2 ====================

TEST(NeonTranscendentals, Atan2) {
  float32x4_t y = make4(1.0f, -1.0f, 1.0f, 0.0f);
  float32x4_t x = make4(1.0f, 1.0f, -1.0f, 1.0f);
  auto result = dfm::atan2(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    uint32_t rBits, eBits;
    float rLane = lane(result, i);
    memcpy(&rBits, &rLane, 4);
    memcpy(&eBits, &expected, 4);
    EXPECT_EQ(rBits, eBits) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
  }
}

// ==================== Edge Cases ====================

TEST(NeonEdgeCases, SinEdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(0.0f, nan, inf, -inf);
  auto result = dfm::sin(input);
  EXPECT_NEAR(lane(result, 0), ::sinf(0.0f), 1e-7f);
}

TEST(NeonEdgeCases, ExpEdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(0.0f, -100.0f, 100.0f, nan);
  auto result = dfm::exp(input);
  EXPECT_FLOAT_EQ(lane(result, 0), 1.0f);
}

TEST(NeonEdgeCases, LogEdgeCases) {
  float32x4_t input = make4(1.0f, 0.0f, -1.0f, std::numeric_limits<float>::infinity());
  auto result = dfm::log(input);
  EXPECT_FLOAT_EQ(lane(result, 0), 0.0f);
}

// ==================== Frexp / Ldexp Special Values ====================

TEST(NeonFrexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  dfm::IntType_t<float32x4_t> eptr;
  float32x4_t mantissa = dfm::frexp(input, &eptr);

  for (int i = 0; i < kLanes; ++i) {
    int32_t scalar_exp;
    float scalar_m = dfm::frexp(lane(input, i), &scalar_exp);
    if (std::isnan(scalar_m)) {
      EXPECT_TRUE(std::isnan(lane(mantissa, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(mantissa, i), scalar_m) << "Lane " << i;
      EXPECT_EQ(lane(eptr, i), scalar_exp) << "Lane " << i;
    }
  }
}

TEST(NeonLdexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(kInf, -kInf, kNaN, 0.0f);
  int32x4_t exp = makeInt4(0, 7, 1, 2);
  auto result = dfm::ldexp(input, exp);

  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// ==================== Mixed Special Values ====================

TEST(NeonSin, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(kNaN, kInf, -kInf, 1.0f);
  auto result = dfm::sin(input);
  EXPECT_TRUE(std::isnan(lane(result, 0))) << "sin(NaN) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 1))) << "sin(+Inf) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "sin(-Inf) should be NaN";
  EXPECT_NEAR(lane(result, 3), ::sinf(1.0f), 1e-7f) << "sin(1) lane contaminated";
}

TEST(NeonCos, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(0.0f, kNaN, kInf, -kInf);
  auto result = dfm::cos(input);
  EXPECT_EQ(lane(result, 0), 1.0f) << "cos(0) should be 1";
  EXPECT_TRUE(std::isnan(lane(result, 1))) << "cos(NaN) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "cos(+Inf) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 3))) << "cos(-Inf) should be NaN";
}

TEST(NeonTan, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(0.0f, kNaN, kInf, -kInf);
  auto result = dfm::tan(input);
  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_TRUE(std::isnan(lane(result, 1)));
  EXPECT_TRUE(std::isnan(lane(result, 2)));
  EXPECT_TRUE(std::isnan(lane(result, 3)));
}

TEST(NeonAtan, SpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(0.0f, kNaN, kInf, -kInf);
  auto result = dfm::atan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(NeonAtan, LargeMagnitude) {
  float32x4_t input = make4(20000000.0f, -20000000.0f, 1e10f, -1e10f);
  auto result = dfm::atan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan(lane(input, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(NeonAsin, OutOfRange) {
  float32x4_t input = make4(1.00001f, -1.00001f, 2.0f, -5.0f);
  auto result = dfm::asin(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(NeonAcos, OutOfRange) {
  float32x4_t input = make4(1.00001f, -1.00001f, 2.0f, -5.0f);
  auto result = dfm::acos(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

// ==================== Atan2 Edge Cases ====================

TEST(NeonAtan2, NegativeZero) {
  float32x4_t y = make4(0.0f, -0.0f, 0.0f, -0.0f);
  float32x4_t x = make4(-1.0f, -1.0f, 1.0f, 1.0f);
  auto result = dfm::atan2(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
  }
}

TEST(NeonAtan2, ZeroZero) {
  float32x4_t y = make4(0.0f, -0.0f, 0.0f, -0.0f);
  float32x4_t x = make4(0.0f, 0.0f, -0.0f, -0.0f);
  auto result = dfm::atan2(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
  }
}

TEST(NeonAtan2, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t y = make4(kNaN, 1.0f, kInf, -kInf);
  float32x4_t x = make4(1.0f, kNaN, kInf, -kInf);
  auto result = dfm::atan2(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// ==================== Accuracy/Bounds Trait Variants ====================

struct NeonBoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// -- exp bounds --

TEST(NeonExpBounds, EdgeCases) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(kNaN, kInf, -kInf, 89.0f);
  auto result = dfm::exp<float32x4_t, NeonBoundsTraits>(input);
  EXPECT_TRUE(std::isnan(lane(result, 0))) << "exp(NaN)";
  EXPECT_TRUE(std::isinf(lane(result, 1)) && lane(result, 1) > 0) << "exp(+Inf)";
  EXPECT_EQ(lane(result, 2), 0.0f) << "exp(-Inf)";
  EXPECT_TRUE(std::isinf(lane(result, 3)) || lane(result, 3) > 1e38f) << "exp(89)";
}

// -- exp2 bounds --

TEST(NeonExp2Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(kInf, -kInf, kNaN, 0.0f);
  auto result = dfm::exp2<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp2<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// -- exp10 bounds --

TEST(NeonExp10Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(kInf, -kInf, kNaN, 0.0f);
  auto result = dfm::exp10<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp10<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// -- log accurate --

TEST(NeonLogAccurate, NegativeInputs) {
  float32x4_t input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  auto result = dfm::log<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << lane(input, i);
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

TEST(NeonLogAccurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  auto result = dfm::log<float32x4_t, dfm::MaxAccuracyTraits>(input);
  EXPECT_TRUE(std::isinf(lane(result, 0)) && lane(result, 0) < 0) << "log(0) should be -Inf";
  EXPECT_TRUE(std::isinf(lane(result, 1)) && lane(result, 1) > 0) << "log(+Inf) should be +Inf";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "log(NaN) should be NaN";
  float expected_denorm =
      dfm::log<float, dfm::MaxAccuracyTraits>(std::numeric_limits<float>::denorm_min());
  uint32_t dist = dfm::float_distance(expected_denorm, lane(result, 3));
  EXPECT_LE(dist, 0u) << "log(denorm_min)";
}

// -- log2 accurate --

TEST(NeonLog2Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  auto result = dfm::log2<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log2<float, dfm::MaxAccuracyTraits>(lane(input, i));
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

TEST(NeonLog2Accurate, NegativeInputs) {
  float32x4_t input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  auto result = dfm::log2<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log2<float, dfm::MaxAccuracyTraits>(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << lane(input, i);
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

// -- log10 accurate --

TEST(NeonLog10Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  auto result = dfm::log10<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log10<float, dfm::MaxAccuracyTraits>(lane(input, i));
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

TEST(NeonLog10Accurate, NegativeInputs) {
  float32x4_t input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  auto result = dfm::log10<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log10<float, dfm::MaxAccuracyTraits>(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << " input=" << lane(input, i);
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      EXPECT_EQ(expected, actual) << "Lane " << i;
    }
  }
}

// -- cbrt accurate --

TEST(NeonCbrtAccurate, EdgeCases) {
  float32x4_t input = make4(
      0.0f,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min());
  auto result = dfm::cbrt<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(lane(input, i));
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

TEST(NeonCbrtAccurate, NegativeInf) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t input = make4(-kInf, -8.0f, -27.0f, -1.0f);
  auto result = dfm::cbrt<float32x4_t, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 0u) << "Lane " << i;
    }
  }
}

// -- atan2 bounds --

TEST(NeonAtan2Bounds, LaneByLane) {
  float32x4_t y = make4(1.0f, -1.0f, 0.0f, 1.0f);
  float32x4_t x = make4(1.0f, 1.0f, 1.0f, -1.0f);
  auto result = dfm::atan2<float32x4_t, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i;
  }
}

TEST(NeonAtan2Bounds, InfCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t y = make4(kInf, -kInf, kInf, -kInf);
  float32x4_t x = make4(kInf, kInf, -kInf, -kInf);
  auto result = dfm::atan2<float32x4_t, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
  }
}

// ==================== Large Magnitude Trig ====================

TEST(NeonSin, LargeMagnitude) {
  float32x4_t input = make4(100.0f, -100.0f, 1000.0f, -1000.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::sin(x); },
      [](float x) { return ::sinf(x); },
      "sin_large",
      2);
}

TEST(NeonCos, LargeMagnitude) {
  float32x4_t input = make4(100.0f, -100.0f, 1000.0f, -1000.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::cos(x); },
      [](float x) { return ::cosf(x); },
      "cos_large",
      2);
}

TEST(NeonTan, LargeMagnitude) {
  float32x4_t input = make4(50.0f, -50.0f, 200.0f, -200.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::tan(x); },
      [](float x) { return dfm::tan(x); },
      "tan_large");
}

// ==================== Domain Boundary Tests ====================

TEST(NeonAsin, DomainBoundaries) {
  float32x4_t input = make4(-1.0f, 1.0f, 0.0f, 0.99999f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::asin(x); },
      [](float x) { return dfm::asin(x); },
      "asin_boundary");
}

TEST(NeonAcos, DomainBoundaries) {
  float32x4_t input = make4(-1.0f, 1.0f, 0.0f, -0.5f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::acos(x); },
      [](float x) { return dfm::acos(x); },
      "acos_boundary");
}

// ==================== Mixed Values ====================

TEST(NeonMixed, MixedSin) {
  float32x4_t input = make4(0.1f, 1.5f, -0.7f, 2.8f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::sin(x); },
      [](float x) { return ::sinf(x); },
      "mixed_sin",
      2);
}

TEST(NeonMixed, MixedExp) {
  float32x4_t input = make4(-5.0f, 0.0f, 3.0f, -0.1f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::exp(x); },
      [](float x) { return dfm::exp(x); },
      "mixed_exp");
}

TEST(NeonMixed, MixedLog) {
  float32x4_t input = make4(0.001f, 1.0f, 100.0f, 10000.0f);
  checkLaneByLane(
      input,
      [](float32x4_t x) { return dfm::log(x); },
      [](float x) { return dfm::log(x); },
      "mixed_log");
}

// --- hypot ---

TEST(NeonTranscendentals, Hypot) {
  float32x4_t x = make4(3.0f, 0.0f, 1e30f, -5.0f);
  float32x4_t y = make4(4.0f, 1.0f, 1e30f, 12.0f);
  auto result = dfm::hypot(x, y);
  for (int i = 0; i < kLanes; ++i) {
    double xd = static_cast<double>(lane(x, i));
    double yd = static_cast<double>(lane(y, i));
    float expected = static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 2u) << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
                        << " expected=" << expected << " actual=" << actual;
  }
}

// -- hypot bounds --

TEST(NeonHypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t x = make4(inf, -inf, nan, nan);
  float32x4_t y = make4(3.0f, nan, inf, -inf);
  auto result = dfm::hypot<float32x4_t, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    float actual = lane(result, i);
    EXPECT_TRUE(std::isinf(actual) && actual > 0)
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i) << " result=" << actual;
  }
}

TEST(NeonHypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float32x4_t x = make4(nan, 3.0f, nan, nan);
  float32x4_t y = make4(3.0f, nan, 0.0f, nan);
  auto result = dfm::hypot<float32x4_t, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    float actual = lane(result, i);
    EXPECT_TRUE(std::isnan(actual))
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i) << " result=" << actual;
  }
}

// ---- pow ----

static float gt_pow(float x, float y) {
  return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

TEST(NeonPow, LaneByLane) {
  float32x4_t x = make4(2.0f, 3.0f, 4.0f, 0.5f);
  float32x4_t y = make4(3.0f, 2.0f, 0.5f, -1.0f);
  auto result = dfm::pow(x, y);
  EXPECT_EQ(lane(result, 0), 8.0f);
  EXPECT_EQ(lane(result, 1), 9.0f);
  EXPECT_EQ(lane(result, 2), 2.0f);
  EXPECT_EQ(lane(result, 3), 2.0f);
}

TEST(NeonPow, NegativeBase) {
  float32x4_t x = make4(-2.0f, -1.0f, -3.0f, -0.5f);
  float32x4_t y = make4(3.0f, 2.0f, 5.0f, -1.0f);
  auto result = dfm::pow(x, y);
  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(x, i), lane(y, i));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 2u) << "Lane " << i << ": pow(" << lane(x, i) << ", " << lane(y, i)
                        << ") expected=" << expected << " got=" << actual;
  }
}

TEST(NeonPow, NegBaseNonInt) {
  float32x4_t x = make4(-2.0f, -1.0f, -0.5f, -3.0f);
  float32x4_t y = make4(0.5f, 1.5f, 2.7f, -0.5f);
  auto result = dfm::pow(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": pow(" << lane(x, i) << ", " << lane(y, i) << ") should be NaN, got "
        << lane(result, i);
  }
}

TEST(NeonPow, YZero) {
  float32x4_t x = make4(2.0f, -3.0f, 0.0f, 100.0f);
  float32x4_t y = vdupq_n_f32(0.0f);
  auto result = dfm::pow(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 1.0f) << "Lane " << i;
  }
}

TEST(NeonPow, ScalarExp) {
  float32x4_t x = make4(1.0f, 2.0f, 3.0f, 4.0f);
  auto result = dfm::pow(x, 2.0f);
  for (int i = 0; i < kLanes; ++i) {
    float xi = lane(x, i);
    EXPECT_EQ(lane(result, i), xi * xi) << "Lane " << i;
  }
}

TEST(NeonPow, ScalarExpGeneral) {
  float32x4_t x = make4(2.0f, 4.0f, 8.0f, 16.0f);
  auto result = dfm::pow(x, 2.5f);
  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(x, i), 2.5f);
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 4u) << "Lane " << i << ": pow(" << lane(x, i) << ", 2.5) expected=" << expected
                        << " got=" << actual;
  }
}

TEST(NeonPowBounds, Specials) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t x = make4(0.0f, -0.0f, kInf, kNaN);
  float32x4_t y = make4(2.0f, 3.0f, -1.0f, 0.0f);
  auto result = dfm::pow<float32x4_t, dfm::MaxAccuracyTraits>(x, y);

  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_FALSE(std::signbit(lane(result, 0)));

  EXPECT_EQ(lane(result, 1), -0.0f);
  EXPECT_TRUE(std::signbit(lane(result, 1)));

  EXPECT_EQ(lane(result, 2), 0.0f);
  EXPECT_FALSE(std::signbit(lane(result, 2)));

  EXPECT_EQ(lane(result, 3), 1.0f); // pow(NaN, 0) = 1
}

TEST(NeonPowBounds, InfExp) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float32x4_t x = make4(0.5f, 2.0f, -1.0f, -1.0f);
  float32x4_t y = make4(-kInf, -kInf, kInf, -kInf);
  auto result = dfm::pow<float32x4_t, dfm::MaxAccuracyTraits>(x, y);
  EXPECT_EQ(lane(result, 0), kInf);
  EXPECT_EQ(lane(result, 1), 0.0f);
  EXPECT_EQ(lane(result, 2), 1.0f);
  EXPECT_EQ(lane(result, 3), 1.0f);
}

TEST(NeonPowBounds, Subnormal) {
  float32x4_t x = make4(1.0e-40f, 1.0e-42f, 1.0e-44f, std::numeric_limits<float>::denorm_min());
  float32x4_t y = make4(2.0f, 0.5f, -1.0f, 3.0f);
  auto result = dfm::pow<float32x4_t, dfm::MaxAccuracyTraits>(x, y);

  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(x, i), lane(y, i));
    float actual = lane(result, i);
    if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual))
          << "Lane " << i << ": pow(" << lane(x, i) << ", " << lane(y, i) << ") expected inf";
    } else if (expected == 0.0f) {
      EXPECT_EQ(actual, 0.0f) << "Lane " << i << ": pow(" << lane(x, i) << ", " << lane(y, i)
                              << ") expected 0";
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 4u) << "Lane " << i << ": pow(" << lane(x, i) << ", " << lane(y, i)
                          << ") expected=" << expected << " got=" << actual;
    }
  }
}

TEST(NeonPowBounds, SubnormalScalarY) {
  float32x4_t x = make4(1.0e-40f, 1.0e-42f, 1.0e-44f, std::numeric_limits<float>::denorm_min());
  auto result = dfm::pow<float32x4_t, dfm::MaxAccuracyTraits>(x, 2.0f);

  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(x, i), 2.0f);
    float actual = lane(result, i);
    if (expected == 0.0f) {
      EXPECT_EQ(actual, 0.0f) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 4u) << "Lane " << i << ": pow(" << lane(x, i) << ", 2) expected=" << expected
                          << " got=" << actual;
    }
  }
}

TEST(NeonPowBounds, XOne) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  float32x4_t x = vdupq_n_f32(1.0f);
  float32x4_t y = make4(0.0f, kNaN, -1.0f, 42.0f);
  auto result = dfm::pow<float32x4_t, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 1.0f) << "Lane " << i;
  }
}

#endif // defined(__aarch64__)
