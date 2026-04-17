/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cmath>
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

#endif // defined(__aarch64__)
