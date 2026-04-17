/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include <gtest/gtest.h>

#if defined(__SSE4_1__)

namespace dfm = dispenso::fast_math;
using SseFloat = dfm::SseFloat;
using SseInt32 = dfm::SseInt32;
using SseUint32 = dfm::SseUint32;

// Helper: extract lane i from __m128.
static float lane(__m128 v, int i) {
  alignas(16) float buf[4];
  _mm_store_ps(buf, v);
  return buf[i];
}

// Helper: extract lane i from __m128i (as int32_t).
static int32_t lane(__m128i v, int i) {
  alignas(16) int32_t buf[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(buf), v);
  return buf[i];
}

// Helper: create __m128 from 4 distinct values.
static __m128 make4(float a, float b, float c, float d) {
  return _mm_set_ps(d, c, b, a);
}

// ---- SseFloat basic arithmetic ----

TEST(SseFloat, Broadcast) {
  SseFloat v(3.0f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(lane(v, i), 3.0f);
  }
}

TEST(SseFloat, Arithmetic) {
  SseFloat a = make4(1.0f, 2.0f, 3.0f, 4.0f);
  SseFloat b = make4(5.0f, 6.0f, 7.0f, 8.0f);

  SseFloat sum = a + b;
  EXPECT_EQ(lane(sum, 0), 6.0f);
  EXPECT_EQ(lane(sum, 3), 12.0f);

  SseFloat diff = b - a;
  EXPECT_EQ(lane(diff, 0), 4.0f);
  EXPECT_EQ(lane(diff, 3), 4.0f);

  SseFloat prod = a * b;
  EXPECT_EQ(lane(prod, 0), 5.0f);
  EXPECT_EQ(lane(prod, 3), 32.0f);

  SseFloat quot = b / a;
  EXPECT_EQ(lane(quot, 0), 5.0f);
  EXPECT_EQ(lane(quot, 1), 3.0f);
}

TEST(SseFloat, Negation) {
  SseFloat a = make4(1.0f, -2.0f, 0.0f, 3.0f);
  SseFloat neg = -a;
  EXPECT_EQ(lane(neg, 0), -1.0f);
  EXPECT_EQ(lane(neg, 1), 2.0f);
  EXPECT_EQ(lane(neg, 2), -0.0f);
  EXPECT_EQ(lane(neg, 3), -3.0f);
}

// ---- SseFloat comparisons ----

TEST(SseFloat, Comparisons) {
  SseFloat a = make4(1.0f, 2.0f, 3.0f, 4.0f);
  SseFloat b = make4(2.0f, 2.0f, 1.0f, 5.0f);

  SseFloat lt = a < b;
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, 0)), 0u); // 1 < 2 true
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 1)), 0u); // 2 < 2 false
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 2)), 0u); // 3 < 1 false
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, 3)), 0u); // 4 < 5 true
}

// ---- SseInt32 arithmetic ----

TEST(SseInt32, BasicOps) {
  SseInt32 a(10);
  SseInt32 b(3);
  EXPECT_EQ(lane(a + b, 0), 13);
  EXPECT_EQ(lane(a - b, 0), 7);
  EXPECT_EQ(lane(a * b, 0), 30);
}

TEST(SseInt32, ShiftOps) {
  SseInt32 a(8);
  EXPECT_EQ(lane(a << 2, 0), 32);
  EXPECT_EQ(lane(a >> 1, 0), 4);

  SseInt32 neg(-8);
  EXPECT_EQ(lane(neg >> 1, 0), -4); // Arithmetic shift preserves sign
}

// ---- SseUint32 ----

TEST(SseUint32, LogicalShift) {
  SseUint32 a(0x80000000u);
  // Logical shift right — does NOT preserve sign.
  EXPECT_EQ(static_cast<uint32_t>(lane(a >> 1, 0)), 0x40000000u);
}

// ---- bit_cast ----

TEST(SseBitCast, FloatToInt) {
  SseFloat f(1.0f);
  SseInt32 i = dfm::bit_cast<SseInt32>(f);
  EXPECT_EQ(lane(i, 0), 0x3f800000);
}

TEST(SseBitCast, IntToFloat) {
  SseInt32 i(0x40000000);
  SseFloat f = dfm::bit_cast<SseFloat>(i);
  EXPECT_EQ(lane(f, 0), 2.0f);
}

TEST(SseBitCast, RoundTrip) {
  SseFloat original = make4(1.0f, -2.0f, 0.5f, 42.0f);
  SseFloat roundtripped = dfm::bit_cast<SseFloat>(dfm::bit_cast<SseInt32>(original));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(lane(roundtripped, i), lane(original, i));
  }
}

// ---- FloatTraits<SseFloat> ----

TEST(SseFloatTraits, Conditional) {
  SseFloat mask = SseFloat(make4(1.0f, 0.0f, 1.0f, 0.0f)) > SseFloat(0.5f);
  SseFloat x(10.0f);
  SseFloat y(20.0f);
  SseFloat result = dfm::FloatTraits<SseFloat>::conditional(mask, x, y);
  EXPECT_EQ(lane(result, 0), 10.0f);
  EXPECT_EQ(lane(result, 1), 20.0f);
  EXPECT_EQ(lane(result, 2), 10.0f);
  EXPECT_EQ(lane(result, 3), 20.0f);
}

TEST(SseFloatTraits, Fma) {
  SseFloat a(2.0f), b(3.0f), c(4.0f);
  SseFloat result = dfm::FloatTraits<SseFloat>::fma(a, b, c);
  EXPECT_EQ(lane(result, 0), 10.0f);
}

TEST(SseFloatTraits, MinMax) {
  SseFloat a = make4(1.0f, 5.0f, 3.0f, 7.0f);
  SseFloat b = make4(4.0f, 2.0f, 6.0f, 0.0f);
  SseFloat mn = dfm::FloatTraits<SseFloat>::min(a, b);
  SseFloat mx = dfm::FloatTraits<SseFloat>::max(a, b);
  EXPECT_EQ(lane(mn, 0), 1.0f);
  EXPECT_EQ(lane(mn, 1), 2.0f);
  EXPECT_EQ(lane(mx, 0), 4.0f);
  EXPECT_EQ(lane(mx, 1), 5.0f);
}

// ---- Util functions ----

TEST(SseUtil, FloorSmall) {
  __m128 x = make4(1.5f, -1.5f, 2.0f, 0.1f);
  __m128 result = dfm::floor_small(x);
  EXPECT_EQ(lane(result, 0), 1.0f);
  EXPECT_EQ(lane(result, 1), -2.0f);
  EXPECT_EQ(lane(result, 2), 2.0f);
  EXPECT_EQ(lane(result, 3), 0.0f);
}

TEST(SseUtil, ConvertToInt) {
  __m128 x = make4(1.6f, -1.6f, 2.5f, 0.4f);
  __m128i result = dfm::convert_to_int(x);
  EXPECT_EQ(lane(result, 0), 2); // Round to nearest even
  EXPECT_EQ(lane(result, 1), -2);
  EXPECT_EQ(lane(result, 2), 2); // Round to nearest even: 2.5 → 2
  EXPECT_EQ(lane(result, 3), 0);
}

TEST(SseUtil, Gather) {
  float table[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  __m128i indices = _mm_set_epi32(4, 2, 0, 1);
  __m128 result = dfm::gather<SseFloat>(table, indices);
  EXPECT_EQ(lane(result, 0), 20.0f);
  EXPECT_EQ(lane(result, 1), 10.0f);
  EXPECT_EQ(lane(result, 2), 30.0f);
  EXPECT_EQ(lane(result, 3), 50.0f);
}

TEST(SseUtil, IntDivBy3) {
  __m128i a = _mm_set_epi32(30, 9, 6, 3);
  __m128i result = dfm::int_div_by_3(a);
  EXPECT_EQ(lane(result, 0), 1);
  EXPECT_EQ(lane(result, 1), 2);
  EXPECT_EQ(lane(result, 2), 3);
  EXPECT_EQ(lane(result, 3), 10);
}

TEST(SseUtil, Signof) {
  __m128 x = make4(1.0f, -1.0f, 0.0f, -0.0f);
  __m128 result = dfm::signof(x);
  EXPECT_EQ(lane(result, 0), 1.0f);
  EXPECT_EQ(lane(result, 1), -1.0f);
  EXPECT_EQ(lane(result, 2), 1.0f);
  EXPECT_EQ(lane(result, 3), -1.0f);
}

TEST(SseUtil, Signofi) {
  __m128i a = _mm_set_epi32(-5, 0, 10, -1);
  __m128i result = dfm::signofi<SseFloat>(SseInt32(a));
  EXPECT_EQ(lane(result, 0), -1);
  EXPECT_EQ(lane(result, 1), 1);
  EXPECT_EQ(lane(result, 2), 1);
  EXPECT_EQ(lane(result, 3), -1);
}

#else // !defined(__SSE4_1__)

// Dummy test so the binary has at least one test on non-SSE platforms.
TEST(SseFloat, NotAvailable) {
  GTEST_SKIP() << "SSE4.1 not available";
}

#endif // defined(__SSE4_1__)
