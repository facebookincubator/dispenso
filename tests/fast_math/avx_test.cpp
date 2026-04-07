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

#if defined(__AVX2__)

namespace dfm = dispenso::fast_math;
using AvxFloat = dfm::AvxFloat;
using AvxInt32 = dfm::AvxInt32;
using AvxUint32 = dfm::AvxUint32;

constexpr int32_t kLanes = 8;

// Helper: extract lane i from __m256.
static float lane(__m256 v, int i) {
  alignas(32) float buf[kLanes];
  _mm256_store_ps(buf, v);
  return buf[i];
}

// Helper: extract lane i from __m256i (as int32_t).
static int32_t lane(__m256i v, int i) {
  alignas(32) int32_t buf[kLanes];
  _mm256_store_si256(reinterpret_cast<__m256i*>(buf), v);
  return buf[i];
}

// Helper: create __m256 from 8 distinct values.
static __m256 make8(float a, float b, float c, float d, float e, float f, float g, float h) {
  return _mm256_set_ps(h, g, f, e, d, c, b, a);
}

// Helper: create __m256 from base + stride (8 consecutive values).
static __m256 makeSeq(float base, float stride) {
  return _mm256_set_ps(
      base + 7 * stride,
      base + 6 * stride,
      base + 5 * stride,
      base + 4 * stride,
      base + 3 * stride,
      base + 2 * stride,
      base + stride,
      base);
}

// Helper: compare lane-by-lane against scalar function using raw __m256 types.
template <typename ScalarFunc, typename AvxFunc>
static void
checkLaneByLane(ScalarFunc scalar_fn, AvxFunc avx_fn, __m256 input, uint32_t max_ulps = 0) {
  __m256 result = avx_fn(input);
  for (int i = 0; i < kLanes; ++i) {
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

// ---- AvxFloat basic arithmetic ----

TEST(AvxFloat, Broadcast) {
  AvxFloat v(3.0f);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(v, i), 3.0f);
  }
}

TEST(AvxFloat, Arithmetic) {
  AvxFloat a = make8(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
  AvxFloat b = make8(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);

  AvxFloat sum = a + b;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(sum, i), 9.0f) << "Lane " << i;
  }

  AvxFloat diff = b - a;
  EXPECT_EQ(lane(diff, 0), 7.0f);
  EXPECT_EQ(lane(diff, 7), -7.0f);

  AvxFloat prod = a * b;
  EXPECT_EQ(lane(prod, 0), 8.0f);
  EXPECT_EQ(lane(prod, 3), 20.0f);
  EXPECT_EQ(lane(prod, 7), 8.0f);

  AvxFloat quot = a / b;
  EXPECT_FLOAT_EQ(lane(quot, 0), 0.125f);
  EXPECT_FLOAT_EQ(lane(quot, 7), 8.0f);
}

TEST(AvxFloat, Negation) {
  AvxFloat a = make8(1.0f, -2.0f, 0.0f, 3.0f, -0.5f, 100.0f, -0.0f, 42.0f);
  AvxFloat neg = -a;
  EXPECT_EQ(lane(neg, 0), -1.0f);
  EXPECT_EQ(lane(neg, 1), 2.0f);
  EXPECT_EQ(lane(neg, 2), -0.0f);
  EXPECT_EQ(lane(neg, 3), -3.0f);
  EXPECT_EQ(lane(neg, 4), 0.5f);
  EXPECT_EQ(lane(neg, 5), -100.0f);
  // -(-0.0) should be +0.0
  EXPECT_FALSE(std::signbit(lane(neg, 6)));
  EXPECT_EQ(lane(neg, 7), -42.0f);
}

TEST(AvxFloat, CompoundAssignment) {
  AvxFloat a(1.0f);
  a += AvxFloat(2.0f);
  EXPECT_EQ(lane(a, 0), 3.0f);
  a -= AvxFloat(1.0f);
  EXPECT_EQ(lane(a, 0), 2.0f);
  a *= AvxFloat(5.0f);
  EXPECT_EQ(lane(a, 0), 10.0f);
}

// ---- AvxFloat comparisons ----

TEST(AvxFloat, Comparisons) {
  AvxFloat a = make8(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
  AvxFloat b = make8(2.0f, 2.0f, 1.0f, 5.0f, 5.0f, 4.0f, 7.0f, 9.0f);

  AvxFloat lt = a < b;
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, 0)), 0u); // 1 < 2 true
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 1)), 0u); // 2 < 2 false
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 2)), 0u); // 3 < 1 false
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, 3)), 0u); // 4 < 5 true
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 4)), 0u); // 5 < 5 false
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 5)), 0u); // 6 < 4 false
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(lt, 6)), 0u); // 7 < 7 false
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(lt, 7)), 0u); // 8 < 9 true

  AvxFloat gt = a > b;
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(gt, 0)), 0u); // 1 > 2 false
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(gt, 2)), 0u); // 3 > 1 true
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(gt, 5)), 0u); // 6 > 4 true

  AvxFloat eq = a == b;
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(eq, 0)), 0u); // 1 == 2 false
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(eq, 1)), 0u); // 2 == 2 true
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(eq, 4)), 0u); // 5 == 5 true
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(eq, 6)), 0u); // 7 == 7 true
}

TEST(AvxFloat, LogicalNot) {
  AvxFloat mask = AvxFloat(make8(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f)) > AvxFloat(0.5f);
  AvxFloat notmask = !mask;
  EXPECT_EQ(dfm::bit_cast<uint32_t>(lane(notmask, 0)), 0u);
  EXPECT_NE(dfm::bit_cast<uint32_t>(lane(notmask, 1)), 0u);
}

// ---- AvxInt32 arithmetic ----

TEST(AvxInt32, BasicOps) {
  AvxInt32 a(10);
  AvxInt32 b(3);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a + b, i), 13) << "Lane " << i;
    EXPECT_EQ(lane(a - b, i), 7) << "Lane " << i;
    EXPECT_EQ(lane(a * b, i), 30) << "Lane " << i;
  }
}

TEST(AvxInt32, Negation) {
  AvxInt32 a = AvxInt32(_mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1));
  AvxInt32 neg = -a;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(neg, i), -(i + 1)) << "Lane " << i;
  }
}

TEST(AvxInt32, ShiftOps) {
  AvxInt32 a(8);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a << 2, i), 32) << "Lane " << i;
    EXPECT_EQ(lane(a >> 1, i), 4) << "Lane " << i;
  }

  AvxInt32 neg(-8);
  EXPECT_EQ(lane(neg >> 1, 0), -4); // Arithmetic shift preserves sign
}

TEST(AvxInt32, Comparisons) {
  AvxInt32 a = AvxInt32(_mm256_set_epi32(8, 7, 5, 5, 4, 3, 2, 1));
  AvxInt32 b = AvxInt32(_mm256_set_epi32(1, 7, 6, 4, 4, 3, 9, 0));

  AvxInt32 eq = a == b;
  EXPECT_EQ(lane(eq, 0), 0); // 1 == 0 false
  EXPECT_NE(lane(eq, 2), 0); // 3 == 3 true
  EXPECT_NE(lane(eq, 3), 0); // 4 == 4 true
  EXPECT_NE(lane(eq, 6), 0); // 7 == 7 true

  AvxInt32 lt = a < b;
  EXPECT_EQ(lane(lt, 0), 0); // 1 < 0 false
  EXPECT_NE(lane(lt, 1), 0); // 2 < 9 true
}

// ---- AvxUint32 ----

TEST(AvxUint32, LogicalShift) {
  AvxUint32 a(0x80000000u);
  // Logical shift right — does NOT preserve sign.
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(static_cast<uint32_t>(lane(a >> 1, i)), 0x40000000u) << "Lane " << i;
  }
}

TEST(AvxUint32, UnsignedComparison) {
  AvxUint32 a(0x80000000u);
  AvxUint32 b(0x7FFFFFFFu);
  // Unsigned: 0x80000000 > 0x7FFFFFFF
  AvxUint32 gt = a > b;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_NE(static_cast<uint32_t>(lane(gt, i)), 0u) << "Lane " << i;
  }
}

// ---- bit_cast ----

TEST(AvxBitCast, FloatToInt) {
  AvxFloat f(1.0f);
  AvxInt32 i = dfm::bit_cast<AvxInt32>(f);
  for (int j = 0; j < kLanes; ++j) {
    EXPECT_EQ(lane(i, j), 0x3f800000) << "Lane " << j;
  }
}

TEST(AvxBitCast, IntToFloat) {
  AvxInt32 i(0x40000000);
  AvxFloat f = dfm::bit_cast<AvxFloat>(i);
  for (int j = 0; j < kLanes; ++j) {
    EXPECT_EQ(lane(f, j), 2.0f) << "Lane " << j;
  }
}

TEST(AvxBitCast, RoundTrip) {
  AvxFloat original = make8(1.0f, -2.0f, 0.5f, 42.0f, -0.0f, 3.14f, 1e10f, -1e-10f);
  AvxFloat roundtripped = dfm::bit_cast<AvxFloat>(dfm::bit_cast<AvxInt32>(original));
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(roundtripped, i), lane(original, i)) << "Lane " << i;
  }
}

// ---- FloatTraits<AvxFloat> ----

TEST(AvxFloatTraits, Conditional) {
  AvxFloat mask = AvxFloat(make8(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f)) > AvxFloat(0.5f);
  AvxFloat x(10.0f);
  AvxFloat y(20.0f);
  AvxFloat result = dfm::FloatTraits<AvxFloat>::conditional(mask, x, y);
  for (int i = 0; i < kLanes; ++i) {
    float expected = (i % 2 == 0) ? 10.0f : 20.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(AvxFloatTraits, ConditionalInt32) {
  AvxFloat mask = AvxFloat(make8(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f)) > AvxFloat(0.5f);
  AvxInt32 x(100);
  AvxInt32 y(200);
  AvxInt32 result = dfm::FloatTraits<AvxFloat>::conditional(mask, x, y);
  for (int i = 0; i < kLanes; ++i) {
    int32_t expected = (i % 2 == 0) ? 100 : 200;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(AvxFloatTraits, Fma) {
  AvxFloat a(2.0f), b(3.0f), c(4.0f);
  AvxFloat result = dfm::FloatTraits<AvxFloat>::fma(a, b, c);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 10.0f) << "Lane " << i;
  }
}

TEST(AvxFloatTraits, Sqrt) {
  AvxFloat input = make8(1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f);
  AvxFloat result = dfm::FloatTraits<AvxFloat>::sqrt(input);
  float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_FLOAT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxFloatTraits, MinMax) {
  AvxFloat a = make8(1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 8.0f, 4.0f, 6.0f);
  AvxFloat b = make8(4.0f, 2.0f, 6.0f, 0.0f, 9.0f, 1.0f, 4.0f, 3.0f);
  AvxFloat mn = dfm::FloatTraits<AvxFloat>::min(a, b);
  AvxFloat mx = dfm::FloatTraits<AvxFloat>::max(a, b);
  float emin[] = {1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 1.0f, 4.0f, 3.0f};
  float emax[] = {4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 8.0f, 4.0f, 6.0f};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(mn, i), emin[i]) << "Lane " << i;
    EXPECT_EQ(lane(mx, i), emax[i]) << "Lane " << i;
  }
}

// ---- Util functions ----

TEST(AvxUtil, FloorSmall) {
  __m256 x = make8(1.5f, -1.5f, 2.0f, 0.1f, 3.9f, -0.1f, 0.0f, -3.7f);
  __m256 result = dfm::floor_small(x);
  float expected[] = {1.0f, -2.0f, 2.0f, 0.0f, 3.0f, -1.0f, 0.0f, -4.0f};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxUtil, ConvertToInt) {
  __m256 x = make8(1.6f, -1.6f, 2.5f, 0.4f, 3.5f, -2.5f, 0.0f, 100.0f);
  __m256i result = dfm::convert_to_int(x);
  // Round to nearest even.
  int32_t expected[] = {2, -2, 2, 0, 4, -2, 0, 100};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxUtil, ConvertToInt_NaN) {
  // NaN/Inf should be masked to 0.
  __m256 x = make8(
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f);
  __m256i result = dfm::convert_to_int(x);
  EXPECT_EQ(lane(result, 0), 0); // NaN → 0
  EXPECT_EQ(lane(result, 1), 0); // Inf → 0
  EXPECT_EQ(lane(result, 2), 0); // -Inf → 0
  EXPECT_EQ(lane(result, 3), 1);
}

TEST(AvxUtil, Gather) {
  float table[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f};
  __m256i indices = _mm256_set_epi32(8, 0, 5, 2, 7, 1, 4, 3);
  __m256 result = dfm::gather<AvxFloat>(table, indices);
  EXPECT_EQ(lane(result, 0), 40.0f);
  EXPECT_EQ(lane(result, 1), 50.0f);
  EXPECT_EQ(lane(result, 2), 20.0f);
  EXPECT_EQ(lane(result, 3), 80.0f);
  EXPECT_EQ(lane(result, 4), 30.0f);
  EXPECT_EQ(lane(result, 5), 60.0f);
  EXPECT_EQ(lane(result, 6), 10.0f);
  EXPECT_EQ(lane(result, 7), 90.0f);
}

TEST(AvxUtil, IntDivBy3) {
  __m256i a = _mm256_set_epi32(99, 30, 27, 12, 9, 6, 3, 0);
  __m256i result = dfm::int_div_by_3(a);
  int32_t expected[] = {0, 1, 2, 3, 4, 9, 10, 33};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxUtil, Signof) {
  __m256 x = make8(1.0f, -1.0f, 0.0f, -0.0f, 42.0f, -42.0f, 0.5f, -0.5f);
  __m256 result = dfm::signof(x);
  float expected[] = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxUtil, Signofi) {
  __m256i a = _mm256_set_epi32(-100, 100, -1, 0, 10, -5, 1, -1);
  __m256i result = dfm::signofi<AvxFloat>(AvxInt32(a));
  int32_t expected[] = {-1, 1, -1, 1, 1, -1, 1, -1};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(AvxUtil, Nonnormal) {
  __m256 x = make8(
      1.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      -0.0f,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::max());
  AvxInt32 result = dfm::nonnormal(AvxFloat(x));
  EXPECT_EQ(lane(result, 0), 0); // normal
  EXPECT_NE(lane(result, 1), 0); // inf
  EXPECT_NE(lane(result, 2), 0); // -inf
  EXPECT_NE(lane(result, 3), 0); // nan
  EXPECT_EQ(lane(result, 4), 0); // zero
  EXPECT_EQ(lane(result, 5), 0); // -zero
  EXPECT_EQ(lane(result, 6), 0); // denorm
  EXPECT_EQ(lane(result, 7), 0); // normal max
}

// ---- Transcendental functions: lane-by-lane correctness ----
// All tests use raw __m256 types with dfm::func(x) — no explicit template args.
// This exercises the SimdTypeFor forwarding mechanism.

TEST(AvxSin, LaneByLane) {
  __m256 input = make8(0.0f, 0.5f, 1.0f, -0.7f, 2.0f, -1.5f, 3.0f, -3.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m256 x) { return dfm::sin(x); }, input, 2);
}

TEST(AvxSin, LargeRange) {
  __m256 input = make8(-3.14f, 3.14f, 6.28f, -6.28f, 10.0f, -10.0f, 100.0f, -100.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m256 x) { return dfm::sin(x); }, input, 2);
}

TEST(AvxCos, LaneByLane) {
  __m256 input = make8(0.0f, 0.5f, 1.0f, -0.7f, 2.0f, -1.5f, 3.0f, -3.0f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m256 x) { return dfm::cos(x); }, input, 2);
}

TEST(AvxCos, LargeRange) {
  __m256 input = make8(-3.14f, 3.14f, 6.28f, -6.28f, 10.0f, -10.0f, 100.0f, -100.0f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m256 x) { return dfm::cos(x); }, input, 2);
}

TEST(AvxTan, LaneByLane) {
  __m256 input = make8(0.0f, 0.3f, -0.5f, 1.0f, -0.3f, 0.5f, -1.0f, 0.1f);
  checkLaneByLane([](float x) { return dfm::tan(x); }, [](__m256 x) { return dfm::tan(x); }, input);
}

TEST(AvxExp, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 5.0f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m256 x) { return dfm::exp(x); }, input);
}

TEST(AvxExp2, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 3.5f, -3.5f, 10.0f, -10.0f, 0.5f);
  checkLaneByLane(
      [](float x) { return dfm::exp2(x); }, [](__m256 x) { return dfm::exp2(x); }, input);
}

TEST(AvxExp10, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 3.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp10(x); }, [](__m256 x) { return dfm::exp10(x); }, input);
}

TEST(AvxLog, LaneByLane) {
  __m256 input = make8(0.5f, 1.0f, 2.0f, 10.0f, 0.1f, 100.0f, 0.01f, 1000.0f);
  checkLaneByLane([](float x) { return dfm::log(x); }, [](__m256 x) { return dfm::log(x); }, input);
}

TEST(AvxLog2, LaneByLane) {
  __m256 input = make8(0.5f, 1.0f, 2.0f, 10.0f, 0.25f, 4.0f, 8.0f, 16.0f);
  checkLaneByLane(
      [](float x) { return dfm::log2(x); }, [](__m256 x) { return dfm::log2(x); }, input);
}

TEST(AvxLog10, LaneByLane) {
  __m256 input = make8(0.5f, 1.0f, 10.0f, 100.0f, 0.1f, 1000.0f, 0.01f, 10000.0f);
  checkLaneByLane(
      [](float x) { return dfm::log10(x); }, [](__m256 x) { return dfm::log10(x); }, input);
}

TEST(AvxAcos, LaneByLane) {
  __m256 input = make8(-1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f);
  checkLaneByLane(
      [](float x) { return dfm::acos(x); }, [](__m256 x) { return dfm::acos(x); }, input);
}

TEST(AvxAsin, LaneByLane) {
  __m256 input = make8(-1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 0.9f);
  checkLaneByLane(
      [](float x) { return dfm::asin(x); }, [](__m256 x) { return dfm::asin(x); }, input);
}

TEST(AvxAtan, LaneByLane) {
  __m256 input = make8(-10.0f, -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 10.0f, 100.0f);
  checkLaneByLane(
      [](float x) { return dfm::atan(x); }, [](__m256 x) { return dfm::atan(x); }, input);
}

TEST(AvxCbrt, LaneByLane) {
  __m256 input = make8(0.125f, 1.0f, 8.0f, 27.0f, 64.0f, 125.0f, 1000.0f, 0.001f);
  checkLaneByLane(
      [](float x) { return dfm::cbrt(x); }, [](__m256 x) { return dfm::cbrt(x); }, input);
}

// ---- Sweep tests: thorough range coverage to catch cross-lane bugs ----

TEST(AvxSin, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 4096);
  for (float base = static_cast<float>(-4.0 * M_PI); base < static_cast<float>(4.0 * M_PI);
       base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](__m256 x) { return dfm::sin(x); },
        makeSeq(base, delta),
        2);
  }
}

TEST(AvxCos, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 4096);
  for (float base = static_cast<float>(-4.0 * M_PI); base < static_cast<float>(4.0 * M_PI);
       base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
        [](__m256 x) { return dfm::cos(x); },
        makeSeq(base, delta),
        2);
  }
}

TEST(AvxTan, Sweep) {
  // Avoid exact ±π/2 where tan diverges.
  float delta = 0.001f;
  for (float base = -1.5f; base < 1.5f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
        [](__m256 x) { return dfm::tan(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(AvxExp, Sweep) {
  float delta = 0.01f;
  for (float base = -20.0f; base < 20.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](__m256 x) { return dfm::exp(x); },
        makeSeq(base, delta),
        5);
  }
}

TEST(AvxExp2, Sweep) {
  float delta = 0.01f;
  for (float base = -20.0f; base < 20.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
        [](__m256 x) { return dfm::exp2(x); },
        makeSeq(base, delta),
        1);
  }
}

TEST(AvxExp10, Sweep) {
  float delta = 0.005f;
  for (float base = -10.0f; base < 10.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
        [](__m256 x) { return dfm::exp10(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(AvxLog, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
        [](__m256 x) { return dfm::log(x); },
        makeSeq(base, delta),
        2);
    // Increase delta as we go to cover wider range without too many iterations.
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(AvxLog2, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
        [](__m256 x) { return dfm::log2(x); },
        makeSeq(base, delta),
        1);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(AvxLog10, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log10(static_cast<double>(x))); },
        [](__m256 x) { return dfm::log10(x); },
        makeSeq(base, delta),
        3);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(AvxAcos, Sweep) {
  float delta = 0.001f;
  for (float base = -0.999f; base < 0.999f; base += kLanes * delta) {
    float b = std::min(base, 0.999f - (kLanes - 1) * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
        [](__m256 x) { return dfm::acos(x); },
        makeSeq(b, delta),
        4);
  }
}

TEST(AvxAsin, Sweep) {
  float delta = 0.001f;
  for (float base = -0.999f; base < 0.999f; base += kLanes * delta) {
    float b = std::min(base, 0.999f - (kLanes - 1) * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
        [](__m256 x) { return dfm::asin(x); },
        makeSeq(b, delta),
        3);
  }
}

TEST(AvxAtan, Sweep) {
  float delta = 0.05f;
  for (float base = -100.0f; base < 100.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::atan(static_cast<double>(x))); },
        [](__m256 x) { return dfm::atan(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(AvxCbrt, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
        [](__m256 x) { return dfm::cbrt(x); },
        makeSeq(base, delta),
        12);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

// ---- Mixed-value tests to catch per-lane independence ----

TEST(AvxSin, MixedValues) {
  __m256 input = make8(
      0.0f,
      static_cast<float>(M_PI_2),
      static_cast<float>(-M_PI),
      100.0f,
      -100.0f,
      1e-6f,
      static_cast<float>(2.0 * M_PI),
      -0.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m256 x) { return dfm::sin(x); }, input, 2);
}

TEST(AvxExp, MixedValues) {
  __m256 input = make8(-20.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f, 20.0f, 0.001f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m256 x) { return dfm::exp(x); }, input);
}

TEST(AvxLog, MixedValues) {
  __m256 input = make8(1e-6f, 0.001f, 0.1f, 1.0f, 10.0f, 100.0f, 1e6f, 1e10f);
  checkLaneByLane([](float x) { return dfm::log(x); }, [](__m256 x) { return dfm::log(x); }, input);
}

// ---- frexp / ldexp ----

TEST(AvxFrexp, LaneByLane) {
  __m256 input = make8(1.0f, 2.0f, 0.5f, -4.0f, 8.0f, -0.25f, 1024.0f, 0.001f);
  dfm::IntType_t<__m256> eptr;
  __m256 mantissa = dfm::frexp(input, &eptr);

  for (int i = 0; i < kLanes; ++i) {
    int32_t scalar_exp;
    float scalar_m = dfm::frexp(lane(input, i), &scalar_exp);
    EXPECT_EQ(lane(mantissa, i), scalar_m) << "Lane " << i;
    EXPECT_EQ(lane(eptr, i), scalar_exp) << "Lane " << i;
  }
}

TEST(AvxFrexp, Sweep) {
  float delta = 0.5f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    __m256 input = makeSeq(base, delta);
    dfm::IntType_t<__m256> eptr;
    __m256 mantissa = dfm::frexp(input, &eptr);
    for (int i = 0; i < kLanes; ++i) {
      int32_t scalar_exp;
      float scalar_m = dfm::frexp(lane(input, i), &scalar_exp);
      EXPECT_EQ(lane(mantissa, i), scalar_m) << "base=" << base << " lane=" << i;
      EXPECT_EQ(lane(eptr, i), scalar_exp) << "base=" << base << " lane=" << i;
    }
    if (base > 100.0f)
      delta = 5.0f;
  }
}

TEST(AvxLdexp, LaneByLane) {
  __m256 input = make8(1.0f, 0.5f, 2.0f, -1.0f, 0.25f, -0.5f, 3.0f, -3.0f);
  AvxInt32 exp = AvxInt32(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  __m256 result = dfm::ldexp(input, exp);

  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

// ---- atan2 ----

TEST(AvxAtan2, LaneByLane) {
  __m256 y = make8(1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.0f);
  __m256 x = make8(1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f);
  __m256 result = dfm::atan2(y, x);

  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i)
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

TEST(AvxAtan2, Sweep) {
  // Sweep all quadrants.
  float delta = 0.1f;
  for (float ybase = -5.0f; ybase < 5.0f; ybase += 1.0f) {
    for (float xbase = -5.0f; xbase < 5.0f; xbase += 1.0f) {
      __m256 y = make8(
          ybase,
          ybase + delta,
          ybase - delta,
          ybase + 2 * delta,
          ybase - 2 * delta,
          ybase + 0.5f,
          ybase - 0.5f,
          ybase);
      __m256 x = make8(
          xbase,
          xbase - delta,
          xbase + delta,
          xbase - 2 * delta,
          xbase + 2 * delta,
          xbase - 0.5f,
          xbase + 0.5f,
          xbase);
      __m256 result = dfm::atan2(y, x);
      for (int i = 0; i < kLanes; ++i) {
        float expected = dfm::atan2(lane(y, i), lane(x, i));
        uint32_t dist = dfm::float_distance(expected, lane(result, i));
        EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
      }
    }
  }
}

// ---- Edge cases: NaN, Inf, denormals per lane ----

TEST(AvxEdge, SinSpecialValues) {
  __m256 input = make8(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max());
  __m256 result = dfm::sin(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = ::sinf(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else if (!std::isfinite(actual)) {
      // SIMD range reduction overflows for extreme inputs (e.g. float::max) — skip.
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 2u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(AvxEdge, ExpSpecialValues) {
  __m256 input = make8(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      88.7f, // near overflow
      -88.7f, // near underflow
      1e-10f);
  __m256 result = dfm::exp(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(AvxEdge, LogSpecialValues) {
  __m256 input = make8(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -1.0f, // negative → NaN
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max());
  __m256 result = dfm::log(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// ---- Accuracy and bounds trait variants ----

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// -- sin/cos accurate --

TEST(AvxSinAccurate, LaneByLane) {
  __m256 input = make8(0.0f, 0.5f, -1.0f, 2.5f, -3.0f, 1.5f, -0.3f, 3.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); },
      [](__m256 x) { return dfm::sin<__m256, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

TEST(AvxSinAccurate, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 4096);
  for (float base = static_cast<float>(-4.0 * M_PI); base < static_cast<float>(4.0 * M_PI);
       base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](__m256 x) { return dfm::sin<__m256, dfm::MaxAccuracyTraits>(x); },
        makeSeq(base, delta),
        2);
  }
}

TEST(AvxCosAccurate, LaneByLane) {
  __m256 input = make8(0.0f, 0.5f, -1.0f, 2.5f, -3.0f, 1.5f, -0.3f, 3.0f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); },
      [](__m256 x) { return dfm::cos<__m256, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

// -- exp accurate and bounds --

TEST(AvxExpAccurate, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 0.5f, -0.5f, 10.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp<float, dfm::MaxAccuracyTraits>(x); },
      [](__m256 x) { return dfm::exp<__m256, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(AvxExpAccurate, Sweep) {
  float delta = 0.01f;
  for (float base = -10.0f; base < 10.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return dfm::exp<float, dfm::MaxAccuracyTraits>(x); },
        [](__m256 x) { return dfm::exp<__m256, dfm::MaxAccuracyTraits>(x); },
        makeSeq(base, delta));
  }
}

TEST(AvxExpBounds, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 0.5f, -0.5f, 10.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp<float, BoundsTraits>(x); },
      [](__m256 x) { return dfm::exp<__m256, BoundsTraits>(x); },
      input);
}

TEST(AvxExpBounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  __m256 input = make8(nan, inf, -inf, -100.0f, 89.0f, 100.0f, -100.0f, 0.0f);
  __m256 result = dfm::exp<__m256, BoundsTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp<float, BoundsTraits>(lane(input, i));
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

TEST(AvxExp2Bounds, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 10.0f, -10.0f, 0.5f, 3.5f, -3.5f);
  checkLaneByLane(
      [](float x) { return dfm::exp2<float, BoundsTraits>(x); },
      [](__m256 x) { return dfm::exp2<__m256, BoundsTraits>(x); },
      input);
}

TEST(AvxExp2Bounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(nan, 128.0f, -150.0f, 0.0f, 1.0f, -1.0f, 127.0f, -126.0f);
  __m256 result = dfm::exp2<__m256, BoundsTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp2<float, BoundsTraits>(lane(input, i));
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

TEST(AvxExp10Bounds, LaneByLane) {
  __m256 input = make8(0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp10<float, BoundsTraits>(x); },
      [](__m256 x) { return dfm::exp10<__m256, BoundsTraits>(x); },
      input);
}

TEST(AvxExp10Bounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(nan, 39.0f, -39.0f, 0.0f, 1.0f, -1.0f, 38.0f, -38.0f);
  __m256 result = dfm::exp10<__m256, BoundsTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp10<float, BoundsTraits>(lane(input, i));
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

TEST(AvxLogAccurate, LaneByLane) {
  __m256 input = make8(0.5f, 1.0f, 2.0f, 100.0f, 0.01f, 1000.0f, 0.001f, 10.0f);
  checkLaneByLane(
      [](float x) { return dfm::log<float, dfm::MaxAccuracyTraits>(x); },
      [](__m256 x) { return dfm::log<__m256, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(AvxLogAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      0.5f);
  __m256 result = dfm::log<__m256, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(lane(input, i));
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

// -- log2 accurate --

TEST(AvxLog2Accurate, LaneByLane) {
  __m256 input = make8(0.5f, 1.0f, 2.0f, 1024.0f, 0.25f, 4.0f, 8.0f, 0.125f);
  checkLaneByLane(
      [](float x) { return dfm::log2<float, dfm::MaxAccuracyTraits>(x); },
      [](__m256 x) { return dfm::log2<__m256, dfm::MaxAccuracyTraits>(x); },
      input);
}

// -- cbrt accurate --

TEST(AvxCbrtAccurate, LaneByLane) {
  __m256 input = make8(0.125f, 1.0f, 8.0f, 27.0f, 64.0f, 125.0f, 1000.0f, 0.001f);
  checkLaneByLane(
      [](float x) { return dfm::cbrt<float, dfm::MaxAccuracyTraits>(x); },
      [](__m256 x) { return dfm::cbrt<__m256, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(AvxCbrtAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(
      0.0f, inf, nan, std::numeric_limits<float>::denorm_min(), 1.0f, 1000.0f, 0.001f, 100.0f);
  __m256 result = dfm::cbrt<__m256, dfm::MaxAccuracyTraits>(input);
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

// -- atan2 bounds --

TEST(AvxAtan2Bounds, LaneByLane) {
  __m256 y = make8(1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.0f);
  __m256 x = make8(1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f);
  __m256 result = dfm::atan2<__m256, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i;
  }
}

TEST(AvxAtan2Bounds, InfCases) {
  float inf = std::numeric_limits<float>::infinity();
  __m256 y = make8(inf, -inf, inf, -inf, 1.0f, -1.0f, inf, 0.0f);
  __m256 x = make8(inf, inf, -inf, -inf, inf, -inf, 1.0f, inf);
  __m256 result = dfm::atan2<__m256, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i)
                        << " expected=" << expected << " actual=" << actual;
  }
}

// ---- Additional edge case tests ----

TEST(AvxFrexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(
      0.0f,
      kInf,
      -kInf,
      kNaN,
      std::numeric_limits<float>::denorm_min(),
      -0.0f,
      std::numeric_limits<float>::min(),
      1.0f);
  dfm::IntType_t<__m256> eptr;
  __m256 mantissa = dfm::frexp(input, &eptr);
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

TEST(AvxLdexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(kInf, -kInf, kNaN, 0.0f, -0.0f, 1.0f, -1.0f, 0.5f);
  AvxInt32 exp = AvxInt32(_mm256_set_epi32(10, 5, 0, 3, 0, 7, 1, 2));
  __m256 result = dfm::ldexp(input, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(AvxEdge, TanSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m256 input = make8(0.0f, kNaN, kInf, -kInf, -0.0f, 1.0f, -1.0f, 0.5f);
  __m256 result = dfm::tan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::tan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(AvxEdge, AtanSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m256 input = make8(0.0f, kNaN, kInf, -kInf, 20000000.0f, -20000000.0f, 1e10f, -1e10f);
  __m256 result = dfm::atan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(AvxEdge, AsinOutOfRange) {
  __m256 input = make8(1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f);
  __m256 result = dfm::asin(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(AvxEdge, AcosOutOfRange) {
  __m256 input = make8(1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f);
  __m256 result = dfm::acos(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(AvxLogAccurate, NegativeInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m256 input = make8(-1.0f, -100.0f, -kInf, -0.0f, -0.5f, -1e10f, -1e-30f, -42.0f);
  __m256 result = dfm::log<__m256, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(AvxLog2Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input =
      make8(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min(), -1.0f, -kInf, -0.0f, 1.0f);
  __m256 result = dfm::log2<__m256, dfm::MaxAccuracyTraits>(input);
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

TEST(AvxLog10Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input =
      make8(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min(), -1.0f, -kInf, -0.0f, 10.0f);
  __m256 result = dfm::log10<__m256, dfm::MaxAccuracyTraits>(input);
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

TEST(AvxCbrtAccurate, NegativeInf) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m256 input = make8(-kInf, -8.0f, -27.0f, -1.0f, -0.001f, -1e10f, -1e-20f, -125.0f);
  __m256 result = dfm::cbrt<__m256, dfm::MaxAccuracyTraits>(input);
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

TEST(AvxExp2Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(kInf, -kInf, kNaN, 0.0f, 128.0f, -150.0f, 1.0f, -1.0f);
  __m256 result = dfm::exp2<__m256, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp2<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(AvxExp10Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m256 input = make8(kInf, -kInf, kNaN, 0.0f, 39.0f, -39.0f, 1.0f, -1.0f);
  __m256 result = dfm::exp10<__m256, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp10<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// --- hypot ---

TEST(AvxHypot, LaneByLane) {
  __m256 x = make8(3.0f, 0.0f, 1e30f, -5.0f, 1.0f, 1e-20f, -1e15f, 7.0f);
  __m256 y = make8(4.0f, 1.0f, 1e30f, 12.0f, 1.0f, 1e-20f, 1e15f, 24.0f);
  __m256 result = dfm::hypot(x, y);
  for (int i = 0; i < kLanes; ++i) {
    double xd = static_cast<double>(lane(x, i));
    double yd = static_cast<double>(lane(y, i));
    float expected = static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 2u) << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

// -- hypot bounds --

TEST(AvxHypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 x = make8(inf, -inf, nan, nan, inf, 3.0f, -inf, nan);
  __m256 y = make8(3.0f, nan, inf, -inf, inf, inf, 0.0f, -inf);
  __m256 result = dfm::hypot<__m256, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isinf(lane(result, i)) && lane(result, i) > 0)
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
        << " result=" << lane(result, i);
  }
}

TEST(AvxHypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 x = make8(nan, 3.0f, nan, nan, 0.0f, nan, -1.0f, nan);
  __m256 y = make8(3.0f, nan, 0.0f, nan, nan, -5.0f, nan, nan);
  __m256 result = dfm::hypot<__m256, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
        << " result=" << lane(result, i);
  }
}

// --- pow ---

static float gt_pow(float x, float y) {
  return static_cast<float>(std::pow(static_cast<double>(x), static_cast<double>(y)));
}

TEST(AvxPow, LaneByLane) {
  __m256 base = make8(2.0f, 3.0f, 4.0f, 0.5f, 8.0f, 10.0f, 0.25f, 100.0f);
  __m256 exp = make8(3.0f, 2.0f, 0.5f, -1.0f, 1.0f / 3.0f, 0.0f, -2.0f, 0.5f);
  __m256 result = dfm::pow(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(base, i), lane(exp, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 2u) << "Lane " << i << ": base=" << lane(base, i) << " exp=" << lane(exp, i)
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

TEST(AvxPow, NegativeBase) {
  __m256 base = make8(-2.0f, -1.0f, -3.0f, -0.5f, -4.0f, -8.0f, -2.0f, -1.0f);
  __m256 exp = make8(3.0f, 2.0f, 5.0f, -1.0f, 2.0f, 3.0f, 4.0f, 0.0f);
  __m256 result = dfm::pow(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(base, i), lane(exp, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 3u) << "Lane " << i << ": base=" << lane(base, i) << " exp=" << lane(exp, i)
                          << " expected=" << expected << " actual=" << lane(result, i);
    }
  }
}

TEST(AvxPow, NegBaseNonInt) {
  __m256 base = make8(-2.0f, -1.0f, -3.0f, -0.5f, -4.0f, -8.0f, -2.0f, -1.0f);
  __m256 exp = make8(0.5f, 1.5f, 2.5f, 0.3f, -0.5f, -1.5f, 0.1f, -0.7f);
  __m256 result = dfm::pow(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": base=" << lane(base, i) << " exp=" << lane(exp, i)
        << " result=" << lane(result, i);
  }
}

TEST(AvxPow, YZero) {
  __m256 base = make8(0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.5f, -0.5f, 42.0f);
  __m256 exp = _mm256_setzero_ps();
  __m256 result = dfm::pow(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 1.0f)
        << "Lane " << i << ": base=" << lane(base, i) << " expected=1.0"
        << " actual=" << lane(result, i);
  }
}

TEST(AvxPow, ScalarExp) {
  __m256 base = make8(1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 10.0f, 0.1f, 7.0f);
  __m256 exp = _mm256_set1_ps(2.0f);
  __m256 result = dfm::pow(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    float b = lane(base, i);
    float expected = b * b;
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 2u) << "Lane " << i << ": base=" << b << " expected=" << expected
                        << " actual=" << lane(result, i);
  }
}

// -- pow bounds --

TEST(AvxPowBounds, Specials) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  __m256 base = make8(0.0f, -0.0f, inf, nan, 1.0f, -1.0f, 0.5f, 2.0f);
  __m256 exp = make8(2.0f, 3.0f, -1.0f, 0.0f, nan, inf, inf, -inf);
  __m256 result = dfm::pow<__m256, dfm::MaxAccuracyTraits>(base, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = gt_pow(lane(base, i), lane(exp, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i << ": base=" << lane(base, i)
                                      << " exp=" << lane(exp, i) << " actual=" << actual;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i << ": base=" << lane(base, i)
                                      << " exp=" << lane(exp, i) << " actual=" << actual;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 2u) << "Lane " << i << ": base=" << lane(base, i) << " exp=" << lane(exp, i)
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

// --- expm1 ---

TEST(AvxExpm1, LaneByLane) {
  __m256 input = make8(0.0f, 0.001f, -0.5f, 1.0f, -1.0f, 0.1f, -0.1f, 5.0f);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::expm1(static_cast<double>(x))); },
      [](__m256 x) { return dfm::expm1(x); },
      input,
      2);
}

// --- log1p ---

TEST(AvxLog1p, LaneByLane) {
  __m256 input = make8(0.0f, 0.001f, 0.5f, 1.0f, 10.0f, 100.0f, 1e-6f, 1e-3f);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::log1p(static_cast<double>(x))); },
      [](__m256 x) { return dfm::log1p(x); },
      input,
      2);
}

// --- tanh ---

TEST(AvxTanh, LaneByLane) {
  __m256 input = make8(0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 5.0f, -5.0f, 0.01f);
  checkLaneByLane(
      [](float x) { return static_cast<float>(::tanh(static_cast<double>(x))); },
      [](__m256 x) { return dfm::tanh(x); },
      input,
      2);
}

#else // !defined(__AVX2__)

// Dummy test so the binary has at least one test on non-AVX2 platforms.
TEST(AvxFloat, NotAvailable) {
  GTEST_SKIP() << "AVX2 not available";
}

#endif // defined(__AVX2__)
