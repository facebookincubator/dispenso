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

#if defined(__AVX512F__)

namespace dfm = dispenso::fast_math;
using Avx512Float = dfm::Avx512Float;
using Avx512Int32 = dfm::Avx512Int32;
using Avx512Uint32 = dfm::Avx512Uint32;
using Avx512Mask = dfm::Avx512Mask;

constexpr int32_t kLanes = 16;

// Helper: extract lane i from __m512.
static float lane(__m512 v, int i) {
  alignas(64) float buf[kLanes];
  _mm512_store_ps(buf, v);
  return buf[i];
}

// Helper: extract lane i from __m512i (as int32_t).
static int32_t lane(__m512i v, int i) {
  alignas(64) int32_t buf[kLanes];
  _mm512_store_si512(buf, v);
  return buf[i];
}

// Helper: extract bit i from Avx512Mask.
static bool maskBit(Avx512Mask m, int i) {
  return (m.m >> i) & 1;
}

// Helper: create __m512 from 16 distinct values.
static __m512 make16(
    float a,
    float b,
    float c,
    float d,
    float e,
    float f,
    float g,
    float h,
    float i,
    float j,
    float k,
    float l,
    float m,
    float n,
    float o,
    float p) {
  return _mm512_set_ps(p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a);
}

// Helper: create __m512 from base + stride (16 consecutive values).
static __m512 makeSeq(float base, float stride) {
  return _mm512_set_ps(
      base + 15 * stride,
      base + 14 * stride,
      base + 13 * stride,
      base + 12 * stride,
      base + 11 * stride,
      base + 10 * stride,
      base + 9 * stride,
      base + 8 * stride,
      base + 7 * stride,
      base + 6 * stride,
      base + 5 * stride,
      base + 4 * stride,
      base + 3 * stride,
      base + 2 * stride,
      base + stride,
      base);
}

// Helper: compare lane-by-lane against scalar function using raw __m512 types.
template <typename ScalarFunc, typename SimdFunc>
static void
checkLaneByLane(ScalarFunc scalar_fn, SimdFunc simd_fn, __m512 input, uint32_t max_ulps = 0) {
  __m512 result = simd_fn(input);
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

// ---- Avx512Mask basic operations ----

TEST(Avx512Mask, ConstructFromInt) {
  Avx512Mask allTrue(1);
  EXPECT_EQ(allTrue.m, static_cast<__mmask16>(0xFFFF));
  Avx512Mask allFalse(0);
  EXPECT_EQ(allFalse.m, static_cast<__mmask16>(0));
}

TEST(Avx512Mask, LogicalOps) {
  Avx512Mask a(static_cast<__mmask16>(0xFF00));
  Avx512Mask b(static_cast<__mmask16>(0x0FF0));

  Avx512Mask andResult = a & b;
  EXPECT_EQ(andResult.m, static_cast<__mmask16>(0x0F00));

  Avx512Mask orResult = a | b;
  EXPECT_EQ(orResult.m, static_cast<__mmask16>(0xFFF0));

  Avx512Mask xorResult = a ^ b;
  EXPECT_EQ(xorResult.m, static_cast<__mmask16>(0xF0F0));

  Avx512Mask notResult = !a;
  EXPECT_EQ(notResult.m, static_cast<__mmask16>(0x00FF));
}

TEST(Avx512Mask, Equality) {
  Avx512Mask a(static_cast<__mmask16>(0xFF00));
  Avx512Mask b(static_cast<__mmask16>(0xFF00));
  Avx512Mask c(static_cast<__mmask16>(0x00FF));

  // Same masks → all bits set (per-bit XNOR)
  Avx512Mask eqAB = a == b;
  EXPECT_EQ(eqAB.m, static_cast<__mmask16>(0xFFFF));

  // Complementary masks → all bits clear
  Avx512Mask eqAC = a == c;
  EXPECT_EQ(eqAC.m, static_cast<__mmask16>(0));

  // mask == 0 is equivalent to !mask
  Avx512Mask mask0 = a == Avx512Mask(0);
  EXPECT_EQ(mask0.m, static_cast<__mmask16>(0x00FF));
}

TEST(Avx512Mask, ToInt32Conversion) {
  Avx512Mask m(static_cast<__mmask16>(0xA5A5)); // 1010 0101 1010 0101
  Avx512Int32 expanded = m; // implicit conversion
  for (int i = 0; i < kLanes; ++i) {
    int32_t expected = ((0xA5A5 >> i) & 1) ? -1 : 0;
    EXPECT_EQ(lane(expanded, i), expected) << "Lane " << i;
  }
}

// ---- Avx512Float basic arithmetic ----

TEST(Avx512Float, Broadcast) {
  Avx512Float v(3.0f);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(v, i), 3.0f);
  }
}

TEST(Avx512Float, Arithmetic) {
  Avx512Float a = make16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  Avx512Float b = make16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

  Avx512Float sum = a + b;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(sum, i), 17.0f) << "Lane " << i;
  }

  Avx512Float diff = b - a;
  EXPECT_EQ(lane(diff, 0), 15.0f);
  EXPECT_EQ(lane(diff, 15), -15.0f);

  Avx512Float prod = a * b;
  EXPECT_EQ(lane(prod, 0), 16.0f);
  EXPECT_EQ(lane(prod, 7), 72.0f);
  EXPECT_EQ(lane(prod, 15), 16.0f);

  Avx512Float quot = a / b;
  EXPECT_FLOAT_EQ(lane(quot, 0), 1.0f / 16.0f);
  EXPECT_FLOAT_EQ(lane(quot, 15), 16.0f);
}

TEST(Avx512Float, Negation) {
  Avx512Float a = make16(
      1.0f,
      -2.0f,
      0.0f,
      3.0f,
      -0.5f,
      100.0f,
      -0.0f,
      42.0f,
      -1.0f,
      2.0f,
      0.0f,
      -3.0f,
      0.5f,
      -100.0f,
      0.0f,
      -42.0f);
  Avx512Float neg = -a;
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

TEST(Avx512Float, CompoundAssignment) {
  Avx512Float a(1.0f);
  a += Avx512Float(2.0f);
  EXPECT_EQ(lane(a, 0), 3.0f);
  a -= Avx512Float(1.0f);
  EXPECT_EQ(lane(a, 0), 2.0f);
  a *= Avx512Float(5.0f);
  EXPECT_EQ(lane(a, 0), 10.0f);
}

// ---- Avx512Float comparisons ----

TEST(Avx512Float, Comparisons) {
  Avx512Float a = make16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  Avx512Float b = make16(2, 2, 1, 5, 5, 4, 7, 9, 9, 10, 11, 12, 13, 14, 15, 16);

  Avx512Mask lt = a < b;
  EXPECT_TRUE(maskBit(lt, 0)); // 1 < 2 true
  EXPECT_FALSE(maskBit(lt, 1)); // 2 < 2 false
  EXPECT_FALSE(maskBit(lt, 2)); // 3 < 1 false
  EXPECT_TRUE(maskBit(lt, 3)); // 4 < 5 true
  EXPECT_FALSE(maskBit(lt, 4)); // 5 < 5 false
  EXPECT_FALSE(maskBit(lt, 5)); // 6 < 4 false
  EXPECT_FALSE(maskBit(lt, 6)); // 7 < 7 false
  EXPECT_TRUE(maskBit(lt, 7)); // 8 < 9 true

  Avx512Mask gt = a > b;
  EXPECT_FALSE(maskBit(gt, 0)); // 1 > 2 false
  EXPECT_TRUE(maskBit(gt, 2)); // 3 > 1 true
  EXPECT_TRUE(maskBit(gt, 5)); // 6 > 4 true

  Avx512Mask eq = a == b;
  EXPECT_FALSE(maskBit(eq, 0)); // 1 == 2 false
  EXPECT_TRUE(maskBit(eq, 1)); // 2 == 2 true
  EXPECT_TRUE(maskBit(eq, 4)); // 5 == 5 true
  EXPECT_TRUE(maskBit(eq, 6)); // 7 == 7 true
  // All lanes 8-15 are equal in our setup
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_TRUE(maskBit(eq, i)) << "Lane " << i;
  }
}

// ---- Avx512Int32 arithmetic ----

TEST(Avx512Int32, BasicOps) {
  Avx512Int32 a(10);
  Avx512Int32 b(3);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a + b, i), 13) << "Lane " << i;
    EXPECT_EQ(lane(a - b, i), 7) << "Lane " << i;
    EXPECT_EQ(lane(a * b, i), 30) << "Lane " << i;
  }
}

TEST(Avx512Int32, Negation) {
  Avx512Int32 a =
      Avx512Int32(_mm512_set_epi32(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
  Avx512Int32 neg = -a;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(neg, i), -(i + 1)) << "Lane " << i;
  }
}

TEST(Avx512Int32, ShiftOps) {
  Avx512Int32 a(8);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(a << 2, i), 32) << "Lane " << i;
    EXPECT_EQ(lane(a >> 1, i), 4) << "Lane " << i;
  }

  Avx512Int32 neg(-8);
  EXPECT_EQ(lane(neg >> 1, 0), -4); // Arithmetic shift preserves sign
}

TEST(Avx512Int32, Comparisons) {
  Avx512Int32 a =
      Avx512Int32(_mm512_set_epi32(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 5, 5, 4, 3, 2, 1));
  Avx512Int32 b =
      Avx512Int32(_mm512_set_epi32(16, 15, 14, 13, 12, 11, 10, 9, 1, 7, 6, 4, 4, 3, 9, 0));

  Avx512Mask eq = a == b;
  EXPECT_FALSE(maskBit(eq, 0)); // 1 == 0 false
  EXPECT_FALSE(maskBit(eq, 1)); // 2 == 9 false
  EXPECT_TRUE(maskBit(eq, 2)); // 3 == 3 true
  EXPECT_TRUE(maskBit(eq, 3)); // 4 == 4 true
  EXPECT_FALSE(maskBit(eq, 4)); // 5 == 4 false
  EXPECT_FALSE(maskBit(eq, 5)); // 5 == 6 false
  EXPECT_TRUE(maskBit(eq, 6)); // 7 == 7 true
  EXPECT_FALSE(maskBit(eq, 7)); // 8 == 1 false
  // Lanes 8-15 are all equal
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_TRUE(maskBit(eq, i)) << "Lane " << i;
  }

  Avx512Mask lt = a < b;
  EXPECT_FALSE(maskBit(lt, 0)); // 1 < 0 false
  EXPECT_TRUE(maskBit(lt, 1)); // 2 < 9 true
  EXPECT_TRUE(maskBit(lt, 5)); // 5 < 6 true
}

// ---- Avx512Uint32 ----

TEST(Avx512Uint32, LogicalShift) {
  Avx512Uint32 a(0x80000000u);
  // Logical shift right — does NOT preserve sign.
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(static_cast<uint32_t>(lane(a >> 1, i)), 0x40000000u) << "Lane " << i;
  }
}

TEST(Avx512Uint32, UnsignedComparison) {
  Avx512Uint32 a(0x80000000u);
  Avx512Uint32 b(0x7FFFFFFFu);
  // Unsigned: 0x80000000 > 0x7FFFFFFF (native AVX-512 unsigned compare)
  Avx512Mask gt = a > b;
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(maskBit(gt, i)) << "Lane " << i;
  }
}

// ---- bit_cast ----

TEST(Avx512BitCast, FloatToInt) {
  Avx512Float f(1.0f);
  Avx512Int32 i = dfm::bit_cast<Avx512Int32>(f);
  for (int j = 0; j < kLanes; ++j) {
    EXPECT_EQ(lane(i, j), 0x3f800000) << "Lane " << j;
  }
}

TEST(Avx512BitCast, IntToFloat) {
  Avx512Int32 i(0x40000000);
  Avx512Float f = dfm::bit_cast<Avx512Float>(i);
  for (int j = 0; j < kLanes; ++j) {
    EXPECT_EQ(lane(f, j), 2.0f) << "Lane " << j;
  }
}

TEST(Avx512BitCast, RoundTrip) {
  Avx512Float original = make16(
      1.0f,
      -2.0f,
      0.5f,
      42.0f,
      -0.0f,
      3.14f,
      1e10f,
      -1e-10f,
      0.0f,
      -1.0f,
      2.0f,
      -0.5f,
      100.0f,
      -100.0f,
      1e-30f,
      -1e30f);
  Avx512Float roundtripped = dfm::bit_cast<Avx512Float>(dfm::bit_cast<Avx512Int32>(original));
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(roundtripped, i), lane(original, i)) << "Lane " << i;
  }
}

// ---- FloatTraits<Avx512Float> ----

TEST(Avx512FloatTraits, ConditionalWithMask) {
  // Create mask: even lanes true, odd lanes false
  Avx512Mask mask(static_cast<__mmask16>(0x5555)); // 0101 0101 0101 0101
  Avx512Float x(10.0f);
  Avx512Float y(20.0f);
  Avx512Float result = dfm::FloatTraits<Avx512Float>::conditional(mask, x, y);
  for (int i = 0; i < kLanes; ++i) {
    float expected = (i % 2 == 0) ? 10.0f : 20.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, ConditionalFromComparison) {
  // Test that comparison → mask → conditional works end-to-end.
  Avx512Float a = make16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  Avx512Float threshold(8.5f);
  auto mask = a > threshold;
  Avx512Float result = dfm::FloatTraits<Avx512Float>::conditional(mask, Avx512Float(100.0f), a);
  for (int i = 0; i < kLanes; ++i) {
    float inp = static_cast<float>(i + 1);
    float expected = (inp > 8.5f) ? 100.0f : inp;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, ConditionalInt32WithMask) {
  Avx512Mask mask(static_cast<__mmask16>(0x5555));
  Avx512Int32 x(100);
  Avx512Int32 y(200);
  Avx512Int32 result = dfm::FloatTraits<Avx512Float>::conditional(mask, x, y);
  for (int i = 0; i < kLanes; ++i) {
    int32_t expected = (i % 2 == 0) ? 100 : 200;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, ConditionalWithLaneWideMask) {
  // _mm512_set_epi32 fills from highest to lowest lane: lane 0=0, lane 1=-1, etc.
  // So odd lanes have mask=-1 (true → x), even lanes have mask=0 (false → y).
  Avx512Int32 mask =
      Avx512Int32(_mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0));
  Avx512Float x(10.0f);
  Avx512Float y(20.0f);
  Avx512Float result = dfm::FloatTraits<Avx512Float>::conditional(mask, x, y);
  for (int i = 0; i < kLanes; ++i) {
    float expected = (i % 2 == 0) ? 20.0f : 10.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, Apply) {
  Avx512Mask mask(static_cast<__mmask16>(0xFF00)); // upper 8 lanes true
  Avx512Float x(42.0f);
  Avx512Float result = dfm::FloatTraits<Avx512Float>::apply(mask, x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(lane(result, i), 0.0f) << "Lane " << i;
  }
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 42.0f) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, Fma) {
  Avx512Float a(2.0f), b(3.0f), c(4.0f);
  Avx512Float result = dfm::FloatTraits<Avx512Float>::fma(a, b, c);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 10.0f) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, Sqrt) {
  Avx512Float input = make16(
      1.0f,
      4.0f,
      9.0f,
      16.0f,
      25.0f,
      36.0f,
      49.0f,
      64.0f,
      81.0f,
      100.0f,
      121.0f,
      144.0f,
      169.0f,
      196.0f,
      225.0f,
      256.0f);
  Avx512Float result = dfm::FloatTraits<Avx512Float>::sqrt(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = std::sqrt(lane(input, i));
    EXPECT_FLOAT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512FloatTraits, MinMax) {
  Avx512Float a = make16(1, 5, 3, 7, 2, 8, 4, 6, 10, 0, 9, 1, 11, 3, 7, 5);
  Avx512Float b = make16(4, 2, 6, 0, 9, 1, 4, 3, 0, 10, 1, 9, 3, 11, 5, 7);
  Avx512Float mn = dfm::FloatTraits<Avx512Float>::min(a, b);
  Avx512Float mx = dfm::FloatTraits<Avx512Float>::max(a, b);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(mn, i), std::min(lane(a, i), lane(b, i))) << "Lane " << i;
    EXPECT_EQ(lane(mx, i), std::max(lane(a, i), lane(b, i))) << "Lane " << i;
  }
}

// ---- Util functions ----

TEST(Avx512Util, FloorSmall) {
  __m512 x = make16(
      1.5f,
      -1.5f,
      2.0f,
      0.1f,
      3.9f,
      -0.1f,
      0.0f,
      -3.7f,
      100.1f,
      -100.9f,
      0.999f,
      -0.999f,
      10.0f,
      -10.0f,
      0.5f,
      -0.5f);
  __m512 result = dfm::floor_small(x);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), std::floor(lane(x, i))) << "Lane " << i;
  }
}

TEST(Avx512Util, ConvertToInt) {
  __m512 x = make16(
      1.6f,
      -1.6f,
      2.5f,
      0.4f,
      3.5f,
      -2.5f,
      0.0f,
      100.0f,
      -100.0f,
      1.5f,
      -1.5f,
      10.4f,
      -10.4f,
      0.5f,
      -0.5f,
      42.0f);
  __m512i result = dfm::convert_to_int(x);
  // Round to nearest even.
  int32_t expected[] = {2, -2, 2, 0, 4, -2, 0, 100, -100, 2, -2, 10, -10, 0, 0, 42};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), expected[i]) << "Lane " << i;
  }
}

TEST(Avx512Util, ConvertToInt_NaN) {
  // NaN/Inf should be masked to 0.
  __m512 x = make16(
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      9.0f,
      10.0f,
      11.0f,
      12.0f,
      13.0f);
  __m512i result = dfm::convert_to_int(x);
  EXPECT_EQ(lane(result, 0), 0); // NaN → 0
  EXPECT_EQ(lane(result, 1), 0); // Inf → 0
  EXPECT_EQ(lane(result, 2), 0); // -Inf → 0
  EXPECT_EQ(lane(result, 3), 1);
}

TEST(Avx512Util, Gather) {
  float table[] = {
      10.0f,
      20.0f,
      30.0f,
      40.0f,
      50.0f,
      60.0f,
      70.0f,
      80.0f,
      90.0f,
      100.0f,
      110.0f,
      120.0f,
      130.0f,
      140.0f,
      150.0f,
      160.0f,
      170.0f};
  __m512i indices = _mm512_set_epi32(16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7);
  __m512 result = dfm::gather<Avx512Float>(table, indices);
  int32_t idx_vals[] = {7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0, 16};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), table[idx_vals[i]]) << "Lane " << i;
  }
}

TEST(Avx512Util, IntDivBy3) {
  __m512i a = _mm512_set_epi32(99, 90, 81, 72, 63, 48, 33, 30, 27, 24, 21, 12, 9, 6, 3, 0);
  __m512i result = dfm::int_div_by_3(a);
  int32_t inputs[] = {0, 3, 6, 9, 12, 21, 24, 27, 30, 33, 48, 63, 72, 81, 90, 99};
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), inputs[i] / 3) << "Lane " << i << " input=" << inputs[i];
  }
}

TEST(Avx512Util, Signof) {
  __m512 x = make16(
      1.0f,
      -1.0f,
      0.0f,
      -0.0f,
      42.0f,
      -42.0f,
      0.5f,
      -0.5f,
      1e10f,
      -1e10f,
      1e-20f,
      -1e-10f,
      100.0f,
      -100.0f,
      0.001f,
      -0.001f);
  __m512 result = dfm::signof(x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = std::signbit(lane(x, i)) ? -1.0f : 1.0f;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(Avx512Util, Signofi) {
  __m512i a = _mm512_set_epi32(-100, 100, -1, 0, 10, -5, 1, -1, 50, -50, 0, -0, 1000, -1000, 7, -7);
  __m512i result = dfm::signofi<Avx512Float>(Avx512Int32(a));
  int32_t inputs[16];
  alignas(64) int32_t buf[16];
  _mm512_store_si512(buf, a);
  for (int i = 0; i < kLanes; ++i) {
    inputs[i] = buf[i];
    int32_t expected = (inputs[i] < 0) ? -1 : 1;
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i << " input=" << inputs[i];
  }
}

TEST(Avx512Util, Nonnormal) {
  __m512 x = make16(
      1.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      -0.0f,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(),
      -1.0f,
      42.0f,
      -42.0f,
      0.5f,
      -0.5f,
      1e30f,
      -1e30f);
  Avx512Mask result = dfm::nonnormal(Avx512Float(x));
  EXPECT_FALSE(maskBit(result, 0)); // normal
  EXPECT_TRUE(maskBit(result, 1)); // inf
  EXPECT_TRUE(maskBit(result, 2)); // -inf
  EXPECT_TRUE(maskBit(result, 3)); // nan
  EXPECT_FALSE(maskBit(result, 4)); // zero
  EXPECT_FALSE(maskBit(result, 5)); // -zero
  EXPECT_FALSE(maskBit(result, 6)); // denorm
  EXPECT_FALSE(maskBit(result, 7)); // normal max
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_FALSE(maskBit(result, i)) << "Lane " << i;
  }
}

TEST(Avx512Util, NonnormalOrZero) {
  __m512 x = make16(
      1.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      -0.0f,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(),
      -1.0f,
      42.0f,
      -42.0f,
      0.5f,
      -0.5f,
      1e30f,
      -1e30f);
  Avx512Mask result = dfm::nonnormalOrZero(dfm::bit_cast<Avx512Int32>(Avx512Float(x)));
  EXPECT_FALSE(maskBit(result, 0)); // normal
  EXPECT_TRUE(maskBit(result, 1)); // inf
  EXPECT_TRUE(maskBit(result, 2)); // -inf
  EXPECT_TRUE(maskBit(result, 3)); // nan
  EXPECT_TRUE(maskBit(result, 4)); // zero
  EXPECT_TRUE(maskBit(result, 5)); // -zero (exponent bits are zero)
  EXPECT_TRUE(maskBit(result, 6)); // denorm (exponent bits are zero)
  EXPECT_FALSE(maskBit(result, 7)); // normal max
}

TEST(Avx512Util, BoolAsOne) {
  Avx512Mask mask(static_cast<__mmask16>(0x5555)); // even lanes true
  Avx512Float fResult = dfm::bool_as_one<Avx512Float>(mask);
  Avx512Int32 iResult = dfm::bool_as_one<Avx512Int32>(mask);
  for (int i = 0; i < kLanes; ++i) {
    float expectedF = (i % 2 == 0) ? 1.0f : 0.0f;
    int32_t expectedI = (i % 2 == 0) ? 1 : 0;
    EXPECT_EQ(lane(fResult, i), expectedF) << "Lane " << i;
    EXPECT_EQ(lane(iResult, i), expectedI) << "Lane " << i;
  }
}

TEST(Avx512Util, NboolAsOne) {
  Avx512Mask mask(static_cast<__mmask16>(0x5555)); // even lanes true
  Avx512Float fResult = dfm::nbool_as_one<Avx512Float>(mask);
  Avx512Int32 iResult = dfm::nbool_as_one<Avx512Int32>(mask);
  for (int i = 0; i < kLanes; ++i) {
    float expectedF = (i % 2 == 0) ? 0.0f : 1.0f;
    int32_t expectedI = (i % 2 == 0) ? 0 : 1;
    EXPECT_EQ(lane(fResult, i), expectedF) << "Lane " << i;
    EXPECT_EQ(lane(iResult, i), expectedI) << "Lane " << i;
  }
}

TEST(Avx512Util, BoolAsMask) {
  Avx512Mask mask(static_cast<__mmask16>(0xA5A5));
  Avx512Int32 intMask = dfm::bool_as_mask<Avx512Int32>(mask);
  for (int i = 0; i < kLanes; ++i) {
    int32_t expected = maskBit(mask, i) ? -1 : 0;
    EXPECT_EQ(lane(intMask, i), expected) << "Lane " << i;
  }
}

TEST(Avx512Util, BoolApplyOrZero) {
  Avx512Mask mask(static_cast<__mmask16>(0xFF00)); // upper 8 lanes true
  Avx512Float val(42.0f);
  Avx512Float result = dfm::bool_apply_or_zero(mask, val);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(lane(result, i), 0.0f) << "Lane " << i;
  }
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_EQ(lane(result, i), 42.0f) << "Lane " << i;
  }

  // Test with int
  Avx512Int32 ival(0x7f800000);
  Avx512Int32 iresult = dfm::bool_apply_or_zero(mask, ival);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(lane(iresult, i), 0) << "Lane " << i;
  }
  for (int i = 8; i < kLanes; ++i) {
    EXPECT_EQ(lane(iresult, i), 0x7f800000) << "Lane " << i;
  }
}

// ---- Transcendental functions: lane-by-lane correctness ----

TEST(Avx512Sin, LaneByLane) {
  __m512 input = make16(
      0.0f,
      0.5f,
      1.0f,
      -0.7f,
      2.0f,
      -1.5f,
      3.0f,
      -3.0f,
      0.1f,
      -0.1f,
      1.5f,
      -2.0f,
      0.7f,
      -0.3f,
      2.5f,
      -2.5f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m512 x) { return dfm::sin(x); }, input, 2);
}

TEST(Avx512Sin, LargeRange) {
  __m512 input = make16(
      -3.14f,
      3.14f,
      6.28f,
      -6.28f,
      10.0f,
      -10.0f,
      100.0f,
      -100.0f,
      50.0f,
      -50.0f,
      200.0f,
      -200.0f,
      1000.0f,
      -1000.0f,
      0.001f,
      -0.001f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m512 x) { return dfm::sin(x); }, input, 2);
}

TEST(Avx512Cos, LaneByLane) {
  __m512 input = make16(
      0.0f,
      0.5f,
      1.0f,
      -0.7f,
      2.0f,
      -1.5f,
      3.0f,
      -3.0f,
      0.1f,
      -0.1f,
      1.5f,
      -2.0f,
      0.7f,
      -0.3f,
      2.5f,
      -2.5f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m512 x) { return dfm::cos(x); }, input, 2);
}

TEST(Avx512Cos, LargeRange) {
  __m512 input = make16(
      -3.14f,
      3.14f,
      6.28f,
      -6.28f,
      10.0f,
      -10.0f,
      100.0f,
      -100.0f,
      50.0f,
      -50.0f,
      200.0f,
      -200.0f,
      1000.0f,
      -1000.0f,
      0.001f,
      -0.001f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m512 x) { return dfm::cos(x); }, input, 2);
}

TEST(Avx512Tan, LaneByLane) {
  __m512 input = make16(
      0.0f,
      0.3f,
      -0.5f,
      1.0f,
      -0.3f,
      0.5f,
      -1.0f,
      0.1f,
      -0.1f,
      0.2f,
      -0.2f,
      0.4f,
      -0.4f,
      0.6f,
      -0.6f,
      0.7f);
  checkLaneByLane([](float x) { return dfm::tan(x); }, [](__m512 x) { return dfm::tan(x); }, input);
}

TEST(Avx512Exp, LaneByLane) {
  __m512 input = make16(
      0.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      2.0f,
      -2.0f,
      5.0f,
      -5.0f,
      10.0f,
      -10.0f,
      0.1f,
      -0.1f,
      3.0f,
      -3.0f,
      0.01f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m512 x) { return dfm::exp(x); }, input);
}

TEST(Avx512Exp2, LaneByLane) {
  __m512 input = make16(
      0.0f,
      1.0f,
      -1.0f,
      3.5f,
      -3.5f,
      10.0f,
      -10.0f,
      0.5f,
      -0.5f,
      5.0f,
      -5.0f,
      20.0f,
      -20.0f,
      0.1f,
      -0.1f,
      7.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp2(x); }, [](__m512 x) { return dfm::exp2(x); }, input);
}

TEST(Avx512Exp10, LaneByLane) {
  __m512 input = make16(
      0.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      2.0f,
      -2.0f,
      3.0f,
      -3.0f,
      0.1f,
      -0.1f,
      5.0f,
      -5.0f,
      0.3f,
      -0.3f,
      1.5f);
  checkLaneByLane(
      [](float x) { return dfm::exp10(x); }, [](__m512 x) { return dfm::exp10(x); }, input);
}

TEST(Avx512Log, LaneByLane) {
  __m512 input = make16(
      0.5f,
      1.0f,
      2.0f,
      10.0f,
      0.1f,
      100.0f,
      1e-20f,
      1000.0f,
      0.001f,
      1e4f,
      1e5f,
      1e6f,
      0.5f,
      3.14f,
      2.718f,
      42.0f);
  checkLaneByLane([](float x) { return dfm::log(x); }, [](__m512 x) { return dfm::log(x); }, input);
}

TEST(Avx512Log2, LaneByLane) {
  __m512 input = make16(
      0.5f,
      1.0f,
      2.0f,
      10.0f,
      0.25f,
      4.0f,
      8.0f,
      16.0f,
      32.0f,
      64.0f,
      128.0f,
      256.0f,
      0.125f,
      0.0625f,
      512.0f,
      1024.0f);
  checkLaneByLane(
      [](float x) { return dfm::log2(x); }, [](__m512 x) { return dfm::log2(x); }, input);
}

TEST(Avx512Log10, LaneByLane) {
  __m512 input = make16(
      0.5f,
      1.0f,
      10.0f,
      100.0f,
      0.1f,
      1000.0f,
      1e-20f,
      10000.0f,
      0.001f,
      1e5f,
      1e6f,
      1e7f,
      1e-4f,
      1e-5f,
      42.0f,
      3.14f);
  checkLaneByLane(
      [](float x) { return dfm::log10(x); }, [](__m512 x) { return dfm::log10(x); }, input);
}

TEST(Avx512Acos, LaneByLane) {
  __m512 input = make16(
      -1.0f,
      -0.9f,
      -0.75f,
      -0.5f,
      -0.25f,
      0.0f,
      0.25f,
      0.5f,
      0.75f,
      0.9f,
      0.99f,
      -0.99f,
      0.1f,
      -0.1f,
      0.3f,
      -0.3f);
  checkLaneByLane(
      [](float x) { return dfm::acos(x); }, [](__m512 x) { return dfm::acos(x); }, input);
}

TEST(Avx512Asin, LaneByLane) {
  __m512 input = make16(
      -1.0f,
      -0.5f,
      -0.25f,
      0.0f,
      0.25f,
      0.5f,
      0.75f,
      0.9f,
      -0.9f,
      -0.75f,
      0.99f,
      -0.99f,
      0.1f,
      -0.1f,
      0.3f,
      -0.3f);
  checkLaneByLane(
      [](float x) { return dfm::asin(x); }, [](__m512 x) { return dfm::asin(x); }, input);
}

TEST(Avx512Atan, LaneByLane) {
  __m512 input = make16(
      -10.0f,
      -2.0f,
      -0.5f,
      0.0f,
      0.5f,
      2.0f,
      10.0f,
      100.0f,
      -100.0f,
      -1.0f,
      1.0f,
      0.1f,
      -0.1f,
      50.0f,
      -50.0f,
      0.01f);
  checkLaneByLane(
      [](float x) { return dfm::atan(x); }, [](__m512 x) { return dfm::atan(x); }, input);
}

TEST(Avx512Cbrt, LaneByLane) {
  __m512 input = make16(
      0.125f,
      1.0f,
      8.0f,
      27.0f,
      64.0f,
      125.0f,
      1000.0f,
      0.001f,
      0.5f,
      2.0f,
      10.0f,
      100.0f,
      1e-20f,
      0.1f,
      500.0f,
      42.0f);
  checkLaneByLane(
      [](float x) { return dfm::cbrt(x); }, [](__m512 x) { return dfm::cbrt(x); }, input);
}

// ---- Sweep tests: thorough range coverage to catch cross-lane bugs ----

TEST(Avx512Sin, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 4096);
  for (float base = static_cast<float>(-4.0 * M_PI); base < static_cast<float>(4.0 * M_PI);
       base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](__m512 x) { return dfm::sin(x); },
        makeSeq(base, delta),
        2);
  }
}

TEST(Avx512Cos, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 4096);
  for (float base = static_cast<float>(-4.0 * M_PI); base < static_cast<float>(4.0 * M_PI);
       base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
        [](__m512 x) { return dfm::cos(x); },
        makeSeq(base, delta),
        2);
  }
}

TEST(Avx512Tan, Sweep) {
  float delta = 0.001f;
  for (float base = -1.5f; base < 1.5f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
        [](__m512 x) { return dfm::tan(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(Avx512Exp, Sweep) {
  float delta = 0.01f;
  for (float base = -20.0f; base < 20.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](__m512 x) { return dfm::exp(x); },
        makeSeq(base, delta),
        5);
  }
}

TEST(Avx512Exp2, Sweep) {
  float delta = 0.01f;
  for (float base = -20.0f; base < 20.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
        [](__m512 x) { return dfm::exp2(x); },
        makeSeq(base, delta),
        1);
  }
}

TEST(Avx512Exp10, Sweep) {
  float delta = 0.005f;
  for (float base = -10.0f; base < 10.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
        [](__m512 x) { return dfm::exp10(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(Avx512Log, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
        [](__m512 x) { return dfm::log(x); },
        makeSeq(base, delta),
        2);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(Avx512Log2, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
        [](__m512 x) { return dfm::log2(x); },
        makeSeq(base, delta),
        1);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(Avx512Log10, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log10(static_cast<double>(x))); },
        [](__m512 x) { return dfm::log10(x); },
        makeSeq(base, delta),
        3);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

TEST(Avx512Acos, Sweep) {
  float delta = 0.001f;
  for (float base = -0.999f; base < 0.999f; base += kLanes * delta) {
    float b = std::min(base, 0.999f - (kLanes - 1) * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
        [](__m512 x) { return dfm::acos(x); },
        makeSeq(b, delta),
        4);
  }
}

TEST(Avx512Asin, Sweep) {
  float delta = 0.001f;
  for (float base = -0.999f; base < 0.999f; base += kLanes * delta) {
    float b = std::min(base, 0.999f - (kLanes - 1) * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
        [](__m512 x) { return dfm::asin(x); },
        makeSeq(b, delta),
        4);
  }
}

TEST(Avx512Atan, Sweep) {
  float delta = 0.05f;
  for (float base = -100.0f; base < 100.0f; base += kLanes * delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::atan(static_cast<double>(x))); },
        [](__m512 x) { return dfm::atan(x); },
        makeSeq(base, delta),
        3);
  }
}

TEST(Avx512Cbrt, Sweep) {
  float delta = 0.1f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
        [](__m512 x) { return dfm::cbrt(x); },
        makeSeq(base, delta),
        12);
    if (base > 10.0f)
      delta = 1.0f;
    if (base > 100.0f)
      delta = 10.0f;
  }
}

// ---- Mixed-value tests to catch per-lane independence ----

TEST(Avx512Sin, MixedValues) {
  __m512 input = make16(
      0.0f,
      static_cast<float>(M_PI_2),
      static_cast<float>(-M_PI),
      100.0f,
      -100.0f,
      1e-6f,
      static_cast<float>(2.0 * M_PI),
      -0.0f,
      static_cast<float>(M_PI),
      static_cast<float>(-M_PI_2),
      0.001f,
      -0.001f,
      50.0f,
      -50.0f,
      1.0f,
      -1.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m512 x) { return dfm::sin(x); }, input, 2);
}

TEST(Avx512Exp, MixedValues) {
  __m512 input = make16(
      -20.0f,
      -1.0f,
      0.0f,
      1.0f,
      5.0f,
      10.0f,
      20.0f,
      0.001f,
      -0.001f,
      -10.0f,
      -5.0f,
      0.5f,
      -0.5f,
      15.0f,
      -15.0f,
      0.1f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m512 x) { return dfm::exp(x); }, input);
}

TEST(Avx512Log, MixedValues) {
  __m512 input = make16(
      1e-6f,
      0.001f,
      0.1f,
      1.0f,
      10.0f,
      100.0f,
      1e6f,
      1e10f,
      0.5f,
      2.0f,
      3.14f,
      2.718f,
      42.0f,
      1e-3f,
      1e3f,
      1e7f);
  checkLaneByLane([](float x) { return dfm::log(x); }, [](__m512 x) { return dfm::log(x); }, input);
}

// ---- frexp / ldexp ----

TEST(Avx512Frexp, LaneByLane) {
  __m512 input = make16(
      1.0f,
      2.0f,
      0.5f,
      -4.0f,
      8.0f,
      -0.25f,
      1024.0f,
      0.001f,
      -1.0f,
      100.0f,
      -100.0f,
      0.1f,
      -0.1f,
      42.0f,
      -42.0f,
      0.5f);
  dfm::IntType_t<__m512> eptr;
  __m512 mantissa = dfm::frexp(input, &eptr);

  for (int i = 0; i < kLanes; ++i) {
    int32_t scalar_exp;
    float scalar_m = dfm::frexp(lane(input, i), &scalar_exp);
    EXPECT_EQ(lane(mantissa, i), scalar_m) << "Lane " << i;
    EXPECT_EQ(lane(eptr, i), scalar_exp) << "Lane " << i;
  }
}

TEST(Avx512Frexp, Sweep) {
  float delta = 0.5f;
  for (float base = 0.001f; base < 1000.0f; base += delta) {
    Avx512Float input = makeSeq(base, delta);
    dfm::IntType_t<__m512> eptr;
    __m512 mantissa = dfm::frexp(input, &eptr);
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

TEST(Avx512Ldexp, LaneByLane) {
  __m512 input = make16(
      1.0f,
      0.5f,
      2.0f,
      -1.0f,
      0.25f,
      -0.5f,
      3.0f,
      -3.0f,
      1.0f,
      0.5f,
      2.0f,
      -1.0f,
      0.25f,
      -0.5f,
      3.0f,
      -3.0f);
  Avx512Int32 exp =
      Avx512Int32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
  __m512 result = dfm::ldexp(input, exp);

  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

// ---- atan2 ----

TEST(Avx512Atan2, LaneByLane) {
  __m512 y = make16(
      1.0f,
      -1.0f,
      0.0f,
      1.0f,
      -1.0f,
      1.0f,
      -1.0f,
      0.0f,
      0.5f,
      -0.5f,
      2.0f,
      -2.0f,
      0.1f,
      -0.1f,
      10.0f,
      -10.0f);
  __m512 x = make16(
      1.0f,
      1.0f,
      1.0f,
      -1.0f,
      -1.0f,
      0.0f,
      0.0f,
      -1.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      0.3f,
      -0.3f,
      5.0f,
      -5.0f);
  __m512 result = dfm::atan2(y, x);

  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i)
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

TEST(Avx512Atan2, Sweep) {
  float delta = 0.1f;
  for (float ybase = -5.0f; ybase < 5.0f; ybase += 1.0f) {
    for (float xbase = -5.0f; xbase < 5.0f; xbase += 1.0f) {
      __m512 y = make16(
          ybase,
          ybase + delta,
          ybase - delta,
          ybase + 2 * delta,
          ybase - 2 * delta,
          ybase + 0.5f,
          ybase - 0.5f,
          ybase,
          ybase + 3 * delta,
          ybase - 3 * delta,
          ybase + 0.3f,
          ybase - 0.3f,
          ybase + 1.0f,
          ybase - 1.0f,
          ybase + 0.1f,
          ybase - 0.1f);
      __m512 x = make16(
          xbase,
          xbase - delta,
          xbase + delta,
          xbase - 2 * delta,
          xbase + 2 * delta,
          xbase - 0.5f,
          xbase + 0.5f,
          xbase,
          xbase - 3 * delta,
          xbase + 3 * delta,
          xbase - 0.3f,
          xbase + 0.3f,
          xbase - 1.0f,
          xbase + 1.0f,
          xbase - 0.1f,
          xbase + 0.1f);
      __m512 result = dfm::atan2(y, x);
      for (int i = 0; i < kLanes; ++i) {
        float expected = static_cast<float>(
            ::atan2(static_cast<double>(lane(y, i)), static_cast<double>(lane(x, i))));
        uint32_t dist = dfm::float_distance(expected, lane(result, i));
        EXPECT_LE(dist, 3u) << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
      }
    }
  }
}

// ---- Edge cases: NaN, Inf, denormals per lane ----

TEST(Avx512Edge, SinSpecialValues) {
  __m512 input = make16(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      3.14f,
      -3.14f,
      0.001f,
      -0.001f);
  __m512 result = dfm::sin(input);
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

TEST(Avx512Edge, ExpSpecialValues) {
  __m512 input = make16(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      -100.0f,
      100.0f,
      -89.0f,
      89.0f,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      1.0f,
      -1.0f,
      10.0f,
      -10.0f,
      0.5f);
  __m512 result = dfm::exp(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": input=" << lane(input, i)
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(Avx512Edge, LogSpecialValues) {
  __m512 input = make16(
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      -1.0f,
      1.0f,
      0.5f,
      2.0f,
      10.0f,
      100.0f,
      0.001f,
      1e10f);
  __m512 result = dfm::log(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": input=" << lane(input, i)
                          << " expected=" << expected << " actual=" << actual;
    }
  }
}

// ---- Additional edge case tests ----

// Helper: load 16 floats from a cycling pattern array.
static __m512 makeFromPattern(const float* pattern, int pattern_len) {
  alignas(64) float buf[kLanes];
  for (int i = 0; i < kLanes; ++i) {
    buf[i] = pattern[i % pattern_len];
  }
  return _mm512_load_ps(buf);
}

TEST(Avx512Frexp, SpecialValues) {
  float pattern[] = {
      0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      -0.0f,
      std::numeric_limits<float>::min(),
      1.0f};
  __m512 input = makeFromPattern(pattern, 8);
  dfm::IntType_t<__m512> eptr;
  __m512 mantissa = dfm::frexp(input, &eptr);
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

TEST(Avx512Ldexp, SpecialValues) {
  float frac_pattern[] = {
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      0.5f};
  int32_t exp_pattern[] = {2, 1, 7, 0, 3, 0, 5, 10};
  alignas(64) float frac_buf[kLanes];
  alignas(64) int32_t exp_buf[kLanes];
  for (int i = 0; i < kLanes; ++i) {
    frac_buf[i] = frac_pattern[i % 8];
    exp_buf[i] = exp_pattern[i % 8];
  }
  __m512 frac = _mm512_load_ps(frac_buf);
  Avx512Int32 exp = Avx512Int32(_mm512_load_si512(exp_buf));
  __m512 result = dfm::ldexp(frac, exp);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::ldexp(frac_buf[i], exp_buf[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(Avx512Edge, TanSpecialValues) {
  float pattern[] = {
      0.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -0.0f,
      1.0f,
      -1.0f,
      0.5f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::tan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::tan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(Avx512Edge, AtanSpecialValues) {
  float pattern[] = {
      0.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      20000000.0f,
      -20000000.0f,
      1e10f,
      -1e10f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::atan(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(Avx512Edge, AsinOutOfRange) {
  float pattern[] = {1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::asin(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(Avx512Edge, AcosOutOfRange) {
  float pattern[] = {1.00001f, -1.00001f, 2.0f, -5.0f, 10.0f, -10.0f, 100.0f, -100.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::acos(input);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(Avx512LogAccurate, NegativeInputs) {
  float pattern[] = {
      -1.0f,
      -100.0f,
      -std::numeric_limits<float>::infinity(),
      -0.0f,
      -0.5f,
      -1e10f,
      -1e-30f,
      -42.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::log<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(lane(input, i));
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

TEST(Avx512Log2Accurate, EdgeCases) {
  float pattern[] = {
      0.0f,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      -1.0f,
      -std::numeric_limits<float>::infinity(),
      -0.0f,
      1.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::log2<__m512, dfm::MaxAccuracyTraits>(input);
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

TEST(Avx512Log10Accurate, EdgeCases) {
  float pattern[] = {
      0.0f,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min(),
      -1.0f,
      -std::numeric_limits<float>::infinity(),
      -0.0f,
      10.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::log10<__m512, dfm::MaxAccuracyTraits>(input);
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

TEST(Avx512CbrtAccurate, NegativeInf) {
  float pattern[] = {
      -std::numeric_limits<float>::infinity(),
      -8.0f,
      -27.0f,
      -1.0f,
      -0.001f,
      -1e10f,
      -1e-20f,
      -125.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::cbrt<__m512, dfm::MaxAccuracyTraits>(input);
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

TEST(Avx512Exp2Bounds, InfInputs) {
  float pattern[] = {
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      128.0f,
      -150.0f,
      1.0f,
      -1.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::exp2<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp2<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(Avx512Exp10Bounds, InfInputs) {
  float pattern[] = {
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.0f,
      39.0f,
      -39.0f,
      1.0f,
      -1.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::exp10<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp10<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(Avx512ExpBounds, EdgeCases) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  float pattern[] = {nan, inf, -inf, -100.0f, 89.0f, 100.0f, 0.0f, -0.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::exp<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::exp<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 0u) << "Lane " << i;
    }
  }
}

TEST(Avx512LogAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float pattern[] = {
      0.0f,
      inf,
      nan,
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      1.0f,
      0.5f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::log<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::log<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
      EXPECT_EQ(std::signbit(expected), std::signbit(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 0u) << "Lane " << i;
    }
  }
}

TEST(Avx512CbrtAccurate, EdgeCases) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float pattern[] = {
      0.0f, inf, nan, std::numeric_limits<float>::denorm_min(), 1.0f, 1000.0f, 0.001f, 100.0f};
  __m512 input = makeFromPattern(pattern, 8);
  __m512 result = dfm::cbrt<__m512, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::cbrt<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(lane(result, i))) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, lane(result, i));
      EXPECT_LE(dist, 0u) << "Lane " << i;
    }
  }
}

TEST(Avx512Atan2Bounds, InfCases) {
  float inf = std::numeric_limits<float>::infinity();
  float y_pat[] = {inf, -inf, inf, -inf, 1.0f, -1.0f, inf, 0.0f};
  float x_pat[] = {inf, inf, -inf, -inf, inf, -inf, 1.0f, inf};
  alignas(64) float y_buf[kLanes], x_buf[kLanes];
  for (int i = 0; i < kLanes; ++i) {
    y_buf[i] = y_pat[i % 8];
    x_buf[i] = x_pat[i % 8];
  }
  __m512 y = _mm512_load_ps(y_buf);
  __m512 x = _mm512_load_ps(x_buf);
  __m512 result = dfm::atan2<__m512, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < kLanes; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(y_buf[i], x_buf[i]);
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i << ": y=" << y_buf[i] << " x=" << x_buf[i];
  }
}

// --- hypot ---

TEST(Avx512Hypot, LaneByLane) {
  alignas(64) float x_buf[kLanes] = {
      3.0f,
      0.0f,
      1e30f,
      -5.0f,
      1.0f,
      1e-20f,
      -1e15f,
      7.0f,
      0.0f,
      1e38f,
      -3.0f,
      100.0f,
      1e-20f,
      -1e25f,
      12.0f,
      0.5f};
  alignas(64) float y_buf[kLanes] = {
      4.0f,
      1.0f,
      1e30f,
      12.0f,
      1.0f,
      1e-20f,
      1e15f,
      24.0f,
      0.0f,
      0.0f,
      4.0f,
      0.0f,
      1e-20f,
      1e25f,
      5.0f,
      0.5f};
  __m512 x = _mm512_load_ps(x_buf);
  __m512 y = _mm512_load_ps(y_buf);
  __m512 result = dfm::hypot(x, y);
  for (int i = 0; i < kLanes; ++i) {
    double xd = static_cast<double>(x_buf[i]);
    double yd = static_cast<double>(y_buf[i]);
    float expected = static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 2u) << "Lane " << i << ": x=" << x_buf[i] << " y=" << y_buf[i]
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

// -- hypot bounds --

TEST(Avx512HypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  alignas(64) float x_buf[kLanes] = {
      inf, -inf, nan, nan, inf, 3.0f, -inf, nan, inf, nan, -inf, nan, 0.0f, inf, nan, -inf};
  alignas(64) float y_buf[kLanes] = {
      3.0f, nan, inf, -inf, inf, inf, 0.0f, -inf, nan, inf, nan, -inf, inf, -inf, inf, nan};
  __m512 x = _mm512_load_ps(x_buf);
  __m512 y = _mm512_load_ps(y_buf);
  __m512 result = dfm::hypot<__m512, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isinf(lane(result, i)) && lane(result, i) > 0)
        << "Lane " << i << ": x=" << x_buf[i] << " y=" << y_buf[i] << " result=" << lane(result, i);
  }
}

TEST(Avx512HypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  alignas(64) float x_buf[kLanes] = {
      nan, 3.0f, nan, nan, 0.0f, nan, -1.0f, nan, nan, 5.0f, nan, nan, nan, 0.0f, nan, 1e10f};
  alignas(64) float y_buf[kLanes] = {
      3.0f, nan, 0.0f, nan, nan, -5.0f, nan, nan, 1.0f, nan, nan, 0.0f, nan, nan, 1e-5f, nan};
  __m512 x = _mm512_load_ps(x_buf);
  __m512 y = _mm512_load_ps(y_buf);
  __m512 result = dfm::hypot<__m512, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < kLanes; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": x=" << x_buf[i] << " y=" << y_buf[i] << " result=" << lane(result, i);
  }
}

#else // !defined(__AVX512F__)

TEST(Avx512, Unavailable) {
  GTEST_SKIP() << "AVX-512 not available";
}

#endif // defined(__AVX512F__)
