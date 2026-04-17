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

#else // !defined(__AVX2__)

// Dummy test so the binary has at least one test on non-AVX2 platforms.
TEST(AvxFloat, NotAvailable) {
  GTEST_SKIP() << "AVX2 not available";
}

#endif // defined(__AVX2__)
