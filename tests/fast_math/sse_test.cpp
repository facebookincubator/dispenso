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

// Helper: compare lane-by-lane against scalar function using raw __m128 types.
// Each lane of `input` is run through the scalar function and compared with the SSE result.
template <typename ScalarFunc, typename SseFunc>
static void
checkLaneByLane(ScalarFunc scalar_fn, SseFunc sse_fn, __m128 input, uint32_t max_ulps = 0) {
  __m128 result = sse_fn(input);
  for (int i = 0; i < 4; ++i) {
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

// ---- Transcendental functions: lane-by-lane correctness ----
// All tests use raw __m128 types with dfm::func(x) — no explicit template args.
// This exercises the SimdTypeFor forwarding mechanism.

TEST(SseSin, LaneByLane) {
  __m128 input = make4(0.0f, 0.5f, 1.0f, -0.7f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m128 x) { return dfm::sin(x); }, input, 2);
}

TEST(SseSin, LargeRange) {
  __m128 input = make4(-3.14f, 3.14f, 6.28f, -6.28f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m128 x) { return dfm::sin(x); }, input, 2);
}

TEST(SseCos, LaneByLane) {
  __m128 input = make4(0.0f, 0.5f, 1.0f, -0.7f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m128 x) { return dfm::cos(x); }, input, 2);
}

TEST(SseCos, LargeRange) {
  __m128 input = make4(-3.14f, 3.14f, 6.28f, -6.28f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m128 x) { return dfm::cos(x); }, input, 2);
}

TEST(SseTan, LaneByLane) {
  __m128 input = make4(0.0f, 0.3f, -0.5f, 1.0f);
  checkLaneByLane([](float x) { return dfm::tan(x); }, [](__m128 x) { return dfm::tan(x); }, input);
}

TEST(SseExp, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 0.5f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m128 x) { return dfm::exp(x); }, input);
}

TEST(SseExp2, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 3.5f);
  checkLaneByLane(
      [](float x) { return dfm::exp2(x); }, [](__m128 x) { return dfm::exp2(x); }, input);
}

TEST(SseExp10, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 0.5f);
  checkLaneByLane(
      [](float x) { return dfm::exp10(x); }, [](__m128 x) { return dfm::exp10(x); }, input);
}

TEST(SseLog2, LaneByLane) {
  __m128 input = make4(0.5f, 1.0f, 2.0f, 10.0f);
  checkLaneByLane(
      [](float x) { return dfm::log2(x); }, [](__m128 x) { return dfm::log2(x); }, input);
}

TEST(SseLog, LaneByLane) {
  __m128 input = make4(0.5f, 1.0f, 2.0f, 10.0f);
  checkLaneByLane([](float x) { return dfm::log(x); }, [](__m128 x) { return dfm::log(x); }, input);
}

TEST(SseLog10, LaneByLane) {
  __m128 input = make4(0.5f, 1.0f, 10.0f, 100.0f);
  checkLaneByLane(
      [](float x) { return dfm::log10(x); }, [](__m128 x) { return dfm::log10(x); }, input);
}

TEST(SseAcos, LaneByLane) {
  __m128 input = make4(-0.5f, 0.0f, 0.5f, 0.9f);
  checkLaneByLane(
      [](float x) { return dfm::acos(x); }, [](__m128 x) { return dfm::acos(x); }, input);
}

TEST(SseAsin, LaneByLane) {
  __m128 input = make4(-0.5f, 0.0f, 0.5f, 0.9f);
  checkLaneByLane(
      [](float x) { return dfm::asin(x); }, [](__m128 x) { return dfm::asin(x); }, input);
}

TEST(SseAtan, LaneByLane) {
  __m128 input = make4(-2.0f, -0.5f, 0.5f, 2.0f);
  checkLaneByLane(
      [](float x) { return dfm::atan(x); }, [](__m128 x) { return dfm::atan(x); }, input);
}

TEST(SseCbrt, LaneByLane) {
  __m128 input = make4(0.125f, 1.0f, 8.0f, 27.0f);
  checkLaneByLane(
      [](float x) { return dfm::cbrt(x); }, [](__m128 x) { return dfm::cbrt(x); }, input);
}

// ---- Sweep tests: many values to catch cross-lane bugs ----

TEST(SseSin, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 1024);
  for (float base = static_cast<float>(-M_PI); base < static_cast<float>(M_PI); base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](__m128 x) { return dfm::sin(x); },
        input,
        2);
  }
}

TEST(SseCos, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 1024);
  for (float base = static_cast<float>(-M_PI); base < static_cast<float>(M_PI); base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
        [](__m128 x) { return dfm::cos(x); },
        input,
        2);
  }
}

TEST(SseExp, Sweep) {
  float delta = 0.01f;
  for (float base = -10.0f; base < 10.0f; base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](__m128 x) { return dfm::exp(x); },
        input,
        5);
  }
}

TEST(SseLog, Sweep) {
  float delta = 0.1f;
  for (float base = 0.01f; base < 100.0f; base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
        [](__m128 x) { return dfm::log(x); },
        input,
        2);
  }
}

// ---- Mixed-value tests to catch per-lane independence ----

TEST(SseSin, MixedValues) {
  // Deliberately different magnitudes and signs per lane.
  __m128 input = make4(0.0f, static_cast<float>(M_PI_2), static_cast<float>(-M_PI), 100.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m128 x) { return dfm::sin(x); }, input, 2);
}

TEST(SseExp, MixedValues) {
  __m128 input = make4(-10.0f, 0.0f, 1.0f, 10.0f);
  checkLaneByLane([](float x) { return dfm::exp(x); }, [](__m128 x) { return dfm::exp(x); }, input);
}

// ---- frexp / ldexp (two-argument functions) ----

TEST(SseFrexp, LaneByLane) {
  __m128 input = make4(1.0f, 2.0f, 0.5f, -4.0f);
  dfm::IntType_t<__m128> eptr;
  __m128 mantissa = dfm::frexp(input, &eptr);

  for (int i = 0; i < 4; ++i) {
    int32_t scalar_exp;
    float scalar_m = dfm::frexp(lane(input, i), &scalar_exp);
    EXPECT_EQ(lane(mantissa, i), scalar_m) << "Lane " << i;
    EXPECT_EQ(lane(eptr, i), scalar_exp) << "Lane " << i;
  }
}

TEST(SseFrexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  dfm::IntType_t<__m128> eptr;
  __m128 mantissa = dfm::frexp(input, &eptr);

  for (int i = 0; i < 4; ++i) {
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

TEST(SseLdexp, LaneByLane) {
  __m128 input = make4(1.0f, 0.5f, 2.0f, -1.0f);
  SseInt32 exp = SseInt32(_mm_set_epi32(3, 2, 1, 0));
  __m128 result = dfm::ldexp(input, exp);

  for (int i = 0; i < 4; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(SseLdexp, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(kInf, -kInf, kNaN, 0.0f);
  SseInt32 exp = SseInt32(_mm_set_epi32(0, 7, 1, 2));
  __m128 result = dfm::ldexp(input, exp);

  for (int i = 0; i < 4; ++i) {
    float expected = dfm::ldexp(lane(input, i), lane(exp, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// ---- atan2 ----

TEST(SseAtan2, LaneByLane) {
  __m128 y = make4(1.0f, -1.0f, 0.0f, 1.0f);
  __m128 x = make4(1.0f, 1.0f, 1.0f, -1.0f);
  __m128 result = dfm::atan2(y, x);

  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected
                        << " actual=" << lane(result, i);
  }
}

// ---- Accuracy and bounds trait variants ----
// Non-default traits: use dfm::func<__m128, Traits>(x) — user specifies raw type + traits.

struct BoundsTraits {
  static constexpr bool kMaxAccuracy = false;
  static constexpr bool kBoundsValues = true;
};

// -- sin/cos accurate --

TEST(SseSinAccurate, LaneByLane) {
  __m128 input = make4(0.0f, 0.5f, -1.0f, 2.5f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); },
      [](__m128 x) { return dfm::sin<__m128, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

TEST(SseSinAccurate, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 1024);
  for (float base = static_cast<float>(-M_PI); base < static_cast<float>(M_PI); base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::sin(static_cast<double>(x))); },
        [](__m128 x) { return dfm::sin<__m128, dfm::MaxAccuracyTraits>(x); },
        input,
        2);
  }
}

TEST(SseCosAccurate, LaneByLane) {
  __m128 input = make4(0.0f, 0.5f, -1.0f, 2.5f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); },
      [](__m128 x) { return dfm::cos<__m128, dfm::MaxAccuracyTraits>(x); },
      input,
      2);
}

TEST(SseCosAccurate, Sweep) {
  float delta = static_cast<float>(2.0 * M_PI / 1024);
  for (float base = static_cast<float>(-M_PI); base < static_cast<float>(M_PI); base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cos(static_cast<double>(x))); },
        [](__m128 x) { return dfm::cos<__m128, dfm::MaxAccuracyTraits>(x); },
        input,
        2);
  }
}

// -- exp accurate and bounds --

TEST(SseExpAccurate, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 5.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp<float, dfm::MaxAccuracyTraits>(x); },
      [](__m128 x) { return dfm::exp<__m128, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(SseExpAccurate, Sweep) {
  float delta = 0.01f;
  for (float base = -10.0f; base < 10.0f; base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp(static_cast<double>(x))); },
        [](__m128 x) { return dfm::exp<__m128, dfm::MaxAccuracyTraits>(x); },
        input,
        5);
  }
}

TEST(SseExpBounds, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 5.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp<float, BoundsTraits>(x); },
      [](__m128 x) { return dfm::exp<__m128, BoundsTraits>(x); },
      input);
}

TEST(SseExpBounds, EdgeCases) {
  // NaN, +Inf, -Inf, large positive (overflow), large negative (underflow)
  __m128 input = make4(
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -100.0f);
  __m128 result = dfm::exp<__m128, BoundsTraits>(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::exp<float, BoundsTraits>(lane(input, i));
    float actual = lane(result, i);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "Lane " << i;
    } else if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i << ": expected inf, got " << actual;
      EXPECT_EQ(std::signbit(expected), std::signbit(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

TEST(SseExpBounds, Overflow) {
  __m128 input = make4(89.0f, 100.0f, -100.0f, 0.0f);
  __m128 result = dfm::exp<__m128, BoundsTraits>(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::exp<float, BoundsTraits>(lane(input, i));
    float actual = lane(result, i);
    if (std::isinf(expected)) {
      EXPECT_TRUE(std::isinf(actual)) << "Lane " << i;
    } else {
      uint32_t dist = dfm::float_distance(expected, actual);
      EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
    }
  }
}

// -- exp2 bounds --

TEST(SseExp2Bounds, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 10.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp2<float, BoundsTraits>(x); },
      [](__m128 x) { return dfm::exp2<__m128, BoundsTraits>(x); },
      input);
}

TEST(SseExp2Bounds, EdgeCases) {
  __m128 input = make4(
      std::numeric_limits<float>::quiet_NaN(),
      128.0f, // overflow
      -150.0f, // underflow
      0.0f);
  __m128 result = dfm::exp2<__m128, BoundsTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseExp10Bounds, LaneByLane) {
  __m128 input = make4(0.0f, 1.0f, -1.0f, 2.0f);
  checkLaneByLane(
      [](float x) { return dfm::exp10<float, BoundsTraits>(x); },
      [](__m128 x) { return dfm::exp10<__m128, BoundsTraits>(x); },
      input);
}

TEST(SseExp10Bounds, EdgeCases) {
  __m128 input = make4(
      std::numeric_limits<float>::quiet_NaN(),
      39.0f, // overflow
      -39.0f, // underflow
      0.0f);
  __m128 result = dfm::exp10<__m128, BoundsTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseLogAccurate, LaneByLane) {
  __m128 input = make4(0.5f, 1.0f, 2.0f, 100.0f);
  checkLaneByLane(
      [](float x) { return dfm::log<float, dfm::MaxAccuracyTraits>(x); },
      [](__m128 x) { return dfm::log<__m128, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(SseLogAccurate, Sweep) {
  float delta = 0.1f;
  for (float base = 0.01f; base < 100.0f; base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log(static_cast<double>(x))); },
        [](__m128 x) { return dfm::log<__m128, dfm::MaxAccuracyTraits>(x); },
        input,
        2);
    if (base > 10.0f)
      delta = 1.0f;
  }
}

TEST(SseLogAccurate, EdgeCases) {
  __m128 input = make4(
      0.0f, // -inf
      std::numeric_limits<float>::infinity(), // +inf
      std::numeric_limits<float>::quiet_NaN(), // NaN
      std::numeric_limits<float>::denorm_min()); // denorm
  __m128 result = dfm::log<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseLogAccurate, NegativeInputs) {
  // log(negative) and log(-inf) must return NaN, matching scalar behavior.
  __m128 input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  __m128 result = dfm::log<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

// -- log2 accurate --

TEST(SseLog2Accurate, LaneByLane) {
  __m128 input = make4(0.5f, 1.0f, 2.0f, 1024.0f);
  checkLaneByLane(
      [](float x) { return dfm::log2<float, dfm::MaxAccuracyTraits>(x); },
      [](__m128 x) { return dfm::log2<__m128, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(SseLog2Accurate, NegativeInputs) {
  __m128 input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  __m128 result = dfm::log2<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseLog10Accurate, NegativeInputs) {
  __m128 input = make4(-1.0f, -100.0f, -std::numeric_limits<float>::infinity(), -0.0f);
  __m128 result = dfm::log10<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseCbrtAccurate, LaneByLane) {
  __m128 input = make4(0.125f, 1.0f, 8.0f, 27.0f);
  checkLaneByLane(
      [](float x) { return dfm::cbrt<float, dfm::MaxAccuracyTraits>(x); },
      [](__m128 x) { return dfm::cbrt<__m128, dfm::MaxAccuracyTraits>(x); },
      input);
}

TEST(SseCbrtAccurate, EdgeCases) {
  __m128 input = make4(
      0.0f,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::denorm_min());
  __m128 result = dfm::cbrt<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseAtan2Bounds, LaneByLane) {
  __m128 y = make4(1.0f, -1.0f, 0.0f, 1.0f);
  __m128 x = make4(1.0f, 1.0f, 1.0f, -1.0f);
  __m128 result = dfm::atan2<__m128, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 0u) << "Lane " << i;
  }
}

TEST(SseAtan2Bounds, InfCases) {
  float inf = std::numeric_limits<float>::infinity();
  // Both inf: should give ±π/4 or ±3π/4
  __m128 y = make4(inf, -inf, inf, -inf);
  __m128 x = make4(inf, inf, -inf, -inf);
  __m128 result = dfm::atan2<__m128, dfm::MaxAccuracyTraits>(y, x);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan2<float, dfm::MaxAccuracyTraits>(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    uint32_t dist = dfm::float_distance(expected, actual);
    EXPECT_LE(dist, 0u) << "Lane " << i << ": expected=" << expected << " actual=" << actual;
  }
}

// ---- Mixed special values: different edge cases in different lanes ----
// These tests catch bugs in per-lane conditional/masking logic.

TEST(SseSin, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  // NaN, Inf, -Inf, and normal in different lanes.
  __m128 input = make4(kNaN, kInf, -kInf, 1.0f);
  __m128 result = dfm::sin(input);
  EXPECT_TRUE(std::isnan(lane(result, 0))) << "sin(NaN) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 1))) << "sin(+Inf) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "sin(-Inf) should be NaN";
  EXPECT_NEAR(lane(result, 3), dfm::sin(1.0f), 1e-7f) << "sin(1) lane contaminated";
}

TEST(SseCos, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  __m128 input = make4(0.0f, kNaN, kInf, -kInf);
  __m128 result = dfm::cos(input);
  EXPECT_EQ(lane(result, 0), 1.0f) << "cos(0) should be 1";
  EXPECT_TRUE(std::isnan(lane(result, 1))) << "cos(NaN) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "cos(+Inf) should be NaN";
  EXPECT_TRUE(std::isnan(lane(result, 3))) << "cos(-Inf) should be NaN";
}

TEST(SseExp, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  // Bounds-checked variant: NaN, +Inf (→+Inf), -Inf (→0), overflow (→+Inf)
  __m128 input = make4(kNaN, kInf, -kInf, 89.0f);
  __m128 result = dfm::exp<__m128, BoundsTraits>(input);
  EXPECT_TRUE(std::isnan(lane(result, 0))) << "exp(NaN)";
  EXPECT_TRUE(std::isinf(lane(result, 1)) && lane(result, 1) > 0) << "exp(+Inf)";
  EXPECT_EQ(lane(result, 2), 0.0f) << "exp(-Inf)";
  EXPECT_TRUE(std::isinf(lane(result, 3)) || lane(result, 3) > 1e38f) << "exp(89)";

  // Underflow: very negative input should be near zero
  __m128 input2 = make4(-100.0f, 0.0f, -0.0f, 1.0f);
  __m128 result2 = dfm::exp<__m128, BoundsTraits>(input2);
  EXPECT_LT(lane(result2, 0), 1e-38f) << "exp(-100) should underflow";
  EXPECT_EQ(lane(result2, 1), 1.0f) << "exp(0) should be 1";
  EXPECT_EQ(lane(result2, 2), 1.0f) << "exp(-0) should be 1";
  float expected_e = dfm::exp<float, BoundsTraits>(1.0f);
  uint32_t dist = dfm::float_distance(expected_e, lane(result2, 3));
  EXPECT_LE(dist, 0u) << "exp(1)";
}

TEST(SseLog, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  // Note: log(negative) returns NaN for scalar but is not guaranteed for SIMD —
  // the bounds-check logic doesn't explicitly detect negative inputs.
  // Test only the guaranteed edge cases: 0 → -Inf, +Inf → +Inf, NaN → NaN, denorm.
  // Lane 0: 0.0f, Lane 1: +Inf, Lane 2: NaN, Lane 3: denorm_min
  __m128 input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  __m128 result = dfm::log<__m128, dfm::MaxAccuracyTraits>(input);
  EXPECT_TRUE(std::isinf(lane(result, 0)) && lane(result, 0) < 0) << "log(0) should be -Inf";
  EXPECT_TRUE(std::isinf(lane(result, 1)) && lane(result, 1) > 0) << "log(+Inf) should be +Inf";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "log(NaN) should be NaN";
  // denorm_min: compare against scalar
  float expected_denorm =
      dfm::log<float, dfm::MaxAccuracyTraits>(std::numeric_limits<float>::denorm_min());
  uint32_t dist = dfm::float_distance(expected_denorm, lane(result, 3));
  EXPECT_LE(dist, 0u) << "log(denorm_min)";
}

TEST(SseCbrt, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  __m128 input = make4(0.0f, kInf, kNaN, 8.0f);
  __m128 result = dfm::cbrt<__m128, dfm::MaxAccuracyTraits>(input);
  float expected_0 = dfm::cbrt<float, dfm::MaxAccuracyTraits>(0.0f);
  EXPECT_EQ(lane(result, 0), expected_0) << "cbrt(0)";
  EXPECT_TRUE(std::isinf(lane(result, 1))) << "cbrt(+Inf) should be +Inf";
  EXPECT_TRUE(std::isnan(lane(result, 2))) << "cbrt(NaN) should be NaN";
  float expected_8 = dfm::cbrt<float, dfm::MaxAccuracyTraits>(8.0f);
  uint32_t dist = dfm::float_distance(expected_8, lane(result, 3));
  EXPECT_LE(dist, 0u) << "cbrt(8)";
}

// ---- Domain boundary tests ----

TEST(SseAsin, DomainBoundaries) {
  // Exact domain boundaries: -1, 1, and slightly inside.
  __m128 input = make4(-1.0f, 1.0f, 0.0f, 0.99999f);
  checkLaneByLane(
      [](float x) { return dfm::asin(x); }, [](__m128 x) { return dfm::asin(x); }, input);
}

TEST(SseAcos, DomainBoundaries) {
  __m128 input = make4(-1.0f, 1.0f, 0.0f, -0.5f);
  checkLaneByLane(
      [](float x) { return dfm::acos(x); }, [](__m128 x) { return dfm::acos(x); }, input);
}

TEST(SseAtan2, NegativeZero) {
  // -0 sign preservation is critical for atan2.
  __m128 y = make4(0.0f, -0.0f, 0.0f, -0.0f);
  __m128 x = make4(-1.0f, -1.0f, 1.0f, 1.0f);
  __m128 result = dfm::atan2(y, x);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    // Use bit comparison for -0 / +0 distinction.
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i) << " expected=" << expected
        << " actual=" << actual;
  }
}

TEST(SseAtan2, ZeroZero) {
  // All four zero combinations.
  __m128 y = make4(0.0f, -0.0f, 0.0f, -0.0f);
  __m128 x = make4(0.0f, 0.0f, -0.0f, -0.0f);
  __m128 result = dfm::atan2(y, x);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan2(lane(y, i), lane(x, i));
    float actual = lane(result, i);
    EXPECT_EQ(dfm::bit_cast<uint32_t>(expected), dfm::bit_cast<uint32_t>(actual))
        << "Lane " << i << ": y=" << lane(y, i) << " x=" << lane(x, i);
  }
}

TEST(SseAtan2, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  __m128 y = make4(kNaN, 1.0f, kInf, -kInf);
  __m128 x = make4(1.0f, kNaN, kInf, -kInf);
  __m128 result = dfm::atan2(y, x);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseTan, MixedSpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m128 input = make4(0.0f, kNaN, kInf, -kInf);
  __m128 result = dfm::tan(input);
  EXPECT_EQ(lane(result, 0), 0.0f);
  EXPECT_TRUE(std::isnan(lane(result, 1)));
  EXPECT_TRUE(std::isnan(lane(result, 2)));
  EXPECT_TRUE(std::isnan(lane(result, 3)));
}

TEST(SseAtan, SpecialValues) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m128 input = make4(0.0f, kNaN, kInf, -kInf);
  __m128 result = dfm::atan(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(SseAtan, LargeMagnitude) {
  __m128 input = make4(20000000.0f, -20000000.0f, 1e10f, -1e10f);
  __m128 result = dfm::atan(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::atan(lane(input, i));
    EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
  }
}

TEST(SseAsin, OutOfRange) {
  __m128 input = make4(1.00001f, -1.00001f, 2.0f, -5.0f);
  __m128 result = dfm::asin(input);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(SseAcos, OutOfRange) {
  __m128 input = make4(1.00001f, -1.00001f, 2.0f, -5.0f);
  __m128 result = dfm::acos(input);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i << " input=" << lane(input, i);
  }
}

TEST(SseLog2Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  __m128 result = dfm::log2<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseLog10Accurate, EdgeCases) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(0.0f, kInf, kNaN, std::numeric_limits<float>::denorm_min());
  __m128 result = dfm::log10<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseCbrtAccurate, NegativeInf) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  __m128 input = make4(-kInf, -8.0f, -27.0f, -1.0f);
  __m128 result = dfm::cbrt<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
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

TEST(SseExp2Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(kInf, -kInf, kNaN, 0.0f);
  __m128 result = dfm::exp2<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::exp2<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

TEST(SseExp10Bounds, InfInputs) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  __m128 input = make4(kInf, -kInf, kNaN, 0.0f);
  __m128 result = dfm::exp10<__m128, dfm::MaxAccuracyTraits>(input);
  for (int i = 0; i < 4; ++i) {
    float expected = dfm::exp10<float, dfm::MaxAccuracyTraits>(lane(input, i));
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(lane(result, i))) << "Lane " << i;
    } else {
      EXPECT_EQ(lane(result, i), expected) << "Lane " << i;
    }
  }
}

// ---- Broader sweep tests for functions with thin SIMD coverage ----

TEST(SseAtan, Sweep) {
  for (float base = -10.0f; base < 10.0f; base += 0.05f) {
    __m128 input = make4(base, base + 0.01f, base + 0.02f, base + 0.03f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::atan(static_cast<double>(x))); },
        [](__m128 x) { return dfm::atan(x); },
        input,
        3);
  }
}

TEST(SseAsin, Sweep) {
  for (float base = -0.99f; base < 0.99f; base += 0.01f) {
    float b1 = std::min(base + 0.005f, 0.999f);
    float b2 = std::min(base + 0.010f, 0.999f);
    float b3 = std::min(base + 0.015f, 0.999f);
    __m128 input = make4(base, b1, b2, b3);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::asin(static_cast<double>(x))); },
        [](__m128 x) { return dfm::asin(x); },
        input,
        4);
  }
}

TEST(SseAcos, Sweep) {
  for (float base = -0.99f; base < 0.99f; base += 0.01f) {
    float b1 = std::min(base + 0.005f, 0.999f);
    float b2 = std::min(base + 0.010f, 0.999f);
    float b3 = std::min(base + 0.015f, 0.999f);
    __m128 input = make4(base, b1, b2, b3);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::acos(static_cast<double>(x))); },
        [](__m128 x) { return dfm::acos(x); },
        input,
        4);
  }
}

TEST(SseCbrt, Sweep) {
  for (float base = 0.01f; base < 100.0f; base += 0.5f) {
    __m128 input = make4(base, base + 0.1f, base + 0.2f, base + 0.3f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::cbrt(static_cast<double>(x))); },
        [](__m128 x) { return dfm::cbrt(x); },
        input,
        12);
  }
}

TEST(SseExp2, Sweep) {
  for (float base = -10.0f; base < 10.0f; base += 0.05f) {
    __m128 input = make4(base, base + 0.01f, base + 0.02f, base + 0.03f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::exp2(static_cast<double>(x))); },
        [](__m128 x) { return dfm::exp2(x); },
        input,
        1);
  }
}

TEST(SseExp10, Sweep) {
  for (float base = -5.0f; base < 5.0f; base += 0.025f) {
    __m128 input = make4(base, base + 0.005f, base + 0.01f, base + 0.015f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::pow(10.0, static_cast<double>(x))); },
        [](__m128 x) { return dfm::exp10(x); },
        input,
        3);
  }
}

TEST(SseLog2, Sweep) {
  for (float base = 0.01f; base < 100.0f; base += 0.5f) {
    __m128 input = make4(base, base + 0.1f, base + 0.2f, base + 0.3f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log2(static_cast<double>(x))); },
        [](__m128 x) { return dfm::log2(x); },
        input,
        1);
  }
}

TEST(SseLog10, Sweep) {
  for (float base = 0.01f; base < 100.0f; base += 0.5f) {
    __m128 input = make4(base, base + 0.1f, base + 0.2f, base + 0.3f);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::log10(static_cast<double>(x))); },
        [](__m128 x) { return dfm::log10(x); },
        input,
        3);
  }
}

TEST(SseTan, Sweep) {
  // Avoid near-pi/2 where tan diverges.
  float delta = static_cast<float>(M_PI / 512);
  for (float base = static_cast<float>(-M_PI / 2 + 0.1); base < static_cast<float>(M_PI / 2 - 0.1);
       base += delta) {
    __m128 input = make4(base, base + delta, base + 2 * delta, base + 3 * delta);
    checkLaneByLane(
        [](float x) { return static_cast<float>(::tan(static_cast<double>(x))); },
        [](__m128 x) { return dfm::tan(x); },
        input,
        3);
  }
}

TEST(SseAtan2, Sweep) {
  // Sweep all four quadrants.
  for (float y = -5.0f; y <= 5.0f; y += 0.5f) {
    for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
      if (x == 0.0f && y == 0.0f)
        continue;
      __m128 yv = make4(y, y + 0.1f, y, -y);
      __m128 xv = make4(x, x, x + 0.1f, -x);
      __m128 result = dfm::atan2(yv, xv);
      for (int i = 0; i < 4; ++i) {
        float expected = static_cast<float>(
            ::atan2(static_cast<double>(lane(yv, i)), static_cast<double>(lane(xv, i))));
        float actual = lane(result, i);
        uint32_t dist = dfm::float_distance(expected, actual);
        EXPECT_LE(dist, 3u) << "y=" << lane(yv, i) << " x=" << lane(xv, i)
                            << " expected=" << expected << " actual=" << actual;
      }
    }
  }
}

// ---- Large magnitude trig tests ----

TEST(SseSin, LargeMagnitude) {
  // Test range reduction with large inputs.
  __m128 input = make4(100.0f, -100.0f, 1000.0f, -1000.0f);
  checkLaneByLane(
      [](float x) { return ::sinf(x); }, [](__m128 x) { return dfm::sin(x); }, input, 2);
}

TEST(SseCos, LargeMagnitude) {
  __m128 input = make4(100.0f, -100.0f, 1000.0f, -1000.0f);
  checkLaneByLane(
      [](float x) { return ::cosf(x); }, [](__m128 x) { return dfm::cos(x); }, input, 2);
}

TEST(SseTan, LargeMagnitude) {
  // Avoid exact multiples of pi/2.
  __m128 input = make4(50.0f, -50.0f, 200.0f, -200.0f);
  checkLaneByLane([](float x) { return dfm::tan(x); }, [](__m128 x) { return dfm::tan(x); }, input);
}

// --- hypot ---

TEST(SseHypot, LaneByLane) {
  __m128 x = make4(3.0f, 0.0f, 1e30f, -5.0f);
  __m128 y = make4(4.0f, 1.0f, 1e30f, 12.0f);
  __m128 result = dfm::hypot(x, y);
  for (int i = 0; i < 4; ++i) {
    double xd = static_cast<double>(lane(x, i));
    double yd = static_cast<double>(lane(y, i));
    float expected = static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
    uint32_t dist = dfm::float_distance(expected, lane(result, i));
    EXPECT_LE(dist, 2u) << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
                        << " expected=" << expected << " actual=" << lane(result, i);
  }
}

// -- hypot bounds --

TEST(SseHypotBounds, InfNaN) {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  // IEEE 754: hypot(inf, anything) = +inf, including hypot(inf, NaN).
  __m128 x = make4(inf, -inf, nan, nan);
  __m128 y = make4(3.0f, nan, inf, -inf);
  __m128 result = dfm::hypot<__m128, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isinf(lane(result, i)) && lane(result, i) > 0)
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
        << " result=" << lane(result, i);
  }
}

TEST(SseHypotBounds, NaNFinite) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  // hypot(NaN, finite) = NaN when neither is inf.
  __m128 x = make4(nan, 3.0f, nan, nan);
  __m128 y = make4(3.0f, nan, 0.0f, nan);
  __m128 result = dfm::hypot<__m128, dfm::MaxAccuracyTraits>(x, y);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isnan(lane(result, i)))
        << "Lane " << i << ": x=" << lane(x, i) << " y=" << lane(y, i)
        << " result=" << lane(result, i);
  }
}

#else // !defined(__SSE4_1__)

// Dummy test so the binary has at least one test on non-SSE platforms.
TEST(SseFloat, NotAvailable) {
  GTEST_SKIP() << "SSE4.1 not available";
}

#endif // defined(__SSE4_1__)
