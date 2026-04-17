/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "simd_test_utils.h"

#include <gtest/gtest.h>

namespace dfm = dispenso::fast_math;
using namespace dispenso::fast_math::testing;

TEST(Ldexp, SpecialVals) {
  auto res = dispenso::fast_math::ldexp(-std::numeric_limits<float>::infinity(), 1);
  EXPECT_EQ(res, -std::numeric_limits<float>::infinity());
  res = dispenso::fast_math::ldexp(std::numeric_limits<float>::infinity(), 2);
  EXPECT_EQ(res, std::numeric_limits<float>::infinity());

  res = dispenso::fast_math::ldexp(std::numeric_limits<float>::quiet_NaN(), 7);
  EXPECT_NE(res, res);

  int exp;
  float tval = -0.49999997f;

  float a = dispenso::fast_math::frexp(tval, &exp);
  res = dispenso::fast_math::ldexp(a, exp);
  EXPECT_EQ(res, tval);
}

TEST(Ldexp, Range) {
  auto test = [](float f) {
    int exp;
    float a = dispenso::fast_math::frexp(f, &exp);
    float res = dispenso::fast_math::ldexp(a, exp);
    EXPECT_EQ(res, f);
  };

  dispenso::fast_math::evalForEach(
      -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), test);
}

// Unified SIMD ldexp test — round-trip frexp → ldexp, verified per lane.
static const float kLdexpInputs[] = {
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    2.0f,
    -2.0f,
    100.0f,
    -100.0f,
    0.001f,
    1e-20f,
    1e20f,
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN()};

template <typename Flt>
void checkLdexpSimd() {
  using Traits = SimdTestTraits<Flt>;
  using IntT = dfm::IntType_t<Flt>;
  const int32_t N = Traits::laneCount();
  constexpr int32_t numInputs =
      static_cast<int32_t>(sizeof(kLdexpInputs) / sizeof(kLdexpInputs[0]));

  alignas(64) float buf[kMaxSimdLanes];
  alignas(64) float outResult[kMaxSimdLanes];

  for (int32_t base = 0; base < numInputs; base += N) {
    int32_t count = std::min(N, numInputs - base);
    for (int32_t i = 0; i < count; ++i) {
      buf[i] = kLdexpInputs[base + i];
    }
    for (int32_t i = count; i < N; ++i) {
      buf[i] = buf[count - 1];
    }

    // Round-trip: frexp → ldexp should reproduce the input.
    IntT exponent;
    Flt mantissa = dfm::frexp(Traits::load(buf), &exponent);
    Flt result = dfm::ldexp(mantissa, exponent);
    Traits::store(result, outResult);

    for (int32_t i = 0; i < count; ++i) {
      if (std::isnan(buf[i])) {
        EXPECT_TRUE(std::isnan(outResult[i])) << "ldexp(frexp(NaN)) should be NaN";
      } else if (std::isinf(buf[i])) {
        EXPECT_EQ(outResult[i], buf[i]) << "ldexp(frexp(inf)) should be inf";
      } else {
        EXPECT_EQ(outResult[i], buf[i])
            << "ldexp(frexp(" << buf[i] << ")) round-trip mismatch: got " << outResult[i];
      }
    }
  }
}

#if defined(__SSE4_1__)
TEST(LdexpSse, RoundTrip) {
  checkLdexpSimd<__m128>();
}
#endif
#if defined(__AVX2__)
TEST(LdexpAvx, RoundTrip) {
  checkLdexpSimd<__m256>();
}
#endif
#if defined(__AVX512F__)
TEST(LdexpAvx512, RoundTrip) {
  checkLdexpSimd<__m512>();
}
#endif
#if defined(__aarch64__)
TEST(LdexpNeon, RoundTrip) {
  checkLdexpSimd<float32x4_t>();
}
#endif
#if __has_include("hwy/highway.h")
TEST(LdexpHwy, RoundTrip) {
  checkLdexpSimd<dfm::HwyFloat>();
}
#endif
