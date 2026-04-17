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

TEST(Frexp, SpecialVals) {
  int exp;
  auto res = dispenso::fast_math::frexp(-std::numeric_limits<float>::infinity(), &exp);
  EXPECT_EQ(res, -std::numeric_limits<float>::infinity());
  res = dispenso::fast_math::frexp(std::numeric_limits<float>::infinity(), &exp);
  EXPECT_EQ(res, std::numeric_limits<float>::infinity());

  res = dispenso::fast_math::frexp(std::numeric_limits<float>::quiet_NaN(), &exp);
  EXPECT_NE(res, res);
}

// evalForEach skips subnormals (nextafter jumps from 0 to float::min), which
// is consistent with dispenso::frexp's intentional subnormal pass-through.
TEST(Frexp, RangeNeg) {
  dispenso::fast_math::evalForEach(-std::numeric_limits<float>::max(), 0.0f, [](float f) {
    int exp;
    int expT;
    float a = ::frexpf(f, &exp);
    float aT = dispenso::fast_math::frexp(f, &expT);
    EXPECT_EQ(exp, expT) << "frexpf(" << f << ",&exp)";
    EXPECT_EQ(a, aT) << "frexpf(" << f << ",&exp)";
  });
}

TEST(Frexp, RangePos) {
  dispenso::fast_math::evalForEach(0.0f, std::numeric_limits<float>::max(), [](float f) {
    int exp;
    int expT;
    float a = ::frexpf(f, &exp);
    float aT = dispenso::fast_math::frexp(f, &expT);
    EXPECT_EQ(exp, expT) << "frexpf(" << f << ",&exp)";
    EXPECT_EQ(a, aT) << "frexpf(" << f << ",&exp)";
  });
}

// Unified SIMD frexp test — verify mantissa and exponent per lane.
// Uses SimdTestTraits for float load/store, and stores int exponent via
// aligned buffer + platform-specific store (frexp output is IntType_t<Flt>).
//
// Note: subnormals (denorm_min, etc.) are excluded. dispenso::frexp passes
// subnormals through unchanged with exponent 0, rather than normalizing them
// into [0.5, 1) as std::frexp does. This is intentional — the ldexp(frexp(x))
// round-trip still works, but the mantissa contract is violated. See the frexp
// docstring in fast_math.h for details.
static const float kFrexpInputs[] = {
    0.0f,
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    2.0f,
    -2.0f,
    100.0f,
    0.001f,
    1e-20f,
    1e20f,
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN()};

template <typename Flt>
void checkFrexpSimd() {
  using Traits = SimdTestTraits<Flt>;
  using IntT = dfm::IntType_t<Flt>;
  const int32_t N = Traits::laneCount();
  constexpr int32_t numInputs =
      static_cast<int32_t>(sizeof(kFrexpInputs) / sizeof(kFrexpInputs[0]));

  alignas(64) float buf[kMaxSimdLanes];
  alignas(64) float outMantissa[kMaxSimdLanes];

  for (int32_t base = 0; base < numInputs; base += N) {
    int32_t count = std::min(N, numInputs - base);
    for (int32_t i = 0; i < count; ++i) {
      buf[i] = kFrexpInputs[base + i];
    }
    for (int32_t i = count; i < N; ++i) {
      buf[i] = buf[count - 1];
    }

    IntT exponent;
    Flt result = dfm::frexp(Traits::load(buf), &exponent);
    Traits::store(result, outMantissa);

    // Extract int exponents by storing the result via bit_cast to float.
    alignas(64) float expAsFloat[kMaxSimdLanes];
    Traits::store(dfm::bit_cast<Flt>(exponent), expAsFloat);

    for (int32_t i = 0; i < count; ++i) {
      int scalarExp;
      float scalarMantissa = ::frexpf(buf[i], &scalarExp);

      if (std::isnan(buf[i])) {
        EXPECT_TRUE(std::isnan(outMantissa[i])) << "frexp(NaN) mantissa should be NaN";
      } else if (std::isinf(buf[i])) {
        EXPECT_EQ(outMantissa[i], buf[i]) << "frexp(inf) mantissa should be inf";
      } else {
        EXPECT_EQ(outMantissa[i], scalarMantissa) << "frexp(" << buf[i] << ") mantissa mismatch";
        int32_t actualExp = dfm::bit_cast<int32_t>(expAsFloat[i]);
        EXPECT_EQ(actualExp, scalarExp)
            << "frexp(" << buf[i] << ") exponent mismatch: got " << actualExp;
      }
    }
  }
}

#if defined(__SSE4_1__)
TEST(FrexpSse, SpecialVals) {
  checkFrexpSimd<__m128>();
}
#endif
#if defined(__AVX2__)
TEST(FrexpAvx, SpecialVals) {
  checkFrexpSimd<__m256>();
}
#endif
#if defined(__AVX512F__)
TEST(FrexpAvx512, SpecialVals) {
  checkFrexpSimd<__m512>();
}
#endif
#if defined(__aarch64__)
TEST(FrexpNeon, SpecialVals) {
  checkFrexpSimd<float32x4_t>();
}
#endif
#if __has_include("hwy/highway.h")
TEST(FrexpHwy, SpecialVals) {
  checkFrexpSimd<dfm::HwyFloat>();
}
#endif
