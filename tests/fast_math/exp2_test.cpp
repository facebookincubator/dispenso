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

TEST(Exp2, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  auto exp2 = dispenso::fast_math::exp2<float, dispenso::fast_math::MaxAccuracyTraits>;

  EXPECT_EQ(exp2(0.0f), 1.0f);
  EXPECT_EQ(exp2(-kInf), 0.0f);
  EXPECT_EQ(exp2(kInf), kInf);
  EXPECT_NE(exp2(kNaN), exp2(kNaN));
}

#ifdef _WIN32
auto gtfunc = [](float f) { return static_cast<float>(::exp2l(f)); };
#else
auto gtfunc = ::exp2f;
#endif //_WIN32

TEST(Exp2, RangeSmall) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, -127.0f, -1.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp2, RangeMedium) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, -1.0f, 0.0f);
  EXPECT_LE(res, 1u);
}

TEST(Exp2, RangeLarge) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(gtfunc, dispenso::fast_math::exp2<float>, 0.0f, 128.0f);
  EXPECT_LE(res, 1u);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kExp2MaxUlps = 1;
FAST_MATH_ACCURACY_TESTS(Exp2All, gtfunc, dfm::exp2, -127.0f, 128.0f, kExp2MaxUlps)

// Special values tested across all SIMD backends.
// Default exp2 doesn't handle NaN/Inf — only test in-range values.
static const float kExp2Specials[] =
    {0.0f, -0.0f, 1.0f, -1.0f, 3.5f, -3.5f, 10.0f, -10.0f, 0.5f, 127.0f, -126.0f};
FAST_MATH_SPECIAL_TESTS(Exp2Special, gtfunc, dfm::exp2, kExp2Specials, kExp2MaxUlps)
