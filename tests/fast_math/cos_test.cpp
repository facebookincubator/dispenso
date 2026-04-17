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

auto cos_accurate = dispenso::fast_math::cos<float, dispenso::fast_math::MaxAccuracyTraits>;

// MacOS seems to have a slightly busted cos and cosf, but cosl seems to agree with other platforms.
#if defined(__APPLE__)
float gt_cos(float x) {
  return ::cosl(x);
}
#else
auto gt_cos = ::cosf;
#endif //__APPLE__

TEST(Cos, SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
  auto cos = dispenso::fast_math::cos<float, dispenso::fast_math::MaxAccuracyTraits>;

  EXPECT_NE(cos(-kInf), cos(-kInf));
  EXPECT_NE(cos(kInf), cos(kInf));
  EXPECT_NE(cos(kNaN), cos(kNaN));
  EXPECT_EQ(cos(0.0f), 1.0f);
}

constexpr uint32_t kCosAccurateUlps = 1;
constexpr uint32_t kCosAccurateUlpsLarge = 1;
constexpr uint32_t kCosAccurateUlpsVeryLarge = 2;

constexpr uint32_t kCosUlps = 1;
constexpr uint32_t kCosUlpsLarge = 1;
constexpr uint32_t kCosUlpsVeryLarge = 2;

TEST(Cos, RangePi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -kPi, kPi);

  EXPECT_LE(result, kCosAccurateUlps);
}

TEST(Cos, Range128Pi) {
  auto result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -128 * kPi, 128 * kPi);

  EXPECT_LE(result, kCosAccurateUlpsLarge);
}

TEST(Cos, Range1MPi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, -(1 << 20) * kPi, -128 * kPi);
  EXPECT_LE(result, kCosAccurateUlpsVeryLarge);
  result = dispenso::fast_math::evalAccuracy(gt_cos, cos_accurate, 128 * kPi, (1 << 20) * kPi);

  EXPECT_LE(result, kCosAccurateUlpsVeryLarge);
}

TEST(CosLessAccurate, RangePi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, -kPi, kPi);

  EXPECT_LE(result, kCosUlps);
}

TEST(CosLessAccurate, Range128Pi) {
  auto result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, -128 * kPi, -kPi);

  EXPECT_LE(result, kCosUlpsLarge);

  result =
      dispenso::fast_math::evalAccuracy(gt_cos, dispenso::fast_math::cos<float>, kPi, 128 * kPi);

  EXPECT_LE(result, kCosUlpsLarge);
}

TEST(CosLessAccurate, Range32768Pi) {
  auto result = dispenso::fast_math::evalAccuracy(
      gt_cos, dispenso::fast_math::cos<float>, -32768 * kPi, -128 * kPi);

  EXPECT_LE(result, kCosUlpsVeryLarge);

  result = dispenso::fast_math::evalAccuracy(
      gt_cos, dispenso::fast_math::cos<float>, 128 * kPi, 32768 * kPi);

  EXPECT_LE(result, kCosUlpsVeryLarge);
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt cos_max(Flt x) {
  return dfm::cos<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
FAST_MATH_ACCURACY_TESTS(
    CosDefaultAll,
    gt_cos,
    dfm::cos,
    -32768 * kPi,
    32768 * kPi,
    kCosUlpsVeryLarge)
FAST_MATH_ACCURACY_TESTS(
    CosMaxAccAll,
    gt_cos,
    cos_max,
    -(1 << 20) * kPi,
    (1 << 20) * kPi,
    kCosAccurateUlpsVeryLarge)

// Special values tested across all SIMD backends.
static const float kCosSpecials[] = {
    0.0f,
    -0.0f,
    kPi,
    -kPi,
    kPi_2,
    -kPi_2,
    2.0f * kPi,
    -2.0f * kPi,
    100.0f,
    -100.0f,
    1000.0f,
    -1000.0f,
    1e-6f,
    std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(CosMaxAccSpecial, gt_cos, cos_max, kCosSpecials, kCosAccurateUlpsVeryLarge)
FAST_MATH_SPECIAL_TESTS(CosDefaultSpecial, gt_cos, dfm::cos, kCosSpecials, kCosUlpsVeryLarge)
