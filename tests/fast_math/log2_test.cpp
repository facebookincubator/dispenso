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

auto log2_w_bounds = dispenso::fast_math::log2<float, dispenso::fast_math::MaxAccuracyTraits>;

#ifdef _WIN32
float groundTruth(float x) {
  return static_cast<float>(::log2l(x));
}
#else
auto groundTruth = ::log2f;
#endif // _WIN32

TEST(Log2, SpecialValues) {
  constexpr auto kInf = std::numeric_limits<float>::infinity();
  constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(log2_w_bounds(0.0f), -kInf);
  EXPECT_NE(log2_w_bounds(-1.0f), log2_w_bounds(-1.0f));
  EXPECT_EQ(log2_w_bounds(kInf), kInf);
  EXPECT_NE(log2_w_bounds(kNaN), log2_w_bounds(kNaN));

  EXPECT_EQ(log2_w_bounds(1e-45f), groundTruth(1e-45f));
}

constexpr int kNegUlps = 1;
constexpr int kPosUlps = 1;

TEST(Log2WBounds, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, log2_w_bounds, 0.0f, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kNegUlps));
}

TEST(Log2WBounds, RangePos) {
  uint32_t res = dispenso::fast_math::evalAccuracy(groundTruth, log2_w_bounds, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kPosUlps));
}

TEST(Log2, RangeNeg) {
  uint32_t res = dispenso::fast_math::evalAccuracy(
      groundTruth, dispenso::fast_math::log2<float>, 3e-38f, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kNegUlps));
}

TEST(Log2, RangePos) {
  uint32_t res =
      dispenso::fast_math::evalAccuracy(groundTruth, dispenso::fast_math::log2<float>, 1.0f);
  EXPECT_LE(res, static_cast<uint32_t>(kPosUlps));
}

// Wrapper for MaxAccuracyTraits — macro instantiates func<Flt>.
template <typename Flt>
Flt log2_max(Flt x) {
  return dfm::log2<Flt, dfm::MaxAccuracyTraits>(x);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
constexpr uint32_t kLog2MaxUlps = 1;
FAST_MATH_ACCURACY_TESTS(
    Log2MaxAccAll,
    groundTruth,
    log2_max,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLog2MaxUlps)
FAST_MATH_ACCURACY_TESTS(
    Log2DefaultAll,
    groundTruth,
    dfm::log2,
    3e-38f,
    std::numeric_limits<float>::max(),
    kLog2MaxUlps)

// Special values tested across all SIMD backends.
// MaxAccuracy handles out-of-domain inputs (0, negatives, NaN, Inf).
static const float kLog2MaxAccSpecials[] = {
    1.0f,
    0.5f,
    2.0f,
    4.0f,
    8.0f,
    16.0f,
    1024.0f,
    0.25f,
    0.125f,
    0.0f,
    -0.0f,
    -1.0f,
    -100.0f,
    std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
FAST_MATH_SPECIAL_TESTS(Log2MaxAccSpecial, groundTruth, log2_max, kLog2MaxAccSpecials, kLog2MaxUlps)

// Default traits only works for positive normal floats.
static const float kLog2DefaultSpecials[] = {1.0f, 0.5f, 2.0f, 4.0f, 8.0f, 100.0f, 0.25f, 1e-10f};
FAST_MATH_SPECIAL_TESTS(
    Log2DefaultSpecial,
    groundTruth,
    dfm::log2,
    kLog2DefaultSpecials,
    kLog2MaxUlps)
