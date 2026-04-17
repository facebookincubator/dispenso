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

TEST(Acos, OutOfRange) {
  auto res = dfm::acos(1.00001f);
  EXPECT_NE(res, res);
  res = dfm::acos(-1.00001f);
  EXPECT_NE(res, res);
}

TEST(Acos, SpecialVals) {
  auto res = dfm::acos(-1.0f);
  EXPECT_EQ(res, kPi);
  res = dfm::acos(1.0f);
  EXPECT_EQ(res, 0.0f);
  res = dfm::acos(0.0f);
  EXPECT_FLOAT_EQ(res, kPi_2);
}

// Apple's acos implmentation for x86_64 does not match their implementation for
// ARM, nor does it match glibc or Window's acos implementations in the last bit(s).
#if defined(__APPLE__) && (defined(__i386__) || defined(__x86_64__))
constexpr uint32_t kMaxUlpsError = 4;
#else
constexpr uint32_t kMaxUlpsError = 3;
#endif // apple x86(_64)

TEST(Acos, Range) {
  uint32_t ulps = dfm::evalAccuracy(acosf, dfm::acos<float>, -1.0f, 1.0f);
  EXPECT_LE(ulps, kMaxUlpsError);
}

// Unified accuracy tests — scalar + all SIMD backends, same threshold.
FAST_MATH_ACCURACY_TESTS(AcosAll, ::acosf, dfm::acos, -1.0f, 1.0f, kMaxUlpsError)

// Special values tested across all SIMD backends (including out-of-range → NaN).
static const float kAcosSpecials[] = {
    0.0f,
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    0.25f,
    -0.25f,
    0.75f,
    0.99999f,
    1.00001f,
    -1.00001f,
    2.0f,
    -5.0f,
    10.0f,
    -10.0f,
    100.0f,
    -100.0f};
FAST_MATH_SPECIAL_TESTS(AcosSpecial, ::acosf, dfm::acos, kAcosSpecials, kMaxUlpsError)
