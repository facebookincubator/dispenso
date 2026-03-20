/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/fast_math/fast_math.h>

#include "eval.h"

#include <gtest/gtest.h>

TEST(Acos, OutOfRange) {
  auto res = dispenso::fast_math::acos(1.00001f);
  EXPECT_NE(res, res);
  res = dispenso::fast_math::acos(-1.00001f);
  EXPECT_NE(res, res);
}

TEST(Acos, SpecialVals) {
  auto res = dispenso::fast_math::acos(-1.0f);
  EXPECT_EQ(res, kPi);
  res = dispenso::fast_math::acos(1.0f);
  EXPECT_EQ(res, 0.0f);
  res = dispenso::fast_math::acos(0.0f);
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
  uint32_t ulps =
      dispenso::fast_math::evalAccuracy(acosf, dispenso::fast_math::acos<float>, -1.0f, 1.0f);

  EXPECT_LE(ulps, kMaxUlpsError);
}
