/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>

#include <dispenso/fast_math/util.h>

// Float-precision math constants to avoid double-to-float conversion warnings.
constexpr float kPi = static_cast<float>(M_PI);
constexpr float kPi_2 = static_cast<float>(M_PI_2);
constexpr float kPi_4 = static_cast<float>(M_PI_4);

// Reduce accuracy coverage for sanitizer and debug builds, which are much slower.
// Clang uses __has_feature, GCC uses __SANITIZE_ADDRESS__/__SANITIZE_THREAD__.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define REDUCE_TEST_CASES
#elif __has_feature(thread_sanitizer)
#define REDUCE_TEST_CASES
#endif // relevant feature
#endif // has_feature
#if !defined(REDUCE_TEST_CASES)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || !defined(NDEBUG)
#define REDUCE_TEST_CASES
#endif
#endif

namespace dispenso {
namespace fast_math {

namespace detail {

#if !defined(REDUCE_TEST_CASES)
// Use this to skip denormals
inline float nextafter(float f) {
  if (f == 0.0f) {
    return std::numeric_limits<float>::min();
  }

  uint32_t u = bit_cast<uint32_t>(f);
  if (u & 0x80000000) { // negative
    if (u & 0x7f800000) {
      --u;
      if (u & 0x7f800000) {
        return bit_cast<float>(u);
      } else {
        return 0.0f;
      }
    } else {
      return 0.0f;
    }
  } else {
    ++u;
    return bit_cast<float>(u);
  }
}
#else
// Use this to skip denormals, also to skip many "nexts" at a time to speed up tests
inline float nextafter(float f) {
  if (f >= 0.0f && f < std::numeric_limits<float>::min()) {
    return std::numeric_limits<float>::min();
  }

  uint32_t u = bit_cast<uint32_t>(f);
  if (u & 0x80000000) { // negative
    if (u & 0x7f800000) {
      u -= 100;
      if (u & 0x7f800000) {
        return bit_cast<float>(u);
      } else {
        return 0.0f;
      }
    } else {
      return 0.0f;
    }
  } else {
    u += 100;
    return bit_cast<float>(u);
  }
}
#endif // REDUCE_TEST_CASES

} // namespace detail

template <typename Func>
void evalForEach(float rangeBegin, float rangeEnd, Func func) {
  for (float f = rangeBegin; f <= rangeEnd; f = detail::nextafter(f)) {
    func(f);
  }
}

template <typename GroundTruthFunc, typename ApproxFunc>
uint32_t evalAccuracy(
    GroundTruthFunc gtfunc,
    ApproxFunc apxfunc,
    float rangeBegin = -std::numeric_limits<float>::max(),
    float rangeEnd = std::numeric_limits<float>::max()) {
  uint32_t ulps = 0;
  for (float f = rangeBegin; f <= rangeEnd; f = detail::nextafter(f)) {
    float gt = gtfunc(f);
    float a = apxfunc(f);
    uint32_t lulps = float_distance(gt, a);
    ulps = std::max(ulps, lulps);

    if (lulps > 100) {
      printf("%d, f(%.9g): %.9g, %.9g\n", lulps, f, gt, a);
    }
  }
  return ulps;
}

template <typename GroundTruthFunc, typename ApproxFunc>
float evalAccuracyAbs(
    GroundTruthFunc gtfunc,
    ApproxFunc apxfunc,
    float rangeBegin = -std::numeric_limits<float>::max(),
    float rangeEnd = std::numeric_limits<float>::max()) {
  float maxErr = 0.0f;
  for (float f = rangeBegin; f <= rangeEnd; f = detail::nextafter(f)) {
    float gt = gtfunc(f);
    float a = apxfunc(f);
    maxErr = std::max(maxErr, std::fabs(gt - a));
  }
  return maxErr;
}

} // namespace fast_math
} // namespace dispenso
