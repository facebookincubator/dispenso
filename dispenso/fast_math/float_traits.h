/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__SSE__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <dispenso/platform.h>

namespace dispenso {
namespace fast_math {

// Float-precision math constants, avoiding repeated static_cast<float>(M_PI) etc.
constexpr float kPi = static_cast<float>(M_PI);
constexpr float kPi_2 = static_cast<float>(M_PI_2);
constexpr float kPi_4 = static_cast<float>(M_PI_4);
constexpr float kLn2 = static_cast<float>(M_LN2);
constexpr float k1_Ln2 = static_cast<float>(1.0 / M_LN2);

template <typename T>
struct FloatTraits {};

template <>
struct FloatTraits<float> {
  using IntType = int32_t;
  using UintType = uint32_t;
  using BoolType = bool;

  static constexpr uint32_t kOne = 0x3f800000;

  static constexpr float kMagic = 12582912.f; // 1.5 * 2**23

  static constexpr bool kBoolIsMask = false;

  static DISPENSO_INLINE float sqrt(float x) {
    return std::sqrt(x);
  }

  static DISPENSO_INLINE float rcp(float x) {
#if defined(__SSE__)
    return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
#elif defined(__aarch64__)
    return vrecpes_f32(x);
#else
    return 1.0f / x;
#endif
  }

  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(bool b, Arg x, Arg y) {
    return b ? x : y;
  }

  template <typename Arg>
  static DISPENSO_INLINE Arg apply(bool b, Arg x) {
    // For kBoolIsMask cases, this is b & x
    return b * x;
  }

  static DISPENSO_INLINE float min(float a, float b) {
    return std::min(a, b);
  }

  static DISPENSO_INLINE float max(float a, float b) {
    return std::max(a, b);
  }

  static DISPENSO_INLINE float fma(float a, float b, float c) {
    return std::fma(a, b, c);
  }
};

template <>
struct FloatTraits<int32_t> {
  using IntType = int32_t;
};

template <>
struct FloatTraits<uint32_t> {
  using IntType = uint32_t;
};

// Non-deduced context helper: prevents template argument deduction on a parameter.
// Use as function parameter type to force callers to rely on deduction from other args.
template <typename T>
struct NonDeducedHelper {
  using type = T;
};
template <typename T>
using NonDeduced = typename NonDeducedHelper<T>::type;

// Maps raw SIMD intrinsic types to their wrapper types for template deduction.
// Default (identity): scalar types and wrapper types map to themselves.
// Specializations in backend headers map __m128 → SseFloat, __m256 → AvxFloat, etc.
// This enables fm::sin(__m128_val) to work via automatic forwarding.
template <typename T>
struct SimdTypeFor {
  using type = T;
};
template <typename T>
using SimdType_t = typename SimdTypeFor<T>::type;

template <typename Flt>
using IntType_t = typename FloatTraits<SimdType_t<Flt>>::IntType;
template <typename Flt>
using UintType_t = typename FloatTraits<SimdType_t<Flt>>::UintType;

template <typename Flt>
using BoolType_t = typename FloatTraits<SimdType_t<Flt>>::BoolType;

} // namespace fast_math
} // namespace dispenso
