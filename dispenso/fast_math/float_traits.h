/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <cstdint>

#if !defined(__CUDACC__)
#if defined(__SSE__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#endif

#include <dispenso/platform.h>

namespace dispenso {
namespace fast_math {

namespace detail {
// Compile-time bounds check for shuffle/bitmask indices. Extracted to a helper
// so that static_assert doesn't inflate lizard's cyclomatic complexity count.
template <int kMax, int... Is>
constexpr bool indicesInRange() {
  return ((Is >= 0 && Is <= kMax) && ...);
}

// Build a bitmask from compile-time 0/1 values. Each Bi contributes bit i.
// Extracted to avoid lizard counting each ternary as a branch.
template <int... Bs>
constexpr uint32_t buildBitmask() {
  uint32_t mask = 0;
  int bit = 0;
  ((mask |= (Bs ? (1u << bit) : 0u), ++bit), ...);
  return mask;
}
} // namespace detail

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
  static constexpr uint32_t kLanes = 1;

  static DISPENSO_INLINE float load(const float* ptr) {
    return *ptr;
  }
  static DISPENSO_INLINE void store(float* ptr, float val) {
    *ptr = val;
  }
  static DISPENSO_INLINE float extract(float val, uint32_t /*lane*/) {
    return val;
  }
  static DISPENSO_INLINE bool testBit(bool val, uint32_t /*lane*/) {
    return val;
  }
  static DISPENSO_INLINE uint32_t maskBits(bool val) {
    return val ? 1u : 0u;
  }
  static DISPENSO_INLINE float maskLoad(const float* ptr, uint32_t count) {
    return count > 0 ? *ptr : 0.0f;
  }
  static DISPENSO_INLINE void maskStore(float* ptr, uint32_t count, float val) {
    if (count > 0) {
      *ptr = val;
    }
  }

  static DISPENSO_INLINE float sqrt(float x) {
#if defined(__CUDACC__)
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
  }

  static DISPENSO_INLINE float floor(float x) {
#if defined(__CUDACC__)
    return floorf(x);
#else
    return std::floor(x);
#endif
  }

  static DISPENSO_INLINE float rcp(float x) {
#if defined(__CUDACC__)
#if defined(__CUDA_ARCH__)
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
#elif defined(__SSE__)
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
    return a < b ? a : b;
  }

  static DISPENSO_INLINE float max(float a, float b) {
    return a > b ? a : b;
  }

  static DISPENSO_INLINE float fma(float a, float b, float c) {
#if defined(__CUDACC__)
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#elif defined(_MSC_VER)
    return fmaf(a, b, c);
#else
    return __builtin_fmaf(a, b, c);
#endif
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
  }

  // shuffle: for scalar, only identity permutation (index 0) is valid.
  template <int I0>
  static DISPENSO_INLINE float shuffle(float v) {
    static_assert(I0 == 0, "scalar shuffle index must be 0");
    return v;
  }

  // bitmask: for scalar, returns bool.
  template <int B0>
  static DISPENSO_INLINE bool bitmask() {
    static_assert(B0 == 0 || B0 == 1, "bitmask values must be 0 or 1");
    return B0 != 0;
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

#if defined(__CUDACC__)
namespace detail {
DISPENSO_INLINE double doubleSqrt(double x) {
#if defined(__CUDA_ARCH__)
  return __dsqrt_rn(x);
#elif defined(_MSC_VER)
  return sqrt(x);
#else
  return __builtin_sqrt(x);
#endif
}

DISPENSO_INLINE double doubleFma(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
  return __fma_rn(a, b, c);
#elif defined(_MSC_VER)
  return fma(a, b, c);
#else
  return __builtin_fma(a, b, c);
#endif
}
} // namespace detail
#endif

template <>
struct FloatTraits<double> {
  using IntType = int64_t;
  using UintType = uint64_t;

  static DISPENSO_INLINE double sqrt(double x) {
#if defined(__CUDACC__)
    return detail::doubleSqrt(x);
#else
    return std::sqrt(x);
#endif
  }

  static DISPENSO_INLINE double fma(double a, double b, double c) {
#if defined(__CUDACC__)
    return detail::doubleFma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
  }
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
