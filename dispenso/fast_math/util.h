/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#if defined(__SSE__) || defined(__AVX__)
#include <immintrin.h>
#endif // __SSE__

#include <cstring>
#include <type_traits>

#include "float_traits.h"

namespace dispenso {
namespace fast_math {

// Reinterpret the bits of `src` as type `To`. Equivalent to std::bit_cast (C++20).
template <class To, class From>
DISPENSO_INLINE std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_default_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be default constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

// Return the ULP distance between two floats. Denormals are treated as zero.
// Handles mixed-sign comparisons correctly by mapping IEEE 754 bit patterns
// to a linear integer representation where negative floats map to negative integers.
DISPENSO_INLINE uint32_t float_distance(float a, float b) {
  uint32_t ai = bit_cast<uint32_t>(a);
  uint32_t bi = bit_cast<uint32_t>(b);

  // Handle denormal values as zero
  ai = ((ai & 0x7f800000) == 0) ? 0 : ai;
  bi = ((bi & 0x7f800000) == 0) ? 0 : bi;

  // Map to linear integer space: positive floats stay as-is,
  // negative floats map to negative integers (sign-magnitude → two's complement).
  auto toLinear = [](uint32_t bits) -> int32_t {
    return (bits & 0x80000000u) ? static_cast<int32_t>(0x80000000u - bits)
                                : static_cast<int32_t>(bits);
  };
  int32_t la = toLinear(ai);
  int32_t lb = toLinear(bi);
  int32_t diff = la - lb;
  return static_cast<uint32_t>(diff < 0 ? -diff : diff);
}

// Absolute value: clears the sign bit.  Works for all float/SIMD types.
template <typename Flt>
DISPENSO_INLINE Flt fabs(Flt x) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return fabs(SimdType_t<Flt>(x)).v;
  } else {
    using Uint = UintType_t<Flt>;
    return bit_cast<Flt>(bit_cast<Uint>(x) & Uint(0x7fffffff));
  }
}

// Return +1.0 or -1.0 matching the sign bit of x. +0 gives +1, -0 gives -1.
template <typename Flt>
DISPENSO_INLINE Flt signof(Flt x) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return signof(SimdType_t<Flt>(x)).v;
  } else {
    using Uint = UintType_t<Flt>;
    return bit_cast<Flt>((bit_cast<Uint>(x) & 0x80000000) | FloatTraits<Flt>::kOne);
  }
}

// Integer sign: returns +1 for i >= 0, -1 for i < 0. Scalar version (kBoolIsMask=false).
template <typename Flt>
DISPENSO_INLINE std::enable_if_t<!FloatTraits<Flt>::kBoolIsMask, IntType_t<Flt>> signofi(
    IntType_t<Flt> i) {
  return 1 - 2 * (i < 0);
}

// Integer sign: returns +1 for i >= 0, -1 for i < 0. SIMD mask version (kBoolIsMask=true).
template <typename Flt>
DISPENSO_INLINE std::enable_if_t<FloatTraits<Flt>::kBoolIsMask, IntType_t<Flt>> signofi(
    IntType_t<Flt> i) {
  // Explicit IntType_t cast handles AVX-512 where (i < 0) returns Avx512Mask, not Avx512Int32.
  auto neg = IntType_t<Flt>(i < 0);
  return 1 - (2 & neg);
}

// True if the float (as int bits) has all exponent bits set (inf or NaN).
// Returns bool for scalar types, SIMD lane mask for SIMD types.
template <typename Flt>
DISPENSO_INLINE auto nonnormal(IntType_t<Flt> i) {
  return (i & 0x7f800000) == 0x7f800000;
}

// True if the float (as int bits) is inf, NaN, or zero.
// Returns bool for scalar types, SIMD lane mask for SIMD types.
template <typename Flt>
DISPENSO_INLINE auto nonnormalOrZero(IntType_t<Flt> i) {
  auto m = i & 0x7f800000;
  return (m == 0x7f800000) | (m == 0);
}

template <typename Flt>
DISPENSO_INLINE auto nonnormal(Flt f) {
  return nonnormal<Flt>(bit_cast<IntType_t<Flt>>(f));
}

// Truncate float toward zero, returning int. No inf/NaN guard — caller must
// ensure inputs are finite (e.g. after range reduction by a finite constant).
// Maps to cvttps2epi32 (SSE), vcvtq_s32_f32 (NEON), or (int)f (scalar).
template <typename Flt>
DISPENSO_INLINE IntType_t<Flt> convert_to_int_trunc(Flt f) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return convert_to_int_trunc(SimdType_t<Flt>(f)).v;
  } else {
    return static_cast<IntType_t<Flt>>(f);
  }
}

// Truncate float toward zero, returning int. Inf/NaN lanes are zeroed.
// Use when the input may contain non-finite values.
template <typename Flt>
DISPENSO_INLINE IntType_t<Flt> convert_to_int_trunc_safe(Flt f) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return convert_to_int_trunc_safe(SimdType_t<Flt>(f)).v;
  } else {
    auto fi = bit_cast<IntType_t<Flt>>(f);
    if ((fi & 0x7f800000) == 0x7f800000) {
      return 0;
    }
    return static_cast<IntType_t<Flt>>(f);
  }
}

// Convert float to int using round-to-nearest-even (SSE cvtss2si semantics).
// For non-normal inputs (inf/NaN), returns 0 to avoid undefined behavior.
template <typename Flt>
DISPENSO_INLINE IntType_t<Flt> convert_to_int(Flt f) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return convert_to_int(SimdType_t<Flt>(f)).v;
  } else {
#if defined(__SSE__)
    return _mm_cvtss_si32(_mm_set_ss(f));
#else
    // Round to nearest even via magic number addition, guard non-normals → 0.
    auto fi = bit_cast<IntType_t<Flt>>(f);
    if ((fi & 0x7f800000) == 0x7f800000) {
      return 0;
    }
    constexpr float kMagic = FloatTraits<Flt>::kMagic; // 1.5f * 2^23
    return static_cast<IntType_t<Flt>>((f + kMagic) - kMagic);
#endif // __SSE__
  }
}

// Convert float to int with round-to-nearest-even, clamping to [kMin, kMax].
// For non-normal inputs (inf/NaN), returns 0 to avoid undefined behavior.
template <typename Flt, IntType_t<Flt> kMin, IntType_t<Flt> kMax>
DISPENSO_INLINE IntType_t<Flt> convert_to_int_clamped(Flt f) {
#if defined(__SSE4_1__)
  static const __m128i kMn = _mm_set1_epi32(kMin);
  static const __m128i kMx = _mm_set1_epi32(kMax);
  __m128i i = _mm_cvtps_epi32(_mm_set_ss(f));
  i = _mm_min_epi32(i, kMx);
  i = _mm_max_epi32(i, kMn);
  return _mm_cvtsi128_si32(i);
#else
  // Round to nearest even via magic number addition (same technique as rangeReduce),
  // then clamp. Guard against non-normal inputs (inf/NaN) → return 0.
  auto fi = bit_cast<IntType_t<Flt>>(f);
  if ((fi & 0x7f800000) == 0x7f800000) {
    return 0;
  }
  constexpr float kMagic = FloatTraits<Flt>::kMagic; // 1.5f * 2^23
  auto rounded = static_cast<IntType_t<Flt>>((f + kMagic) - kMagic);
  if (rounded > kMax)
    return kMax;
  if (rounded < kMin)
    return kMin;
  return rounded;
#endif // __SSE4_1__
}

// Floor for values within integer range. Uses SSE4.1 roundss when available.
template <typename Flt>
DISPENSO_INLINE Flt floor_small(Flt x) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return floor_small(SimdType_t<Flt>(x)).v;
  } else {
    IntType_t<Flt> i = x;
    Flt xi = i;
    if constexpr (FloatTraits<Flt>::kBoolIsMask) {
      return i - ((x < xi) & 1);
    } else {
      return i - (x < xi) * 1;
    }
  }
}

template <>
DISPENSO_INLINE float floor_small(float x) {
#if defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 r = _mm_floor_ss(f, f);
  return _mm_cvtss_f32(r);
#else
  return std::floor(x);
#endif // __SSE4_1__
}

// Minimum of x and mn. If x is NaN and mn is not, returns mn (relied-upon NaN behavior).
template <typename Flt>
DISPENSO_INLINE Flt min(Flt x, Flt mn);

template <>
DISPENSO_INLINE float min(float x, float mn) {
#if defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 fmn = _mm_set_ss(mn);
  // Ordering matters here for NaN behavior
  __m128 r = _mm_min_ss(f, fmn);
  return _mm_cvtss_f32(r);
#else
  return x < mn ? x : mn;
#endif //__SSE4_1__
}

// Clamp x to [mn, mx]. If x is NaN, NaN propagates (result is NaN).
// Use when NaN should signal an error rather than be silently clamped.
template <typename Flt>
DISPENSO_INLINE Flt clamp_allow_nan(Flt x, Flt mn, Flt mx);

template <>
DISPENSO_INLINE float clamp_allow_nan(float x, float mn, float mx) {
#if defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 fmn = _mm_set_ss(mn);
  __m128 fmx = _mm_set_ss(mx);
  // Ordering matters here for NaN behavior
  __m128 r = _mm_max_ss(fmn, _mm_min_ss(fmx, f));
  return _mm_cvtss_f32(r);
#else
  mx = (mx < x) ? mx : x;
  return (mx < mn) ? mn : mx;
#endif // __SSE4_1__
}

// Clamp x to [mn, mx]. If x is NaN, returns a value in [mn, mx] (NaN is suppressed).
// Use when a valid output is required regardless of input.
template <typename Flt>
DISPENSO_INLINE Flt clamp_no_nan(Flt x, Flt mn, Flt mx);

template <>
DISPENSO_INLINE float clamp_no_nan(float x, float mn, float mx) {
#if defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 fmn = _mm_set_ss(mn);
  __m128 fmx = _mm_set_ss(mx);
  // Ordering matters here for NaN behavior
  __m128 r = _mm_max_ss(fmn, _mm_min_ss(f, fmx));
  return _mm_cvtss_f32(r);

  // TODO: See vrndmq_f32 for ARM NEON

#else
  mx = (mx > x) ? x : mx;
  return (mx > mn) ? mx : mn;
#endif // __SSE4_1__
}

// Load table[index]. Scalar version is a plain array access; SIMD versions use gather instructions.
template <typename Flt>
DISPENSO_INLINE Flt gather(const float* table, IntType_t<Flt> index);

template <>
DISPENSO_INLINE float gather(const float* table, int32_t index) {
  return table[index];
}

// Return 0 if b is true, 1 if b is false (negated bool as numeric).
template <typename T, typename BoolT>
DISPENSO_INLINE T nbool_as_one(BoolT b);
template <>
DISPENSO_INLINE float nbool_as_one<float, bool>(bool b) {
  return b ? 0.0f : 1.0f;
}

template <>
DISPENSO_INLINE int32_t nbool_as_one<int32_t, bool>(bool b) {
  return static_cast<int32_t>(!b);
}

// Return 1 if b is true, 0 if b is false (bool as numeric).
template <typename T, typename BoolT>
DISPENSO_INLINE T bool_as_one(BoolT b);
template <>
DISPENSO_INLINE float bool_as_one<float, bool>(bool b) {
  return b ? 1.0f : 0.0f;
}

template <>
DISPENSO_INLINE int32_t bool_as_one<int32_t, bool>(bool b) {
  return static_cast<int32_t>(b);
}

// Convert bool to bitmask: true → all-ones (0xFFFFFFFF), false → 0x0.
// For SIMD types, this converts a lane mask to an integer mask for bitwise ops.
template <typename T, typename BoolT>
DISPENSO_INLINE T bool_as_mask(BoolT b) {
  T mask = b;
  return ~(mask - 1);
}

// Return val if b is true, 0 otherwise. Uses bitwise AND via bool_as_mask.
template <typename T, typename BoolT>
DISPENSO_INLINE T bool_apply_or_zero(BoolT b, T val) {
  return bit_cast<T>(bool_as_mask<IntType_t<T>>(b) & bit_cast<IntType_t<T>>(val));
}

// Fast integer division by 3 using multiply-and-shift. Used in cbrt magic constant computation.
DISPENSO_INLINE int32_t int_div_by_3(int32_t i) {
  return ((uint64_t(i) * 0x55555556) >> 32);
}

} // namespace fast_math
} // namespace dispenso

// Auto-detect SIMD backends and include FloatTraits specializations.
#if defined(__SSE4_1__)
#include <dispenso/fast_math/float_traits_x86.h>
#endif

#if defined(__AVX2__)
#include <dispenso/fast_math/float_traits_avx.h>
#endif

#if defined(__AVX512F__)
#include <dispenso/fast_math/float_traits_avx512.h>
#endif

#if defined(__aarch64__)
#include <dispenso/fast_math/float_traits_neon.h>
#endif

#if __has_include("hwy/highway.h")
#include <dispenso/fast_math/float_traits_hwy.h>
#endif

namespace dispenso {
namespace fast_math {

// Best available SIMD float type for the current platform.
// Prefer native intrinsic wrappers over Highway for lower overhead.
// Highway is a fallback for platforms without a native wrapper.
#if defined(__aarch64__)
using DefaultSimdFloat = NeonFloat;
#elif defined(__AVX512F__)
using DefaultSimdFloat = Avx512Float;
#elif defined(__AVX2__)
using DefaultSimdFloat = AvxFloat;
#elif defined(__SSE4_1__)
using DefaultSimdFloat = SseFloat;
#elif __has_include("hwy/highway.h")
using DefaultSimdFloat = HwyFloat;
#else
using DefaultSimdFloat = float;
#endif

} // namespace fast_math
} // namespace dispenso
