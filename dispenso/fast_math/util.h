/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

// MSVC never defines __SSE__/__SSE2__ -- it signals SSE availability through
// _M_X64 (always) and _M_IX86_FP (>= 1 means /arch:SSE or better). Without this
// every __SSE__-guarded path below silently falls back to the portable version
// on Windows, which is both slower and, for non-finite input, not equivalent.
#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define DISPENSO_FAST_MATH_HAS_SSE 1
#elif defined(__SSE__)
#define DISPENSO_FAST_MATH_HAS_SSE 1
#else
#define DISPENSO_FAST_MATH_HAS_SSE 0
#endif

#if !defined(__CUDACC__) && (DISPENSO_FAST_MATH_HAS_SSE || defined(__AVX__))
#include <immintrin.h>
#endif

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

// Reciprocal square root — hardware estimate only.
// ~12 bits on x86 (SSE/AVX), ~14 bits on AVX-512, ~8 bits on NEON.
// CUDA: __frsqrt_rn (single hardware instruction).
template <typename Flt>
DISPENSO_INLINE Flt rsqrt_approx(Flt x);

// Reciprocal square root — hardware estimate + Newton refinement.
// ~23 bits (full float32 precision) on all supported platforms.
// x86: 1 Newton iteration on ~12-bit seed.  NEON: 2 iterations on ~8-bit seed.
// CUDA: __frsqrt_rn (IEEE round-to-nearest, single instruction).
template <typename Flt>
DISPENSO_INLINE Flt rsqrt(Flt x);

#if defined(__CUDACC__)
template <>
DISPENSO_INLINE float rsqrt_approx<float>(float x) {
#if defined(__CUDA_ARCH__)
  return __frsqrt_rn(x);
#else
  return rsqrtf(x);
#endif
}
template <>
DISPENSO_INLINE float rsqrt<float>(float x) {
#if defined(__CUDA_ARCH__)
  return __frsqrt_rn(x);
#else
  return rsqrtf(x);
#endif
}
#endif

// Reciprocal — hardware estimate only.
// ~12 bits on x86 (SSE/AVX), ~14 bits on AVX-512, ~8 bits on NEON.
// CUDA: __frcp_rn (IEEE round-to-nearest, single instruction).
template <typename Flt>
DISPENSO_INLINE Flt rcp_approx(Flt x);

// Reciprocal — hardware estimate + Newton refinement.
// ~23 bits (full float32 precision) on all supported platforms.
// x86: 1 Newton iteration on ~12-bit seed.  NEON: 2 iterations on ~8-bit seed.
// CUDA: __frcp_rn (IEEE round-to-nearest, single instruction).
template <typename Flt>
DISPENSO_INLINE Flt rcp(Flt x);

#if defined(__CUDACC__)
template <>
DISPENSO_INLINE float rcp_approx<float>(float x) {
#if defined(__CUDA_ARCH__)
  return __frcp_rn(x);
#else
  return 1.0f / x;
#endif
}
template <>
DISPENSO_INLINE float rcp<float>(float x) {
#if defined(__CUDA_ARCH__)
  return __frcp_rn(x);
#else
  return 1.0f / x;
#endif
}
#endif

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
//
// The result for non-finite input is unspecified and varies by target: the SIMD
// backends and CUDA mask it to 0, while the scalar paths yield INT_MIN, which is
// what cvtss2si produces in hardware. Do not branch on it.
//
// No sentinel can be safe here. 0 is what any x in (-0.5, 0.5) legitimately
// converts to, so a caller treating it as "invalid" also catches its most common
// valid inputs -- and every caller here consumes a range-reduced exponent, where
// 0 is dead centre. Test the input instead (see nonnormal), or order the code so
// bounds checks take precedence over value-dependent shortcuts, as expm1 does.
template <typename Flt>
DISPENSO_INLINE IntType_t<Flt> convert_to_int(Flt f) {
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return convert_to_int(SimdType_t<Flt>(f)).v;
  } else {
#if defined(__CUDACC__)
    auto fi = bit_cast<IntType_t<Flt>>(f);
    if ((fi & 0x7f800000) == 0x7f800000) {
      return 0;
    }
    return static_cast<IntType_t<Flt>>(lrintf(f));
#elif DISPENSO_FAST_MATH_HAS_SSE
    return _mm_cvtss_si32(_mm_set_ss(f));
#else
    // Round to nearest even via magic number addition. Non-finite input yields
    // INT_MIN to match cvtss2si above; see the contract note on this function.
    auto fi = bit_cast<IntType_t<Flt>>(f);
    if ((fi & 0x7f800000) == 0x7f800000) {
      return std::numeric_limits<IntType_t<Flt>>::min();
    }
    constexpr float kMagic = FloatTraits<Flt>::kMagic; // 1.5f * 2^23
    return static_cast<IntType_t<Flt>>((f + kMagic) - kMagic);
#endif // DISPENSO_FAST_MATH_HAS_SSE
  }
}

// Convert float to int with round-to-nearest-even, clamping to [kMin, kMax].
// For non-normal inputs (inf/NaN), returns 0 to avoid undefined behavior.
template <typename Flt, IntType_t<Flt> kMin, IntType_t<Flt> kMax>
DISPENSO_INLINE IntType_t<Flt> convert_to_int_clamped(Flt f) {
#if defined(__CUDACC__)
  auto fi = bit_cast<IntType_t<Flt>>(f);
  if ((fi & 0x7f800000) == 0x7f800000) {
    return 0;
  }
  auto rounded = static_cast<IntType_t<Flt>>(lrintf(f));
  if (rounded > kMax)
    return kMax;
  if (rounded < kMin)
    return kMin;
  return rounded;
#elif defined(__SSE4_1__)
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
#if defined(__CUDACC__)
  return floorf(x);
#elif defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 r = _mm_floor_ss(f, f);
  return _mm_cvtss_f32(r);
#else
  return FloatTraits<float>::floor(x);
#endif
}

// Minimum of x and mn. If x is NaN and mn is not, returns mn (relied-upon NaN behavior).
template <typename Flt>
DISPENSO_INLINE Flt min(Flt x, Flt mn);

template <>
DISPENSO_INLINE float min(float x, float mn) {
#if defined(__CUDACC__)
  return fminf(x, mn);
#elif defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 fmn = _mm_set_ss(mn);
  // Ordering matters here for NaN behavior
  __m128 r = _mm_min_ss(f, fmn);
  return _mm_cvtss_f32(r);
#else
  return x < mn ? x : mn;
#endif
}

// Clamp x to [mn, mx]. If x is NaN, NaN propagates (result is NaN).
// Use when NaN should signal an error rather than be silently clamped.
template <typename Flt>
DISPENSO_INLINE Flt clamp_allow_nan(Flt x, Flt mn, Flt mx);

template <>
DISPENSO_INLINE float clamp_allow_nan(float x, float mn, float mx) {
#if defined(__CUDACC__)
  mx = (mx < x) ? mx : x;
  return (mx < mn) ? mn : mx;
#elif defined(__SSE4_1__)
  __m128 f = _mm_set_ss(x);
  __m128 fmn = _mm_set_ss(mn);
  __m128 fmx = _mm_set_ss(mx);
  // Ordering matters here for NaN behavior
  __m128 r = _mm_max_ss(fmn, _mm_min_ss(fmx, f));
  return _mm_cvtss_f32(r);
#else
  mx = (mx < x) ? mx : x;
  return (mx < mn) ? mn : mx;
#endif
}

// Clamp x to [mn, mx]. If x is NaN, returns a value in [mn, mx] (NaN is suppressed).
// Use when a valid output is required regardless of input.
template <typename Flt>
DISPENSO_INLINE Flt clamp_no_nan(Flt x, Flt mn, Flt mx);

template <>
DISPENSO_INLINE float clamp_no_nan(float x, float mn, float mx) {
#if defined(__CUDACC__)
  mx = (mx > x) ? x : mx;
  return (mx > mn) ? mx : mn;
#elif defined(__SSE4_1__)
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
#endif
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

// Scalar any_true: identity. Enables Flt=float in code templated on SIMD type.
DISPENSO_INLINE bool any_true(bool b) {
  return b;
}

// Return val if b is true, 0 otherwise. Uses bitwise AND via bool_as_mask.
template <typename T, typename BoolT>
DISPENSO_INLINE T bool_apply_or_zero(BoolT b, T val) {
  return bit_cast<T>(bool_as_mask<IntType_t<T>>(b) & bit_cast<IntType_t<T>>(val));
}

// True if x is an integer. Safe for all float values including large, inf, NaN.
// All floats with |x| >= 2^23 are integers (mantissa has no fractional bits).
// For |x| < 2^23, floor_small(x) is valid (no int overflow).
template <typename Flt>
DISPENSO_INLINE BoolType_t<Flt> float_is_int(Flt x) {
  auto ax = fabs(x);
  return (ax >= Flt(8388608.0f)) | (floor_small(x) == x);
}

// True if x is an odd integer.
// For |x| >= 2^24: ULP >= 2, so only even integers are representable → always false.
template <typename Flt>
DISPENSO_INLINE BoolType_t<Flt> float_is_odd(Flt x) {
  if constexpr (std::is_same_v<Flt, float>) {
    return float_is_int(x) && !float_is_int(x * 0.5f);
  } else {
    return float_is_int(x) & !float_is_int(x * Flt(0.5f));
  }
}

// Fast integer division by 3 using multiply-and-shift. Used in cbrt magic constant computation.
DISPENSO_INLINE int32_t int_div_by_3(int32_t i) {
  return static_cast<int32_t>((uint64_t(i) * 0x55555556) >> 32);
}

// ---------------------------------------------------------------------------
// Polynomial evaluation: hornerEval, estrinEval, polyEval
// ---------------------------------------------------------------------------
//
// All take coefficients in HIGH-to-LOW order (highest degree first):
//   polyEval(x, cn, cn-1, ..., c1, c0) = cn*x^n + cn-1*x^(n-1) + ... + c1*x + c0
//
// This matches the natural Horner evaluation order and the existing hand-written
// FMA chains in fast_math. For odd/even polynomial forms like tanh(x) = x*(1+x²*Q(x²)),
// compose as: x * (1 + x2 * polyEval(x2, qn, ..., q1, q0)).

namespace detail {

// --- Horner evaluation ---
// Sequential FMA chain, optimal for low-ILP targets (GPU, scalar, narrow SIMD).
// Degree N polynomial = N FMAs.

template <typename Flt>
DISPENSO_INLINE Flt hornerImpl(Flt, Flt accum) {
  return accum;
}

template <typename Flt, typename... Cs>
DISPENSO_INLINE Flt hornerImpl(Flt x, Flt accum, Flt next, Cs... rest) {
  return hornerImpl(x, FloatTraits<Flt>::fma(accum, x, next), rest...);
}

} // namespace detail

// --- Estrin evaluation ---
// Tree-reduces paired coefficients at each level, cutting critical-path depth
// from N to ceil(log2(N+1)) at the cost of extra multiplies for x powers.
// Better for wide SIMD with high ILP (AVX, AVX-512).
//
// Generic peel-and-recurse via tuples. At each level:
//   1. Pair adjacent values: (a,b) → fma(a, xp, b)
//   2. Carry unpaired odd element
//   3. Recurse with xp² and the paired results
//
// All tuple operations (make_tuple, tuple_cat, apply) are eliminated by the
// optimizer — verified to produce identical assembly to hand-written FMA trees
// with clang 21, GCC 11, and GCC 15 at -O2.
//
// Not available under CUDA: std::apply is not __host__ __device__.
#if !defined(__CUDACC__)

namespace detail {

template <typename Flt>
struct EstrinImpl {
  // Base cases
  DISPENSO_INLINE static Flt reduce(Flt, std::tuple<Flt> done) {
    return std::get<0>(done);
  }

  DISPENSO_INLINE static Flt reduce(Flt xp, std::tuple<Flt, Flt> done) {
    return FloatTraits<Flt>::fma(std::get<0>(done), xp, std::get<1>(done));
  }

  // General: unpack tuple, pair all elements at this level, recurse
  template <typename... Done>
  DISPENSO_INLINE static Flt reduce(Flt xp, std::tuple<Done...> done) {
    return std::apply([xp](auto... ds) { return pairLevel(xp, std::tuple<>{}, ds...); }, done);
  }

  // Done pairing at this level — recurse with xp²
  template <typename... Paired>
  DISPENSO_INLINE static Flt pairLevel(Flt xp, std::tuple<Paired...> paired) {
    return reduce(xp * xp, paired);
  }

  // Odd leftover — carry forward unpaired
  template <typename... Paired>
  DISPENSO_INLINE static Flt pairLevel(Flt xp, std::tuple<Paired...> paired, Flt a) {
    return reduce(xp * xp, std::tuple_cat(paired, std::make_tuple(a)));
  }

  // Peel two from front, pair via FMA, accumulate into paired tuple
  template <typename... Paired, typename... Rest>
  DISPENSO_INLINE static Flt
  pairLevel(Flt xp, std::tuple<Paired...> paired, Flt a, Flt b, Rest... rest) {
    return pairLevel(
        xp, std::tuple_cat(paired, std::make_tuple(FloatTraits<Flt>::fma(a, xp, b))), rest...);
  }
};

} // namespace detail

// Estrin evaluation: same semantics as hornerEval, but uses tree-reduction
// for lower critical-path depth (ceil(log2(N)) vs N dependent FMAs).
// Coefficients HIGH-to-LOW.
template <typename Flt, typename C0, typename... Cs>
DISPENSO_INLINE Flt estrinEval(Flt x, C0 cn, Cs... rest) {
  return detail::EstrinImpl<Flt>::reduce(x, std::make_tuple(Flt(cn), Flt(rest)...));
}

#endif // !__CUDACC__

// Horner evaluation: hornerEval(x, cn, cn-1, ..., c0) = ((cn*x + cn-1)*x + ...)*x + c0
// Coefficients HIGH-to-LOW (highest degree first).
// Flt is deduced from x; coefficients are converted to Flt internally.
template <typename Flt, typename C0, typename... Cs>
DISPENSO_INLINE Flt hornerEval(Flt x, C0 cn, Cs... rest) {
  return detail::hornerImpl(x, Flt(cn), Flt(rest)...);
}

// Platform-adaptive polynomial evaluation.
// Uses Estrin on CPU (tree reduction for ILP on wide SIMD), Horner on GPU
// (sequential FMA chain, no wasted registers on in-order pipelines).
// Use polyEval only at call sites where Estrin has been measured beneficial.
// Use hornerEval directly for accuracy-sensitive polynomials where Estrin's
// different rounding order would regress ULP.
// Coefficients HIGH-to-LOW: polyEval(x, cn, ..., c0).
template <typename Flt, typename C0, typename... Cs>
DISPENSO_INLINE Flt polyEval(Flt x, C0 cn, Cs... rest) {
#if defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__)
  return hornerEval(x, cn, rest...);
#else
  return estrinEval(x, cn, rest...);
#endif
}

} // namespace fast_math
} // namespace dispenso

// Auto-detect SIMD backends and include FloatTraits specializations.
// Skip under CUDA compiler — SIMD vector types are not supported in device code.
#if !defined(__CUDACC__)

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

#endif // !__CUDACC__

namespace dispenso {
namespace fast_math {

// Best available SIMD float type for the current platform.
// Prefer native intrinsic wrappers over Highway for lower overhead.
// Highway is a fallback for platforms without a native wrapper.
// Under CUDA, only scalar float is available.
#if defined(__CUDACC__)
using DefaultSimdFloat = float;
#elif defined(__aarch64__)
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
