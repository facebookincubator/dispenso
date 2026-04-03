/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * EXPERIMENTAL — This sublibrary is under active development.
 * The API is unstable and subject to breaking changes without notice.
 * Do not depend on it in production code.
 */

#pragma once

#include <array>

namespace dispenso {
namespace fast_math {

struct DefaultAccuracyTraits {
  // Check for out-of-range, NaN, Inf inputs and ensure outputs match std.  Note that functions may
  // still provide correct values out of bounds, but if this is false, they are not required to.
  static constexpr bool kBoundsValues = false;
  // Try to get the lowest possible errors to match std.
  static constexpr bool kMaxAccuracy = false;
};

struct MaxAccuracyTraits {
  // Check for out-of-range, NaN, Inf inputs and ensure outputs match std
  static constexpr bool kBoundsValues = true;
  // Try to get the lowest possible errors to match std.
  static constexpr bool kMaxAccuracy = true;
};

} // namespace fast_math
} // namespace dispenso

#include "detail/fast_math_impl.h"

namespace dispenso {
namespace fast_math {

// Compile-time check that Flt is a supported type: float, a SIMD float wrapper,
// or a raw intrinsic type that maps to one via SimdTypeFor.
template <typename Flt>
constexpr void assert_float_type() {
  // For raw intrinsics (__m128 etc.), SimdType_t maps to the wrapper (SseFloat etc.).
  // For wrappers and float, SimdType_t is identity. Check that FloatTraits exists
  // by verifying the kOne constant is present (only defined for float-based types).
  static_assert(
      FloatTraits<SimdType_t<Flt>>::kOne == 0x3f800000,
      "fast_math only supports float and float-based SIMD types (SseFloat, AvxFloat, etc.)");
}

/**
 * @brief Hardware-delegated square root.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored; result is always 0 ULP.
 * @param x Input value.
 * @return Square root of @p x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt sqrt(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return FloatTraits<SimdType_t<Flt>>::sqrt(SimdType_t<Flt>(x)).v;
  } else {
    return std::sqrt(x);
  }
}

/**
 * @brief Cube root approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 12 ULP, MaxAccuracy: 3 ULP.
 *   kBoundsValues: returns input for inf/NaN/zero.
 * @param x Input value (all float domain, including denormals).
 * @return Cube root of @p x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt cbrt(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    // Raw intrinsic type (__m128, Vec512<float>, etc.) — forward to wrapper.
    return cbrt<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar path: branchless for the normal case.
    // Denormals get prescale=2^24, postscale=1/256 (cmov).
    // Zero is zeroed via AND mask (no blend).
    uint32_t ui = bit_cast<uint32_t>(x);
    uint32_t usgn = ui & 0x80000000u;
    uint32_t uabs = ui & 0x7fffffffu;

    if constexpr (AccuracyTraits::kBoundsValues) {
      if (DISPENSO_EXPECT((uabs & 0x7f800000u) == 0x7f800000u, 0)) {
        return x; // inf/NaN
      }
      if (DISPENSO_EXPECT(uabs == 0u, 0)) {
        return x; // ±0 (preserves -0)
      }
    }

    // expZero covers both zero and denormal; zero is handled by AND mask below.
    bool expZero = (uabs & 0x7f800000u) == 0u;
    float prescale = expZero ? 16777216.0f : 1.0f;
    float postscale = expZero ? (1.0f / 256.0f) : 1.0f;

    float absX = bit_cast<float>(uabs) * prescale;
    int32_t iScaled = static_cast<int32_t>(bit_cast<uint32_t>(absX));
    float sgnx = bit_cast<float>(usgn | 0x3f800000u);
    float result = sgnx / detail::rcp_cbrt<float, AccuracyTraits>(absX, iScaled);
    result *= postscale;

    // Zero mask: AND zeros out result for zero input, OR sign bit back in
    // so that cbrt(-0) = -0.
    uint32_t zeroMask = 0u - static_cast<uint32_t>(uabs != 0u);
    return bit_cast<float>((bit_cast<uint32_t>(result) & zeroMask) | usgn);
  } else {
    // SIMD wrapper path: prescale/postscale multipliers + AND mask for zero.
    using IntT = IntType_t<Flt>;
    IntT i = bit_cast<IntT>(x);
    IntT isgn = i & 0x80000000;
    i &= 0x7fffffff;

    // expZero covers both zero and denormal.
    auto expZero = (i & 0x7f800000) == 0;

    Flt prescale = FloatTraits<Flt>::conditional(expZero, Flt(16777216.0f), Flt(1.0f));
    Flt postscale = FloatTraits<Flt>::conditional(expZero, Flt(1.0f / 256.0f), Flt(1.0f));

    Flt absX = bit_cast<Flt>(i) * prescale;
    IntT iScaled = bit_cast<IntT>(absX);

    Flt sgnx = bit_cast<Flt>(isgn | FloatTraits<Flt>::kOne);
    Flt result = sgnx / detail::rcp_cbrt<Flt, AccuracyTraits>(absX, iScaled);
    result = result * postscale;

    // Zero: AND mask zeroes result, OR sign bit back so cbrt(-0) = -0.
    auto zeroMask = bool_as_mask<IntT>(i != 0);
    result = bit_cast<Flt>((bit_cast<IntT>(result) & zeroMask) | isgn);

    if constexpr (AccuracyTraits::kBoundsValues) {
      return FloatTraits<Flt>::conditional(nonnormal<Flt>(iScaled), x, result);
    }
    return result;
  }
}

/**
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored; result is always bit-accurate.
 * @param x Input value.
 * @param eptr Pointer to receive the exponent.
 * @return Mantissa in [0.5, 1). Returns @p x unchanged for inf/NaN/zero.
 *   Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt frexp(Flt x, IntType_t<Flt>* eptr) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return frexp<SimdType_t<Flt>>(SimdType_t<Flt>(x), eptr).v;
  } else {
    using IntT = IntType_t<Flt>;
    IntT ix = bit_cast<IntType_t<Flt>>(x);
    *eptr = 0;
    return FloatTraits<Flt>::conditional(
        nonnormalOrZero<Flt>(ix), x, detail::frexpImpl<Flt>(ix, eptr));
  }
}

/**
 * @brief Multiply a float by a power of 2 (bit-accurate).
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored; result is always bit-accurate.
 * @param x Input value.
 * @param e Exponent to apply (x * 2^e).
 * @return @p x * 2^e. Returns @p x unchanged for inf/NaN.
 *   Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt ldexp(Flt x, IntType_t<Flt> e) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return ldexp<SimdType_t<Flt>>(SimdType_t<Flt>(x), e).v;
  } else {
    using IntT = IntType_t<Flt>;
    IntT hx = bit_cast<IntT>(x);

    return FloatTraits<Flt>::conditional(nonnormal<Flt>(hx), x, detail::ldexpImpl<Flt>(hx, e));
  }
}

/**
 * @brief Arc cosine approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored (uniform implementation); 4 ULP.
 * @param x Input value in [-1, 1].
 * @return Arc cosine of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt acos(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return acos<SimdType_t<Flt>>(SimdType_t<Flt>(x)).v;
  } else {
    BoolType_t<Flt> xneg = x < 0.0f;
    // abs
    UintType_t<Flt> xi = bit_cast<UintType_t<Flt>>(x);
    xi &= 0x7fffffff;
    x = bit_cast<Flt>(xi);

    static constexpr std::array<float, 8> ks = {
        1.570796251296997f,
        -0.2145989686250687f,
        0.08899597078561783f,
        -0.05029246956110001f,
        0.03122001513838768f,
        -0.0175294354557991f,
        0.006957028061151505f,
        -0.001334964530542493f};

    auto fma = FloatTraits<Flt>::fma;
    Flt y = ks[7];
    y = fma(y, x, ks[6]);
    y = fma(y, x, ks[5]);
    y = fma(y, x, ks[4]);
    y = fma(y, x, ks[3]);
    y = fma(y, x, ks[2]);
    y = fma(y, x, ks[1]);
    y = fma(y, x, ks[0]);

    auto sqrt1mx = FloatTraits<Flt>::sqrt(1.0f - x);
    return FloatTraits<Flt>::conditional(xneg, fma(y, -sqrt1mx, Flt(kPi)), y * sqrt1mx);
  }
}

/**
 * @brief Arc sine approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored (uniform implementation); 2 ULP.
 * @param x Input value in [-1, 1].
 * @return Arc sine of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt asin(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return asin<SimdType_t<Flt>>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;

    IntT xi = bit_cast<IntT>(x);
    IntT sgnbit = xi & 0x80000000;
    xi &= 0x7fffffff;
    x = bit_cast<Flt>(xi);

    Flt ret;
    if constexpr (std::is_same_v<Flt, float>) {
      if (xi > 0x3f000000) { // x > 0.5
        ret = detail::asin_pt5_1(x);

      } else {
        ret = detail::asin_0_pt5(x);
      }
    } else {
      // Use compensating sum for better rounding.
      auto y = detail::asin_pt5_1(x);
      auto z = detail::asin_0_pt5(x);
      ret = FloatTraits<Flt>::conditional(xi > 0x3f000000, y, z); // choose between correct estimate
    }
    return bit_cast<Flt>(sgnbit | bit_cast<IntT>(ret));
  }
}

/**
 * @brief Sine approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits kMaxAccuracy uses higher-precision range reduction.
 *   Default: 2 ULP in [-2^20 pi, 2^20 pi]; accuracy degrades for larger |x|.
 * @param x Input value in radians (all float domain).
 * @return Sine of @p x. Compatible with all SIMD backends.
 *
 * Scalar: pi/4 reduction with quadrant branching (lowest latency).
 * SIMD:   pi reduction, single sin polynomial, sign flip (no blending, ~30-40% faster).
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt sin(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return sin<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar: pi/4 reduction + branch selects sin or cos polynomial per quadrant.
    constexpr float kPi_2hi = 1.57079625f;
    constexpr float kPi_2med = 7.54978942e-08f;
    constexpr float kPi_2lo = 5.39032794e-15f;
    constexpr float k2_pi = 0.636619747f;
    Flt j = detail::rangeReduce2(x, k2_pi, kPi_2hi, kPi_2med, kPi_2lo);
    return detail::sincos_pi_impl(x, j, 0);
  } else {
    // SIMD: pi reduction — single sin_pi_2 polynomial, sign flip, no blending.
    constexpr float kPi_hi = 3.1415925f;
    constexpr float kPi_med = 1.50995788e-07f;
    constexpr float kPi_lo = 1.07806559e-14f;
    constexpr float k1_pi = 0.318309886f;
    Flt j = detail::rangeReduce2(x, k1_pi, kPi_hi, kPi_med, kPi_lo);
    return detail::sin_pi_reduction(x, j);
  }
}

/**
 * @brief Cosine approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits kMaxAccuracy uses 4-part Cody-Waite range reduction.
 *   Default: 2 ULP in [-2^15 pi, 2^15 pi].
 *   MaxAccuracy: 2 ULP in [-2^20 pi, 2^20 pi].
 * @param x Input value in radians (all float domain).
 * @return Cosine of @p x. Compatible with all SIMD backends.
 *
 * Scalar: pi/4 reduction with quadrant branching (lowest latency).
 * SIMD:   offset-pi reduction (q = 2*trunc(|x|/pi)+1), single sin polynomial,
 *         sign flip (no blending, ~30% faster).
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt cos(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return cos<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar: pi/4 reduction + branch selects sin or cos polynomial per quadrant.
    Flt j;
    if constexpr (AccuracyTraits::kMaxAccuracy) {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2medh = 7.54978942e-08f;
      constexpr float kPi_2medl = 5.39030253e-15f;
      constexpr float kPi_2lo = 3.28200367e-22f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce3(x, k2_pi, kPi_2hi, kPi_2medh, kPi_2medl, kPi_2lo);
    } else {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2med = 7.54978942e-08f;
      constexpr float kPi_2lo = 5.39032794e-15f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce2(x, k2_pi, kPi_2hi, kPi_2med, kPi_2lo);
    }
    return detail::sincos_pi_impl(x, j, 1);
  } else {
    // SIMD: offset-pi reduction maps cos zeros to x_r = 0, then evaluates sin_pi_2.
    auto fma = FloatTraits<Flt>::fma;
    Flt y = fabs(x);
    constexpr float k1_pi = 0.318309886f;
    auto qi = convert_to_int_trunc(y * k1_pi);
    qi = qi + qi + 1;
    Flt qf = Flt(qi);
    if constexpr (AccuracyTraits::kMaxAccuracy) {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2medh = 7.54978942e-08f;
      constexpr float kPi_2medl = 5.39030253e-15f;
      constexpr float kPi_2lo = 3.28200367e-22f;
      y = fma(qf, Flt(-kPi_2hi), y);
      y = fma(qf, Flt(-kPi_2medh), y);
      y = fma(qf, Flt(-kPi_2medl), y);
      y = fma(qf, Flt(-kPi_2lo), y);
    } else {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2med = 7.54978942e-08f;
      constexpr float kPi_2lo = 5.39032794e-15f;
      y = fma(qf, Flt(-kPi_2hi), y);
      y = fma(qf, Flt(-kPi_2med), y);
      y = fma(qf, Flt(-kPi_2lo), y);
    }
    return detail::cos_offset_pi_reduction(y, qi);
  }
}

/**
 * @brief Simultaneous sine and cosine, sharing range reduction.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Same behavior as sin/cos: kMaxAccuracy controls
 *   range reduction precision. Default: 2 ULP each.
 * @param x Input value in radians (all float domain).
 * @param[out] out_sin Pointer to receive sin(x).
 * @param[out] out_cos Pointer to receive cos(x).
 * @note Can be faster than separate sin() + cos() calls due to shared range
 *   reduction and polynomial evaluation (SIMD). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE void sincos(Flt x, Flt* out_sin, Flt* out_cos) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    SimdType_t<Flt> s, c;
    sincos<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x), &s, &c);
    *out_sin = s.v;
    *out_cos = c.v;
  } else {
    Flt j;
    if constexpr (AccuracyTraits::kMaxAccuracy) {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2medh = 7.54978942e-08f;
      constexpr float kPi_2medl = 5.39030253e-15f;
      constexpr float kPi_2lo = 3.28200367e-22f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce3(x, k2_pi, kPi_2hi, kPi_2medh, kPi_2medl, kPi_2lo);
    } else {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2med = 7.54978942e-08f;
      constexpr float kPi_2lo = 5.39032794e-15f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce2(x, k2_pi, kPi_2hi, kPi_2med, kPi_2lo);
    }
    detail::sincos_pi_impl_both(x, j, out_sin, out_cos);
  }
}

/**
 * @brief Compute sin(pi * x) with exact range reduction.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored (range reduction is always exact).
 * @param x Input value (all float domain). Exact at integers (returns 0) and
 *   half-integers (returns ±1). ~2 ULP for general inputs.
 * @return sin(pi * x). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt sinpi(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return sinpi<SimdType_t<Flt>>(SimdType_t<Flt>(x)).v;
  } else {
    // Exact range reduction: j = round(2*x), r = x - j/2, t = r * pi.
    constexpr float kMagic = FloatTraits<Flt>::kMagic; // 1.5 * 2^23
    Flt j = (2.0f * x + kMagic) - kMagic; // round(2*x)
    Flt r = x - j * 0.5f; // exact for |x| < 2^22
    Flt t = r * kPi; // only error source
    return detail::sincos_pi_impl(t, j, 0);
  }
}

/**
 * @brief Compute cos(pi * x) with exact range reduction.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored (range reduction is always exact).
 * @param x Input value (all float domain). Exact at integers (returns ±1) and
 *   half-integers (returns 0). ~2 ULP for general inputs.
 * @return cos(pi * x). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt cospi(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return cospi<SimdType_t<Flt>>(SimdType_t<Flt>(x)).v;
  } else {
    constexpr float kMagic = FloatTraits<Flt>::kMagic;
    Flt j = (2.0f * x + kMagic) - kMagic;
    Flt r = x - j * 0.5f;
    Flt t = r * kPi;
    return detail::sincos_pi_impl(t, j, 1);
  }
}

/**
 * @brief Simultaneous sin(pi*x) and cos(pi*x) with exact range reduction.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored.
 * @param x Input value (all float domain).
 * @param[out] out_sin Pointer to receive sin(pi * x).
 * @param[out] out_cos Pointer to receive cos(pi * x).
 * @note Faster than separate sinpi() + cospi() calls. ~2 ULP each.
 *   Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE void sincospi(Flt x, Flt* out_sin, Flt* out_cos) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    SimdType_t<Flt> s, c;
    sincospi<SimdType_t<Flt>>(SimdType_t<Flt>(x), &s, &c);
    *out_sin = s.v;
    *out_cos = c.v;
  } else {
    constexpr float kMagic = FloatTraits<Flt>::kMagic;
    Flt j = (2.0f * x + kMagic) - kMagic;
    Flt r = x - j * 0.5f;
    Flt t = r * kPi;
    detail::sincos_pi_impl_both(t, j, out_sin, out_cos);
  }
}

/**
 * @brief Tangent approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 3 ULP in [-256K pi, 256K pi].
 *   kMaxAccuracy uses 4-part Cody-Waite reduction for wider accurate range.
 * @param x Input value in radians; undefined at odd multiples of pi/2.
 * @return Tangent of @p x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt tan(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return tan<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    Flt j;
    if constexpr (AccuracyTraits::kMaxAccuracy) {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2medh = 7.54978942e-08f;
      constexpr float kPi_2medl = 5.39030253e-15f;
      constexpr float kPi_2lo = 3.28200367e-22f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce3(x, k2_pi, kPi_2hi, kPi_2medh, kPi_2medl, kPi_2lo);
    } else {
      constexpr float kPi_2hi = 1.57079625f;
      constexpr float kPi_2med = 7.54978942e-08f;
      constexpr float kPi_2lo = 5.39032794e-15f;
      constexpr float k2_pi = 0.636619747f;
      j = detail::rangeReduce2(x, k2_pi, kPi_2hi, kPi_2med, kPi_2lo);
    }
    return detail::tan_pi_2_impl(x, j);
  }
}

/**
 * @brief Arc tangent approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored (uniform implementation); 3 ULP.
 * @param x Input value (all float domain).
 * @return Arc tangent of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt atan(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return atan<SimdType_t<Flt>>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;

    IntT xi = bit_cast<IntT>(x);
    IntT sgnbit = xi & 0x80000000;
    xi &= 0x7fffffff;
    x = bit_cast<Flt>(xi);

    auto flip = x > 1.0f;
    // TODO(bbudge): Verify, Some compilers (MSVC) don't optimize this as well, check for scalar
    // case.
    if constexpr (std::is_same_v<Flt, float>) {
      if (flip) {
        x = 1.0f / x;
      }
    } else {
      x = FloatTraits<Flt>::conditional(flip, 1.0f / x, x);
    }

    auto y = detail::atan_poly(x);

    y = FloatTraits<Flt>::conditional(flip, kPi_2 - y, y);

    y = bit_cast<Flt>(bit_cast<IntT>(y) | sgnbit);

    return y;
  }
}

/**
 * @brief Base-2 exponential approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 1 ULP. kMaxAccuracy has no additional effect.
 *   kBoundsValues: returns NaN for NaN, clamps input to avoid overflow.
 * @param x Input value; [-127, 128] for normal output.
 * @return 2^x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt exp2(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return exp2<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;

    Flt xf;
    IntT xi;
    if constexpr (AccuracyTraits::kBoundsValues) {
      if constexpr (std::is_same_v<float, Flt>) {
        if (!(x == x)) {
          return x;
        }
        x = clamp_allow_nan(x, -127.0f, 128.0f);
        xf = floor_small(x);
        xi = convert_to_int(xf);
      } else {
        // bool_as_mask converts any comparison result (lane-wide or __mmask16)
        // to IntType uniformly across all SIMD backends.
        IntType_t<Flt> nonanmask = bool_as_mask<IntType_t<Flt>>(x == x);
        x = clamp_allow_nan(x, Flt(-127.0f), Flt(128.0f));
        xf = floor_small(x);
        xi = nonanmask & convert_to_int(xf);
      }
    } else {
      xf = floor_small(x);
      xi = convert_to_int(xf);
    }

    x -= xf;
    xi += 127;
    xi <<= 23;

    Flt powxi = bit_cast<Flt>(xi);

    auto fma = FloatTraits<Flt>::fma;

    constexpr std::array<float, 7> ks = {
        0x1p0f,
        0x1.62e42cp-1f,
        0x1.ebfd38p-3f,
        0x1.c68b2ep-5f,
        0x1.3cfb6p-7f,
        0x1.4748aep-10f,
        0x1.c41242p-13f};

    Flt y = ks[6];
    y = fma(y, x, ks[5]);
    y = fma(y, x, ks[4]);
    y = fma(y, x, ks[3]);
    y = fma(y, x, ks[2]);
    y = fma(y, x, ks[1]);
    y = fma(y, x, ks[0]);

    return powxi * y;
  }
}

/**
 * @brief Natural exponential approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 3 ULP. MaxAccuracy: 1 ULP (Sollya-optimized
 *   polynomial + Norbert Juffa's rounding technique).
 *   kBoundsValues: returns 0 for large negative, inf for large positive, NaN for NaN.
 * @param x Input value; [-89, 89] for normal output.
 * @return e^x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt exp(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return exp<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;
    using UintT = UintType_t<Flt>;
    auto fma = FloatTraits<Flt>::fma;
    constexpr float k1_ln2 = k1_Ln2;
    constexpr float kLn2hi = 6.93145752e-1f;
    constexpr float kLn2lo = 1.42860677e-6f;

    if constexpr (AccuracyTraits::kMaxAccuracy || AccuracyTraits::kBoundsValues) {
      // Rounding algorithm courtesy Norbert Juffa, https://stackoverflow.com/a/40519989/830441
      // exp(x) = 2**i * exp(f); i = rintf (x / log(2))
      Flt f = x;
      Flt j = detail::rangeReduce(f, k1_ln2, kLn2hi, kLn2lo);
      IntT i = convert_to_int(j);
      // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
      constexpr std::array<float, 7> ks = {
          1.f,
          1.f,
          0x1.fffff8p-2f,
          0x1.55548ep-3f,
          0x1.555b98p-5f,
          0x1.123bccp-7f,
          0x1.6850e4p-10f};
      Flt y = ks[6];
      y = fma(y, f, ks[5]);
      y = fma(y, f, ks[4]);
      y = fma(y, f, ks[3]);
      y = fma(y, f, ks[2]);
      y = fma(y, f, ks[1]);
      y = fma(y, f, ks[0]);
      // exp(x) = 2**i * y (with rounding)
      auto ipos = i > IntT(0);
      UintT ia = FloatTraits<Flt>::conditional(ipos, UintT(0u), UintT(0x83000000u));
      Flt s = bit_cast<Flt>(UintT(0x7f000000u) + ia);
      Flt t = bit_cast<Flt>((UintT(i) << 23) - ia);
      y = y * s;
      y = y * t;
      if constexpr (AccuracyTraits::kBoundsValues) {
        if constexpr (std::is_same_v<float, Flt>) {
          auto zero = (x < -89.0f);
          if ((zero || x > 89.0f)) {
            IntT xi = bit_cast<IntT>(x);
            auto inf = xi == 0x7f800000;
            Flt orbits = bit_cast<Flt>(bool_apply_or_zero(inf, 0x7f800000));
            y = FloatTraits<Flt>::conditional(zero | inf, orbits, y);
          }
        } else {
          IntT xi = bit_cast<IntT>(x);
          auto zero = (x < -89.0f);
          auto inf = xi == IntT(0x7f800000);
          IntT orbits = bool_apply_or_zero<IntT>(inf, IntT(0x7f800000));
          auto mask = bool_as_mask<IntT>(zero) | bool_as_mask<IntT>(inf);
          y = FloatTraits<Flt>::conditional(mask, bit_cast<Flt>(orbits), y);
        }
      }
      return y;
    } else {
      Flt j = detail::rangeReduceFloor(x, k1_ln2, kLn2hi, kLn2lo);
      IntT xi = convert_to_int(j);

      xi += 127;
      xi <<= 23;

      Flt powxi = bit_cast<Flt>(xi);

      // approximate r = exp(f) on interval [0, ln2]
      constexpr std::array<float, 6> ks = {
          0x1.fffffep-1f,
          0x1.000086p0f,
          0x1.ffd96ep-2f,
          0x1.57481ep-3f,
          0x1.3f3846p-5f,
          0x1.8014dp-7f};
      Flt y = ks[5];
      y = fma(y, x, ks[4]);
      y = fma(y, x, ks[3]);
      y = fma(y, x, ks[2]);
      y = fma(y, x, ks[1]);
      y = fma(y, x, ks[0]);

      return powxi * y;
    }
  }
}

/**
 * @brief Base-10 exponential approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 2 ULP. kBoundsValues: handles NaN.
 * @param x Input value; [-38, 38] for normal output.
 * @return 10^x. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt exp10(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return exp10<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;
    constexpr double kLog10of2d = 0.3010299956639812;
    constexpr float k1_log10of2 = static_cast<float>(1.0 / kLog10of2d);
    constexpr float kLog10of2 = static_cast<float>(kLog10of2d);
    constexpr float kLog10of2lo = static_cast<float>(kLog10of2d - kLog10of2);

    Flt j;
    IntT xi;

    if constexpr (AccuracyTraits::kBoundsValues) {
      IntType_t<Flt> nonanmask = bool_as_mask<IntType_t<Flt>>(x == x);

      x = clamp_allow_nan(x, Flt(-38.23080944932561f), Flt(38.53183944498959f));
      j = detail::rangeReduceFloor(x, k1_log10of2, kLog10of2, kLog10of2lo);
      xi = nonanmask & convert_to_int(j);
    } else {
      j = detail::rangeReduceFloor(x, k1_log10of2, kLog10of2, kLog10of2lo);
      xi = convert_to_int(j);
    }

    xi += 127;
    xi <<= 23;

    Flt powxi = bit_cast<Flt>(xi);
    auto fma = FloatTraits<Flt>::fma;
    constexpr std::array<float, 7> ks = {
        0x1p0f,
        0x1.26bb18p1f,
        0x1.53534cp1f,
        0x1.0459f6p1f,
        0x1.2d9da2p0f,
        0x1.024feap-1f,
        0x1.293a54p-2f};
    Flt y = ks[6];
    y = fma(y, x, ks[5]);
    y = fma(y, x, ks[4]);
    y = fma(y, x, ks[3]);
    y = fma(y, x, ks[2]);
    y = fma(y, x, ks[1]);
    y = fma(y, x, ks[0]);

    return powxi * y;
  }
}

/**
 * @brief Base-2 logarithm approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 1 ULP. kMaxAccuracy handles denormals correctly.
 *   kBoundsValues: returns -inf for 0, NaN for negative, +inf for +inf.
 * @param x Input value in (0, +inf).
 * @return log2(x). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt log2(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return log2<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    auto [xi, i, m] = detail::logarithmSep<Flt, AccuracyTraits>(x);

    m = m - 1.0f;

    // Compute log2(1+m) for m in [sqrt(0.5)-1, sqrt(2.0)-1]
    Flt y = -1.09985352e-1f;
    auto fma = FloatTraits<Flt>::fma;
    y = fma(y, m, 1.86182275e-1f);
    y = fma(y, m, -1.91066533e-1f);
    y = fma(y, m, 2.04593703e-1f);
    y = fma(y, m, -2.39627063e-1f);
    y = fma(y, m, 2.88573444e-1f);
    y = fma(y, m, -3.60695332e-1f);
    y = fma(y, m, 4.80897635e-1f);
    y = fma(y, m, -7.21347392e-1f);
    y = fma(y, m, 4.42695051e-1f);
    y = fma(y, m, m); // simplify due to constants 0 and 1

    y = y + i;

    /* Check for and handle special cases */
    if constexpr (AccuracyTraits::kBoundsValues) {
      y = detail::logarithmBounds(x, y, xi);
    }

    return y;
  }
}

/**
 * @brief Natural logarithm approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 2 ULP. kMaxAccuracy handles denormals correctly.
 *   kBoundsValues: returns -inf for 0, NaN for negative, +inf for +inf.
 * @param x Input value in (0, +inf).
 * @return ln(x). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt log(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return log<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    auto [xi, i, m] = detail::logarithmSep<Flt, AccuracyTraits>(x);

    m = m - 1.0f;

    // Compute log(1+m) for m in [sqrt(0.5)-1, sqrt(2.0)-1]
    auto fma = FloatTraits<Flt>::fma;
    constexpr std::array<float, 10> ks = {
        0.f,
        1.f,
        -0.499999911f,
        0.333337069f,
        -0.250024557f,
        0.199700251f,
        -0.165455937f,
        0.148145974f,
        -0.14482744f,
        0.0924733654f};
    Flt y = ks[9];
    y = fma(y, m, ks[8]);
    y = fma(y, m, ks[7]);
    y = fma(y, m, ks[6]);
    y = fma(y, m, ks[5]);
    y = fma(y, m, ks[4]);
    y = fma(y, m, ks[3]);
    y = fma(y, m, ks[2]);
    y = fma(y, m * m, m); // simplify due to constants 0 and 1

    y = fma(i, kLn2, y);

    /* Check for and handle special cases */
    if constexpr (AccuracyTraits::kBoundsValues) {
      y = detail::logarithmBounds(x, y, xi);
    }

    return y;
  }
}

/**
 * @brief Base-10 logarithm approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 3 ULP. kMaxAccuracy handles denormals correctly.
 *   kBoundsValues: returns -inf for 0, NaN for negative, +inf for +inf.
 * @param x Input value in (0, +inf).
 * @return log10(x). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt log10(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return log10<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    auto [xi, i, m] = detail::logarithmSep<Flt, AccuracyTraits>(x);

    m = m - 1.0f;

    // Compute log10(1+m) for m in [sqrt(0.5)-1, sqrt(2.0)-1]
    auto fma = FloatTraits<Flt>::fma;
    constexpr std::array<float, 10> ks = {
        0.f,
        0.434294462f,
        -0.217147425f,
        0.144770443f,
        -0.108560629f,
        0.086600922f,
        -0.0723559037f,
        0.0659840927f,
        -0.0601400957f,
        0.0326662175f};
    Flt y = ks[9];
    y = fma(y, m, ks[8]);
    y = fma(y, m, ks[7]);
    y = fma(y, m, ks[6]);
    y = fma(y, m, ks[5]);
    y = fma(y, m, ks[4]);
    y = fma(y, m, ks[3]);
    y = fma(y, m, ks[2]);
    y = fma(y, m, ks[1]);
    y = fma(y, m, ks[0]);

    constexpr float kLog10of2 = 0.30102999566398f;
    y = fma(i, kLog10of2, y);

    /* Check for and handle special cases */
    if constexpr (AccuracyTraits::kBoundsValues) {
      y = detail::logarithmBounds(x, y, xi);
    }

    return y;
  }
}

/**
 * @brief Two-argument arc tangent approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 3 ULP. kMaxAccuracy has no additional effect.
 *   kBoundsValues: handles the case where both arguments are inf.
 * @param y Y coordinate (all float domain).
 * @param x X coordinate (all float domain).
 * @return atan2(y, x) in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt atan2(Flt y, Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return atan2<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(y), SimdType_t<Flt>(x)).v;
  } else {
    using UintT = UintType_t<Flt>;
    UintT yi = bit_cast<UintT>(y);
    UintT xi = bit_cast<UintT>(x);

    auto someNonzero = ((yi | xi) & 0x7fffffff) != 0;

    UintT xsgn = xi & 0x80000000;
    UintT ysgn = yi & 0x80000000;
    UintT allsgn = xsgn ^ ysgn;

    yi &= 0x7fffffff;
    xi &= 0x7fffffff;

    auto flip = yi > xi;

    if constexpr (AccuracyTraits::kBoundsValues) {
      using IntT = IntType_t<Flt>;
      auto yinf = yi == UintT(0x7f800000u);
      auto bothinf = (yinf & (xi == UintT(0x7f800000u)));
      auto bothinfi = bool_as_mask<IntT>(bothinf);
      yi = FloatTraits<Flt>::conditional(bothinfi, UintT(0x7f7fffffu), yi);
      xi = FloatTraits<Flt>::conditional(bothinfi, UintT(0x7f7fffffu), xi);
      // keepsgn = NOT(yinf AND NOT bothinf) = bothinf OR NOT yinf.
      auto keepsgn = !((!bothinf) & yinf);
      xsgn &= bool_as_mask<UintT>(keepsgn);
    }

    Flt den = bit_cast<Flt>(FloatTraits<Flt>::conditional(flip, yi, xi));
    Flt num = bit_cast<Flt>(FloatTraits<Flt>::conditional(flip, xi, yi));

    Flt y_x = bool_apply_or_zero(someNonzero, num / den);

    auto z = detail::atan_poly(y_x);

    z = FloatTraits<Flt>::conditional(flip, kPi_2 - z, z);

    z = bit_cast<Flt>(bit_cast<UintT>(z) | allsgn);

    // Branchless offset: π where x<0, 0 where x≥0, with y's sign applied.
    UintT x_neg_mask = UintT(0) - (xsgn >> 31);
    Flt pi_or_zero = bit_cast<Flt>(x_neg_mask & bit_cast<UintT>(Flt(kPi)));
    Flt offset = bit_cast<Flt>(bit_cast<UintT>(pi_or_zero) | ysgn);
    return z + offset;
  }
}

/**
 * @brief Hypotenuse: sqrt(x*x + y*y) with overflow protection.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~1 ULP for all normal floats.
 *   kBoundsValues: handles inf/NaN per IEEE 754 (hypot(inf,NaN) = +inf).
 *   kMaxAccuracy: accepted but has no additional effect.
 * @param x First input (all float domain).
 * @param y Second input (all float domain).
 * @return sqrt(x*x + y*y). Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt hypot(Flt x, Flt y) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return hypot<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x), SimdType_t<Flt>(y)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    if constexpr (AccuracyTraits::kBoundsValues) {
      uint32_t ax = bit_cast<uint32_t>(x) & 0x7fffffffu;
      uint32_t ay = bit_cast<uint32_t>(y) & 0x7fffffffu;
      if (DISPENSO_EXPECT(ax == 0x7f800000u || ay == 0x7f800000u, 0)) {
        return std::numeric_limits<float>::infinity();
      }
      if (DISPENSO_EXPECT(ax > 0x7f800000u || ay > 0x7f800000u, 0)) {
        return std::numeric_limits<float>::quiet_NaN();
      }
    }
    // Scalar: cast to double for correctly-rounded result.
    double xd = static_cast<double>(x);
    double yd = static_cast<double>(y);
    return static_cast<float>(std::sqrt(std::fma(xd, xd, yd * yd)));
  } else {
    // SIMD: dynamically scale inputs based on their exponent to keep
    // intermediates in normal float range.  Extracts the exponent of
    // max(|x|,|y|), builds scale = 2^(127-E) so the scaled max is in [1,2).
    // Clamp ensures the scale itself stays in normal range.
    using IntT = IntType_t<Flt>;
    using UintT = UintType_t<Flt>;
    Flt abs_mask = bit_cast<Flt>(UintT(0x7fffffffu));
    Flt max_abs = FloatTraits<Flt>::max(x & abs_mask, y & abs_mask);
    // Clamp to [FLT_MIN, 2^126] so scale and unscale stay normal.
    Flt clamped =
        FloatTraits<Flt>::max(FloatTraits<Flt>::min(max_abs, Flt(0x1p126f)), Flt(0x1p-126f));
    IntT clamped_exp = bit_cast<IntT>(clamped) & IntT(0x7f800000);
    // scale = 2^(127-E), unscale = 2^(E-127).
    Flt scale = bit_cast<Flt>(IntT(0x7f000000) - clamped_exp);
    Flt xs = x * scale;
    Flt ys = y * scale;
    auto fma = FloatTraits<Flt>::fma;
    Flt h = FloatTraits<Flt>::sqrt(fma(xs, xs, ys * ys));
    // Newton refinement: h' = h + (xs²+ys²-h²) * rcp(2h)
    Flt r = fma(xs, xs, fma(ys, ys, -(h * h)));
    Flt rcp_2h = FloatTraits<Flt>::rcp(h + h + Flt(std::numeric_limits<float>::min()));
    h = fma(r, rcp_2h, h);
    Flt unscale = bit_cast<Flt>(clamped_exp);
    h = h * unscale;
    if constexpr (AccuracyTraits::kBoundsValues) {
      // IEEE 754: hypot(±inf, y) = +inf even when y is NaN.
      // Inf inputs produce NaN through Newton refinement, so override here.
      UintT ax = bit_cast<UintT>(x) & UintT(0x7fffffffu);
      UintT ay = bit_cast<UintT>(y) & UintT(0x7fffffffu);
      auto either_inf = (ax == UintT(0x7f800000u)) | (ay == UintT(0x7f800000u));
      auto inf_mask = bool_as_mask<IntT>(either_inf);
      h = FloatTraits<Flt>::conditional(inf_mask, bit_cast<Flt>(UintT(0x7f800000u)), h);
    }
    return h;
  }
}

} // namespace fast_math
} // namespace dispenso
