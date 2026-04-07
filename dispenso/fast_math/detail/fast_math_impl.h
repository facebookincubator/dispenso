/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include "../util.h"

namespace dispenso {
namespace fast_math {
namespace detail {

// Constants and algorithms come from "Fast Calculation of Cube and Inverse Cube Roots Using a Magic
// Constant and Its Implementation on Microcontrollers" by Moroz et al.
template <typename Flt, typename AccuracyTraits>
DISPENSO_INLINE Flt rcp_cbrt(Flt x, IntType_t<Flt> i) {
  auto fma = FloatTraits<Flt>::fma;
  if constexpr (AccuracyTraits::kMaxAccuracy) {
    // Algorithm 3 step 1 + FMA quadratic correction (2 ULP, division-free Halley).
    //
    // Step 1 uses Algorithm 3's magic constant and tuned Newton step (~10 bits).
    // Step 2 computes the FMA residual e = 1 - x*y^3 with single rounding (no
    // cancellation), then applies a Sollya-optimized quadratic correction:
    //   y *= 1 + e*(c1 + c2*e)
    // This achieves cubic convergence (bits triple) without Halley's division,
    // at the same op count as the previous Algorithm 5 but with better accuracy.
    //
    // Polynomial: fpminimax for (1-e)^(-1/3), constant=1, on [-0.005, 0.005].
    // Relative error ~1.08e-8 (~26.5 bits).
    constexpr float c1 = 0.3333354890346527099609375f;
    constexpr float c2 = 0.22222582995891571044921875f;
    i = 0x548c39cb - int_div_by_3(i);
    Flt y = bit_cast<Flt>(i);
    y = y * fma(-0.534850249f * x * y, y * y, 1.5015480449f);
    Flt e = fma(x * y, -y * y, 1.0f);
    y = y * fma(e, fma(Flt(c2), e, Flt(c1)), 1.0f);
    return y;
  } else {
    // Algorithm 3 from Moroz et al. (12 ULP).
    i = 0x548c39cb - int_div_by_3(i);
    Flt y = bit_cast<Flt>(i);
    y = y * fma(-0.534850249f * x * y, y * y, 1.5015480449f);
    y = y * fma(-0.3333333333f * x * y, y * y, 1.333333985f);
    return y;
  }
}

// TODO: We likely will never need a double precision version of this function, but note that this
// is definitely busted for double.
template <typename Flt>
DISPENSO_INLINE Flt frexpImpl(IntType_t<Flt> hx, IntType_t<Flt>* eptr) {
  using IntT = IntType_t<Flt>;

  IntT ix = 0x7fffffff & hx;

  auto nonzero = hx != 0;
  *eptr = FloatTraits<Flt>::conditional(nonzero, (ix >> 23) - 126, IntT(0));
  return bit_cast<Flt>((hx & 0x807fffff) | 0x3f000000);
}

template <typename Flt>
DISPENSO_INLINE Flt ldexpImpl(UintType_t<Flt> hx, IntType_t<Flt> e) {
  auto esgn = signofi<Flt>(e);
  e *= esgn;
  hx += esgn * (e << 23);
  return bit_cast<Flt>(hx);
}

template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt cos_pi_4(Flt x2) {
  // Approximate cosine on [-PI/4,+PI/4], max error 0.87444 ULP.
  // Polynomial in x²: c0 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸.
  return dispenso::fast_math::hornerEval(
      x2, 2.44677067e-5f, -1.38877297e-3f, 4.16666567e-2f, -5.00000000e-1f, 1.00000000e+0f);
}

template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt sin_pi_4(Flt x2, Flt x) {
  // Approximate sine on [-PI/4,+PI/4].
  // Sollya fpminimax(sin(x) - x, [|3,5,7,9|], [|SG...|], [-pi/4;pi/4], absolute).
  // Odd polynomial: sin(x) = x + x³ * P(x²).
  Flt s = dispenso::fast_math::hornerEval(
      x2, 2.78547373e-6f, -1.98483889e-4f, 8.33336823e-3f, -1.66666672e-1f);
  return FloatTraits<Flt>::fma(s, x * x2, x);
}

// Wider-domain sin/cos polynomials for pi-reduction (no quadrant blending needed).
// sin(x) on [-pi/2, pi/2], degree 11 odd polynomial: sin(x) = x * P(x^2)
// Sollya fpminimax(sin(x), [|1,3,5,7,9,11|], [|SG...|], [-pi/2;pi/2], absolute)
// Absolute error: 3.2e-10 (31.5 bits)
template <typename Flt>
DISPENSO_INLINE Flt sin_pi_2(Flt x2, Flt x) {
  // Odd polynomial: sin(x) = x + x³ * P(x²).
  Flt s = dispenso::fast_math::hornerEval(
      x2, -0x1.ab55a8p-26f, 0x1.725326p-19f, -0x1.a0205p-13f, 0x1.11112ep-7f, -0x1.555556p-3f);
  return FloatTraits<Flt>::fma(s, x * x2, x);
}

template <typename Flt>
DISPENSO_INLINE Flt
rangeReduceFloor(Flt& x, NonDeduced<Flt> rcpMod, NonDeduced<Flt> valHi, NonDeduced<Flt> valLo) {
  auto j = floor_small(x * rcpMod); // FloatTraits<Flt>::fma(x, rcpMod, kMagic) - kMagic;
  x = FloatTraits<Flt>::fma(j, -valHi, x);
  x = FloatTraits<Flt>::fma(j, -valLo, x);
  return j;
}

template <typename Flt>
DISPENSO_INLINE Flt
rangeReduce(Flt& x, NonDeduced<Flt> rcpMod, NonDeduced<Flt> valHi, NonDeduced<Flt> valLo) {
  const Flt kMagic = FloatTraits<Flt>::kMagic;
  auto j = FloatTraits<Flt>::fma(x, rcpMod, kMagic) - kMagic;
  x = FloatTraits<Flt>::fma(j, -valHi, x);
  x = FloatTraits<Flt>::fma(j, -valLo, x);
  return j;
}

template <typename Flt>
DISPENSO_INLINE Flt rangeReduce2(
    Flt& x,
    NonDeduced<Flt> rcpMod,
    NonDeduced<Flt> valHi,
    NonDeduced<Flt> valMed,
    NonDeduced<Flt> valLo) {
  const Flt kMagic = FloatTraits<Flt>::kMagic;
  auto j = FloatTraits<Flt>::fma(x, rcpMod, kMagic) - kMagic;
  x = FloatTraits<Flt>::fma(j, -valHi, x);
  x = FloatTraits<Flt>::fma(j, -valMed, x);
  x = FloatTraits<Flt>::fma(j, -valLo, x);
  return j;
}

template <typename Flt>
DISPENSO_INLINE Flt rangeReduce3(
    Flt& x,
    NonDeduced<Flt> rcpMod,
    NonDeduced<Flt> valHi,
    NonDeduced<Flt> valMedHi,
    NonDeduced<Flt> valMedLo,
    NonDeduced<Flt> valLo) {
  const Flt kMagic = FloatTraits<Flt>::kMagic;
  auto j = FloatTraits<Flt>::fma(x, rcpMod, kMagic) - kMagic;
  x = FloatTraits<Flt>::fma(j, -valHi, x);
  x = FloatTraits<Flt>::fma(j, -valMedHi, x);
  x = FloatTraits<Flt>::fma(j, -valMedLo, x);
  x = FloatTraits<Flt>::fma(j, -valLo, x);
  return j;
}

// https://stackoverflow.com/questions/63918873/approximating-cosine-on-0-pi-using-only-single-precision-floating-point

// Cosine has quadrant bias 1, sine has quadrant bias 0
template <typename Flt>
DISPENSO_INLINE Flt sincos_pi_impl(Flt x, Flt j, int phaseShift) {
  /* phase shift of pi/2 (one quadrant) for cosine */

  auto i = convert_to_int(j) + phaseShift;
  Flt r;
  if constexpr (std::is_same_v<Flt, float>) { // For scalars, explicit if/else is faster to ensure
    // we avoid computing both functions
    if (i & 1) {
      r = cos_pi_4(x * x);
    } else {
      r = sin_pi_4(x * x, x);
    }
  } else {
    r = FloatTraits<Flt>::conditional((i & 1) != 0, cos_pi_4(x * x), sin_pi_4(x * x, x));
  }

  return FloatTraits<Flt>::conditional((i & 2) != 0, -r, r);
}

// Combined sin+cos computation. Always evaluates both polynomials (needed for sincos).
// For SIMD types, evaluates both polynomials and blends per-lane.
template <typename Flt>
DISPENSO_INLINE void sincos_pi_impl_both(Flt x, Flt j, Flt* out_sin, Flt* out_cos) {
  auto i = convert_to_int(j);
  auto x2 = x * x;
  auto sv = sin_pi_4(x2, x);
  auto cv = cos_pi_4(x2);

  // For sin: quadrant i. For cos: quadrant i+1 (phase shift).
  auto sin_use_cos = (i & 1) != 0;
  auto cos_use_cos = ((i + 1) & 1) != 0;

  Flt sr = FloatTraits<Flt>::conditional(sin_use_cos, cv, sv);
  Flt cr = FloatTraits<Flt>::conditional(cos_use_cos, cv, sv);

  *out_sin = FloatTraits<Flt>::conditional((i & 2) != 0, -sr, sr);
  *out_cos = FloatTraits<Flt>::conditional(((i + 1) & 2) != 0, -cr, cr);
}

// Pi-reduction dispatch: each function evaluates only its own polynomial.
// j = round(x / pi), so x_reduced in [-pi/2, pi/2].
// sin(x) = (-1)^j * sin(x_reduced), cos(x) = (-1)^j * cos(x_reduced).
template <typename Flt>
DISPENSO_INLINE Flt sin_pi_reduction(Flt x, Flt j) {
  auto ji = convert_to_int(j);
  auto x2 = x * x;
  auto result = sin_pi_2(x2, x);
  return FloatTraits<Flt>::conditional((ji & 1) != 0, -result, result);
}

// Cosine via offset-pi reduction (Highway-style).
// q = 2*trunc(|x|/pi) + 1 maps cos zeros to x_r = 0. Then cos(x) = ±sin(x_r).
// No quadrant blending: one polynomial, sign flip only.
template <typename Flt>
DISPENSO_INLINE Flt cos_offset_pi_reduction(Flt x_r, IntType_t<Flt> qi) {
  auto x2 = x_r * x_r;
  auto result = sin_pi_2(x2, x_r);
  // (q & 2) == 0 → negate (cos starts positive at q=1).
  return FloatTraits<Flt>::conditional((qi & 2) != 0, result, -result);
}

template <typename Flt>
DISPENSO_INLINE Flt tan_pi_2_impl(Flt x, Flt j) {
  auto i = convert_to_int(j);

  auto x2 = x * x;

  auto cv = cos_pi_4(x2);
  auto sv = sin_pi_4(x2, x);

  auto flip = (i & 1) != 0;
  Flt r;
  if constexpr (std::is_same_v<Flt, float>) {
    if (flip) {
      r = cv / sv;
    } else {
      r = sv / cv;
    }
  } else {
    auto n = FloatTraits<Flt>::conditional(flip, cv, sv);
    auto d = FloatTraits<Flt>::conditional(flip, sv, cv);
    r = n / d;
  }

  auto neg = ((i ^ (i + 1)) & 2) != 0;
  r = FloatTraits<Flt>::conditional(neg, -r, r);

  return r;
}

template <typename Flt>
DISPENSO_INLINE Flt asin_0_pt5(Flt x) {
  // Polynomial constants from glibc asinf impl.
  // asin(x) = x + x * x² * P(x²).
  auto x2 = x * x;
  auto px =
      dispenso::fast_math::hornerEval(
          x2, 4.216630880e-2f, 2.417951451e-2f, 4.547037598e-2f, 7.495297643e-2f, 1.666675248e-1f) *
      x2;
  return x + x * px;
}

template <typename Flt>
DISPENSO_INLINE Flt asin_pt5_1(Flt x) {
  constexpr float kPi_2hi = 1.57079637050628662109375f;
  constexpr float kPi_2lo = -4.37113900018624283e-8f;

  Flt y = dispenso::fast_math::hornerEval(
      x,
      0x1.ee9b5ep-10f,
      -0x1.7a2022p-7f,
      0x1.21893cp-5f,
      -0x1.5085b4p-4f,
      0x1.b3f124p-3f,
      -0x1.92136p0f);

  // Use compensating sum for better rounding.
  return kPi_2hi + FloatTraits<Flt>::fma(FloatTraits<Flt>::sqrt(1.0f - x), y, kPi_2lo);
}

template <typename Flt, typename AccuracyTraits>
DISPENSO_INLINE std::tuple<IntType_t<Flt>, Flt, Flt> logarithmSep(Flt& x) {
  using IntT = IntType_t<Flt>;
  using UintT = UintType_t<Flt>;
  IntT xi;
  Flt i;
  Flt m;
  auto fma = FloatTraits<Flt>::fma;

  if constexpr (AccuracyTraits::kMaxAccuracy) {
    auto small = x < 1.175494351e-38f;
    x = FloatTraits<Flt>::conditional(small, x * 8388608.0f, x);
    i = FloatTraits<Flt>::conditional(small, Flt(-23.0f), Flt(0.0f));

    xi = bit_cast<IntT>(x);
    // Use unsigned arithmetic to avoid signed overflow UB on negative inputs.
    UintT xu = bit_cast<UintT>(x);
    UintT e = (xu - UintT(0x3f3504f3)) & UintT(0xff800000);
    m = bit_cast<Flt>(xu - e);
    i = fma(Flt(bit_cast<IntT>(e)), 1.19209290e-7f, i);
  } else {
    // Ignore denorms, degradation behavior not that bad
    xi = bit_cast<IntT>(x);
    UintT xu = bit_cast<UintT>(x);
    UintT e = (xu - UintT(0x3f3504f3)) & UintT(0xff800000);
    m = bit_cast<Flt>(xu - e);
    i = Flt(bit_cast<IntT>(e)) * 1.19209290e-7f;
  }
  return {xi, i, m};
}

template <typename Flt>
DISPENSO_INLINE Flt logarithmBounds(Flt x, Flt y, IntType_t<Flt> xi) {
  using IntT = IntType_t<Flt>;
  using UintT = UintType_t<Flt>;
  // The NaN/negative detection below requires unsigned comparison: negative float
  // bit patterns have the sign bit set, making them large unsigned values that
  // exceed 0x7f800000.  Scalar C++ does this implicitly (the literal 0xff8fffff
  // exceeds INT_MAX, so the expression promotes to unsigned), but SIMD integer
  // comparison intrinsics are always signed.  Using UintT explicitly makes both
  // paths correct and documents the intent.
  if constexpr (std::is_same_v<Flt, float>) {
    if (xi < 0x00800000 || xi >= 0x7f800000) {
      int orbits = bool_apply_or_zero<int>(x == 0.0f, 0xff800000);
      orbits |= bool_apply_or_zero<int>((xi & 0x7f800000) == 0x7f800000, xi);
      orbits |= bool_apply_or_zero<int>(
          (static_cast<uint32_t>(xi) & 0xff8fffffu) > 0x7f800000u, 0x7f8fffff);
      y = orbits ? bit_cast<float>(orbits) : y;
    }
  } else {
    IntT orbits = bool_apply_or_zero<IntT>(x == 0.0f, 0xff800000);
    orbits |= bool_apply_or_zero<IntT>((xi & 0x7f800000) == 0x7f800000, xi);
    orbits |= bool_apply_or_zero<IntT>(
        (bit_cast<UintT>(xi) & UintT(0xff8fffff)) > UintT(0x7f800000), 0x7f8fffff);
    // Use a proper mask (all-ones/all-zeros) for conditional, since blendv
    // only checks bit 31.  Raw orbits values like 0x7F800000 (+inf) have bit 31
    // clear and would be incorrectly treated as "false".
    auto hasBound = orbits != IntT(0);
    y = FloatTraits<Flt>::conditional(hasBound, bit_cast<Flt>(orbits), y);
  }
  return y;
}

template <typename Flt>
DISPENSO_INLINE Flt atan_poly(Flt x) {
  // Degree-10 polynomial: atan(x) ≈ x * P(x).
  // Horner (not Estrin): Estrin regresses atan ULP from 3 to 4 and is slower
  // on scalar due to register pressure from the degree-10 x-power tree.
  Flt y = dispenso::fast_math::hornerEval(
      x,
      0.02159089781343937f,
      -0.131040632724762f,
      0.3223767280578613f,
      -0.3689180314540863f,
      0.09664592146873474f,
      0.1743806153535843f,
      0.004009631462395191f,
      -0.3336564600467682f,
      9.442386726732366e-06f,
      1.0f);
  return y * x;
}

} // namespace detail
} // namespace fast_math
} // namespace dispenso
