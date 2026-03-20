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
    // Algorithm 5 3 ulps
    constexpr float k1 = 1.752319676f;
    constexpr float k2 = 1.2509524245f;
    constexpr float k3 = 0.5093818292f;
    i = 0x548c2b4b - int_div_by_3(i);
    Flt y = bit_cast<Flt>(i);
    Flt c = x * y * y * y;
    y = y * fma(-c, fma(c, -k3, k2), k1);
    c = fma(x * y, -y * y, 1.0f);
    y = y * fma(c, 0.333333333333f, 1.0f);
    return y;
  } else {
    // Algorithm 3 12 ulps,
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
  auto fma = FloatTraits<Flt>::fma;
  /* Approximate cosine on [-PI/4,+PI/4] with maximum error of 0.87444 ulp */
  Flt c = 2.44677067e-5f; //  0x1.9a8000p-16
  c = fma(c, x2, -1.38877297e-3f); // -0x1.6c0efap-10
  c = fma(c, x2, 4.16666567e-2f); //  0x1.555550p-5
  c = fma(c, x2, -5.00000000e-1f); // -0x1.000000p-1
  c = fma(c, x2, 1.00000000e+0f); //  1.00000000p+0
  return c;
}

template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt sin_pi_4(Flt x2, Flt x) {
  auto fma = FloatTraits<Flt>::fma;
  /* Approximate sine on [-PI/4,+PI/4] */
  /* Sollya fpminimax(sin(x) - x, [|3,5,7,9|], [|SG...|], [-pi/4;pi/4], absolute) */
  Flt s = 2.78547373e-6f;
  s = fma(s, x2, -1.98483889e-4f);
  s = fma(s, x2, 8.33336823e-3f);
  s = fma(s, x2, -1.66666672e-1f);
  auto t = x * x2;
  s = fma(s, t, x);
  return s;
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
  // Polynomial constants from glibc asinf impl
  constexpr std::array<float, 5> ks = {
      1.666675248e-1f, 7.495297643e-2f, 4.547037598e-2f, 2.417951451e-2f, 4.216630880e-2f};
  auto x2 = x * x;
  auto px = ((((ks[4] * x2 + ks[3]) * x2 + ks[2]) * x2 + ks[1]) * x2 + ks[0]) * x2;
  return x + x * px;
}

template <typename Flt>
DISPENSO_INLINE Flt asin_pt5_1(Flt x) {
  auto fma = FloatTraits<Flt>::fma;
  constexpr float kPi_2hi = 1.57079637050628662109375f;
  constexpr float kPi_2lo = -4.37113900018624283e-8f;
  constexpr std::array<float, 5> ks = {
      -1.570233464f, 0.21018889546f, -0.07465200126f, 0.024976193904f, -0.0044932682067f};

  Flt y = ks[4];
  y = fma(y, x, ks[3]);
  y = fma(y, x, ks[2]);
  y = fma(y, x, ks[1]);
  y = fma(y, x, ks[0]);

  // Use compensating sum for better rounding.
  return kPi_2hi + fma(FloatTraits<Flt>::sqrt(1.0f - x), y, kPi_2lo);
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
  constexpr std::array<float, 11> ks = {
      0,
      1,
      9.442386726732366e-06f,
      -0.3336564600467682f,
      0.004009631462395191f,
      0.1743806153535843f,
      0.09664592146873474f,
      -0.3689180314540863f,
      0.3223767280578613f,
      -0.131040632724762f,
      0.02159089781343937f};
  auto fma = FloatTraits<Flt>::fma;
  Flt y = ks[10];
  y = fma(y, x, ks[9]);
  y = fma(y, x, ks[8]);
  y = fma(y, x, ks[7]);
  y = fma(y, x, ks[6]);
  y = fma(y, x, ks[5]);
  y = fma(y, x, ks[4]);
  y = fma(y, x, ks[3]);
  y = fma(y, x, ks[2]);
  y = fma(y, x, ks[1]);
  return y * x;
}

} // namespace detail
} // namespace fast_math
} // namespace dispenso
