/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__AVX512F__)

#include <immintrin.h>

#include <cstdint>

#include "float_traits.h"
#include "util.h"

namespace dispenso {
namespace fast_math {

struct Avx512Int32;
struct Avx512Uint32;

// 16-bit mask wrapping __mmask16 for AVX-512 predicated operations.
// Supports comparison with int literals (mask == 0 → all-false check),
// logical ops, and implicit conversion to Avx512Int32 for template code
// that assigns BoolType to IntType.
struct Avx512Mask {
  __mmask16 m;

  Avx512Mask() = default;
  Avx512Mask(__mmask16 mask) : m(mask) {}
  // Construct from int literal: 0 → all-false, non-zero → all-true.
  // Used by template code like `(expr) == 0` where expr is BoolType.
  Avx512Mask(int val) : m(val ? 0xFFFF : 0) {}

  // Logical NOT.
  friend Avx512Mask operator!(Avx512Mask a) {
    return static_cast<__mmask16>(~a.m);
  }

  // Logical AND/OR/XOR.
  friend Avx512Mask operator&(Avx512Mask a, Avx512Mask b) {
    return static_cast<__mmask16>(a.m & b.m);
  }
  friend Avx512Mask operator|(Avx512Mask a, Avx512Mask b) {
    return static_cast<__mmask16>(a.m | b.m);
  }
  friend Avx512Mask operator^(Avx512Mask a, Avx512Mask b) {
    return static_cast<__mmask16>(a.m ^ b.m);
  }

  Avx512Mask& operator&=(Avx512Mask o) {
    m &= o.m;
    return *this;
  }
  Avx512Mask& operator|=(Avx512Mask o) {
    m |= o.m;
    return *this;
  }

  // Equality: per-bit XNOR. mask == Avx512Mask(0) is equivalent to !mask.
  friend Avx512Mask operator==(Avx512Mask a, Avx512Mask b) {
    return static_cast<__mmask16>(~(a.m ^ b.m));
  }
  friend Avx512Mask operator!=(Avx512Mask a, Avx512Mask b) {
    return static_cast<__mmask16>(a.m ^ b.m);
  }

  // Implicit conversion to Avx512Int32 (lane-wide mask: all-ones or all-zeros).
  // Used by template code that assigns BoolType to IntType_t<Flt>.
  inline operator Avx512Int32() const;
};

// 16-wide float SIMD wrapper around __m512.
struct Avx512Float {
  __m512 v;

  Avx512Float() = default;
  Avx512Float(__m512 vec) : v(vec) {}
  // Implicit broadcast from scalar — required for polynomial constant propagation.
  Avx512Float(float f) : v(_mm512_set1_ps(f)) {}
  // Explicit int→float conversion for (Flt)intExpr patterns.
  explicit Avx512Float(Avx512Int32 i);

  // Implicit conversion back to raw intrinsic type.
  operator __m512() const {
    return v;
  }

  Avx512Float operator-() const {
    // _mm512_xor_ps requires AVX-512 DQ; use integer XOR on sign bit instead.
    return _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(v), _mm512_set1_epi32(int32_t(0x80000000))));
  }

  Avx512Float& operator+=(Avx512Float o) {
    v = _mm512_add_ps(v, o.v);
    return *this;
  }
  Avx512Float& operator-=(Avx512Float o) {
    v = _mm512_sub_ps(v, o.v);
    return *this;
  }
  Avx512Float& operator*=(Avx512Float o) {
    v = _mm512_mul_ps(v, o.v);
    return *this;
  }
  Avx512Float& operator&=(Avx512Float o) {
    // _mm512_and_ps requires AVX-512 DQ; use integer AND instead.
    v = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v), _mm512_castps_si512(o.v)));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend Avx512Float operator+(Avx512Float a, Avx512Float b) {
    return _mm512_add_ps(a.v, b.v);
  }
  friend Avx512Float operator-(Avx512Float a, Avx512Float b) {
    return _mm512_sub_ps(a.v, b.v);
  }
  friend Avx512Float operator*(Avx512Float a, Avx512Float b) {
    return _mm512_mul_ps(a.v, b.v);
  }
  friend Avx512Float operator/(Avx512Float a, Avx512Float b) {
    return _mm512_div_ps(a.v, b.v);
  }

  // Comparisons return Avx512Mask (native __mmask16).
  friend Avx512Mask operator<(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
  }
  friend Avx512Mask operator>(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
  }
  friend Avx512Mask operator<=(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
  }
  friend Avx512Mask operator>=(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
  }
  friend Avx512Mask operator==(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
  }
  friend Avx512Mask operator!=(Avx512Float a, Avx512Float b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_UQ);
  }

  // Bitwise ops on float values (for sign manipulation, etc.).
  // _mm512_{and,or,xor}_ps require AVX-512 DQ; use integer ops + casts.
  friend Avx512Float operator&(Avx512Float a, Avx512Float b) {
    return _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
  }
  friend Avx512Float operator|(Avx512Float a, Avx512Float b) {
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
  }
  friend Avx512Float operator^(Avx512Float a, Avx512Float b) {
    return _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
  }
};

// 16-wide int32 SIMD wrapper around __m512i.
struct Avx512Int32 {
  __m512i v;

  Avx512Int32() = default;
  Avx512Int32(__m512i vec) : v(vec) {}
  Avx512Int32(int32_t i) : v(_mm512_set1_epi32(i)) {}
  // Reinterpret from Avx512Uint32 (no-op on __m512i).
  Avx512Int32(Avx512Uint32 u);

  // Implicit conversion back to raw intrinsic type.
  operator __m512i() const {
    return v;
  }

  Avx512Int32 operator-() const {
    return _mm512_sub_epi32(_mm512_setzero_si512(), v);
  }

  Avx512Int32& operator+=(Avx512Int32 o) {
    v = _mm512_add_epi32(v, o.v);
    return *this;
  }
  Avx512Int32& operator-=(Avx512Int32 o) {
    v = _mm512_sub_epi32(v, o.v);
    return *this;
  }
  Avx512Int32& operator*=(Avx512Int32 o) {
    v = _mm512_mullo_epi32(v, o.v);
    return *this;
  }
  Avx512Int32& operator&=(Avx512Int32 o) {
    v = _mm512_and_si512(v, o.v);
    return *this;
  }
  Avx512Int32& operator|=(Avx512Int32 o) {
    v = _mm512_or_si512(v, o.v);
    return *this;
  }

  // Shifts.
  Avx512Int32 operator<<(int n) const {
    return _mm512_slli_epi32(v, n);
  }
  Avx512Int32 operator>>(int n) const {
    return _mm512_srai_epi32(v, n); // Arithmetic shift right
  }
  Avx512Int32& operator<<=(int n) {
    v = _mm512_slli_epi32(v, n);
    return *this;
  }
  Avx512Int32& operator>>=(int n) {
    v = _mm512_srai_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend Avx512Int32 operator+(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_add_epi32(a.v, b.v);
  }
  friend Avx512Int32 operator-(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_sub_epi32(a.v, b.v);
  }
  friend Avx512Int32 operator*(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_mullo_epi32(a.v, b.v);
  }

  // Bitwise.
  friend Avx512Int32 operator&(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_and_si512(a.v, b.v);
  }
  friend Avx512Int32 operator|(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_or_si512(a.v, b.v);
  }
  friend Avx512Int32 operator^(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_xor_si512(a.v, b.v);
  }

  friend Avx512Int32 operator~(Avx512Int32 a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }

  // Comparisons return Avx512Mask (native __mmask16).
  friend Avx512Mask operator==(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_cmpeq_epi32_mask(a.v, b.v);
  }
  friend Avx512Mask operator!=(Avx512Int32 a, Avx512Int32 b) {
    return static_cast<__mmask16>(~_mm512_cmpeq_epi32_mask(a.v, b.v));
  }
  friend Avx512Mask operator<(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_cmplt_epi32_mask(a.v, b.v);
  }
  friend Avx512Mask operator>(Avx512Int32 a, Avx512Int32 b) {
    return _mm512_cmpgt_epi32_mask(a.v, b.v);
  }

  friend Avx512Mask operator!(Avx512Int32 a) {
    return _mm512_cmpeq_epi32_mask(a.v, _mm512_setzero_si512());
  }
};

// 16-wide uint32 SIMD wrapper around __m512i.
struct Avx512Uint32 {
  __m512i v;

  Avx512Uint32() = default;
  Avx512Uint32(__m512i vec) : v(vec) {}
  Avx512Uint32(uint32_t u) : v(_mm512_set1_epi32(static_cast<int32_t>(u))) {}
  // Reinterpret from Avx512Int32 (no-op on __m512i).
  Avx512Uint32(Avx512Int32 i) : v(i.v) {}

  // Implicit conversion back to raw intrinsic type.
  operator __m512i() const {
    return v;
  }

  Avx512Uint32& operator+=(Avx512Uint32 o) {
    v = _mm512_add_epi32(v, o.v);
    return *this;
  }
  Avx512Uint32& operator-=(Avx512Uint32 o) {
    v = _mm512_sub_epi32(v, o.v);
    return *this;
  }
  Avx512Uint32& operator*=(Avx512Uint32 o) {
    v = _mm512_mullo_epi32(v, o.v);
    return *this;
  }
  Avx512Uint32& operator&=(Avx512Uint32 o) {
    v = _mm512_and_si512(v, o.v);
    return *this;
  }
  Avx512Uint32& operator|=(Avx512Uint32 o) {
    v = _mm512_or_si512(v, o.v);
    return *this;
  }

  // Shifts.
  Avx512Uint32 operator<<(int n) const {
    return _mm512_slli_epi32(v, n);
  }
  Avx512Uint32 operator>>(int n) const {
    return _mm512_srli_epi32(v, n); // Logical shift right
  }
  Avx512Uint32& operator<<=(int n) {
    v = _mm512_slli_epi32(v, n);
    return *this;
  }
  Avx512Uint32& operator>>=(int n) {
    v = _mm512_srli_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend Avx512Uint32 operator+(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_add_epi32(a.v, b.v);
  }
  friend Avx512Uint32 operator-(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_sub_epi32(a.v, b.v);
  }
  friend Avx512Uint32 operator*(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_mullo_epi32(a.v, b.v);
  }

  // Bitwise.
  friend Avx512Uint32 operator&(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_and_si512(a.v, b.v);
  }
  friend Avx512Uint32 operator|(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_or_si512(a.v, b.v);
  }
  friend Avx512Uint32 operator^(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_xor_si512(a.v, b.v);
  }

  friend Avx512Uint32 operator~(Avx512Uint32 a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }

  // AVX-512 has native unsigned comparisons.
  friend Avx512Mask operator==(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_cmpeq_epi32_mask(a.v, b.v);
  }
  friend Avx512Mask operator!=(Avx512Uint32 a, Avx512Uint32 b) {
    return static_cast<__mmask16>(~_mm512_cmpeq_epi32_mask(a.v, b.v));
  }
  friend Avx512Mask operator>(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_cmpgt_epu32_mask(a.v, b.v);
  }
  friend Avx512Mask operator<(Avx512Uint32 a, Avx512Uint32 b) {
    return _mm512_cmplt_epu32_mask(a.v, b.v);
  }

  friend Avx512Mask operator!(Avx512Uint32 a) {
    return _mm512_cmpeq_epi32_mask(a.v, _mm512_setzero_si512());
  }
};

// --- Deferred inline definitions ---

// Avx512Mask → Avx512Int32: expand to lane-wide mask (all-ones or all-zeros).
inline Avx512Mask::operator Avx512Int32() const {
  return _mm512_maskz_set1_epi32(m, -1);
}

// Avx512Float ↔ Avx512Int32 conversion.
inline Avx512Float::Avx512Float(Avx512Int32 i) : v(_mm512_cvtepi32_ps(i.v)) {}

// Avx512Int32 ↔ Avx512Uint32 reinterpret (no-op on __m512i).
inline Avx512Int32::Avx512Int32(Avx512Uint32 u) : v(u.v) {}

// Map raw __m512 to Avx512Float for SimdTypeFor.
template <>
struct SimdTypeFor<__m512> {
  using type = Avx512Float;
};

// --- bit_cast specializations ---

template <>
inline Avx512Int32 bit_cast<Avx512Int32>(const Avx512Float& f) noexcept {
  return _mm512_castps_si512(f.v);
}
template <>
inline Avx512Uint32 bit_cast<Avx512Uint32>(const Avx512Float& f) noexcept {
  return _mm512_castps_si512(f.v);
}
template <>
inline Avx512Float bit_cast<Avx512Float>(const Avx512Int32& i) noexcept {
  return _mm512_castsi512_ps(i.v);
}
template <>
inline Avx512Float bit_cast<Avx512Float>(const Avx512Uint32& u) noexcept {
  return _mm512_castsi512_ps(u.v);
}
template <>
inline Avx512Int32 bit_cast<Avx512Int32>(const Avx512Uint32& u) noexcept {
  return u.v;
}
template <>
inline Avx512Uint32 bit_cast<Avx512Uint32>(const Avx512Int32& i) noexcept {
  return i.v;
}

// --- FloatTraits<Avx512Float> ---

template <>
struct FloatTraits<Avx512Float> {
  using IntType = Avx512Int32;
  using UintType = Avx512Uint32;
  using BoolType = Avx512Mask;

  static constexpr uint32_t kOne = 0x3f800000;
  static constexpr float kMagic = 12582912.f;
  static constexpr bool kBoolIsMask = true;

  static DISPENSO_INLINE Avx512Float sqrt(Avx512Float x) {
    return _mm512_sqrt_ps(x.v);
  }

  static DISPENSO_INLINE Avx512Float rcp(Avx512Float x) {
    return _mm512_rcp14_ps(x.v);
  }

  // AVX-512 always has FMA.
  static DISPENSO_INLINE Avx512Float fma(Avx512Float a, Avx512Float b, Avx512Float c) {
    return _mm512_fmadd_ps(a.v, b.v, c.v);
  }

  // conditional: select x where mask is true, y where false.
  // Native __mmask16 versions — the fast path.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(Avx512Mask mask, Arg x, Arg y);

  // Lane-wide int32 mask versions — for compatibility with code using IntType masks.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(Avx512Int32 mask, Arg x, Arg y);

  // apply: zero out lanes where mask is false.
  template <typename Arg>
  static DISPENSO_INLINE Arg apply(Avx512Mask mask, Arg x);

  static DISPENSO_INLINE Avx512Float min(Avx512Float a, Avx512Float b) {
    return _mm512_min_ps(a.v, b.v);
  }

  static DISPENSO_INLINE Avx512Float max(Avx512Float a, Avx512Float b) {
    return _mm512_max_ps(a.v, b.v);
  }
};

// conditional specializations (Avx512Mask).
template <>
inline Avx512Float
FloatTraits<Avx512Float>::conditional(Avx512Mask mask, Avx512Float x, Avx512Float y) {
  return _mm512_mask_blend_ps(mask.m, y.v, x.v);
}
template <>
inline Avx512Int32
FloatTraits<Avx512Float>::conditional(Avx512Mask mask, Avx512Int32 x, Avx512Int32 y) {
  return _mm512_mask_blend_epi32(mask.m, y.v, x.v);
}
template <>
inline Avx512Uint32
FloatTraits<Avx512Float>::conditional(Avx512Mask mask, Avx512Uint32 x, Avx512Uint32 y) {
  return _mm512_mask_blend_epi32(mask.m, y.v, x.v);
}

// conditional specializations (Avx512Int32 lane-wide mask).
// Convert lane-wide mask to __mmask16 by checking sign bit (equivalent to movepi32_mask,
// which requires AVX-512 DQ).
template <>
inline Avx512Float
FloatTraits<Avx512Float>::conditional(Avx512Int32 mask, Avx512Float x, Avx512Float y) {
  __mmask16 m = _mm512_cmplt_epi32_mask(mask.v, _mm512_setzero_si512());
  return _mm512_mask_blend_ps(m, y.v, x.v);
}
template <>
inline Avx512Int32
FloatTraits<Avx512Float>::conditional(Avx512Int32 mask, Avx512Int32 x, Avx512Int32 y) {
  __mmask16 m = _mm512_cmplt_epi32_mask(mask.v, _mm512_setzero_si512());
  return _mm512_mask_blend_epi32(m, y.v, x.v);
}
template <>
inline Avx512Uint32
FloatTraits<Avx512Float>::conditional(Avx512Int32 mask, Avx512Uint32 x, Avx512Uint32 y) {
  __mmask16 m = _mm512_cmplt_epi32_mask(mask.v, _mm512_setzero_si512());
  return _mm512_mask_blend_epi32(m, y.v, x.v);
}

// apply specializations (Avx512Mask): zero out lanes where mask is false.
template <>
inline Avx512Float FloatTraits<Avx512Float>::apply(Avx512Mask mask, Avx512Float x) {
  return _mm512_maskz_mov_ps(mask.m, x.v);
}
template <>
inline Avx512Int32 FloatTraits<Avx512Float>::apply(Avx512Mask mask, Avx512Int32 x) {
  return _mm512_maskz_mov_epi32(mask.m, x.v);
}
template <>
inline Avx512Uint32 FloatTraits<Avx512Float>::apply(Avx512Mask mask, Avx512Uint32 x) {
  return _mm512_maskz_mov_epi32(mask.m, x.v);
}

template <>
struct FloatTraits<Avx512Int32> {
  using IntType = Avx512Int32;
};

template <>
struct FloatTraits<Avx512Uint32> {
  using IntType = Avx512Uint32;
};

// --- Util function overloads for AVX-512 types ---

DISPENSO_INLINE Avx512Float floor_small(Avx512Float x) {
  return _mm512_floor_ps(x.v);
}

DISPENSO_INLINE Avx512Int32 convert_to_int_trunc(Avx512Float f) {
  return _mm512_cvttps_epi32(f.v);
}

DISPENSO_INLINE Avx512Int32 convert_to_int_trunc_safe(Avx512Float f) {
  Avx512Int32 fi = bit_cast<Avx512Int32>(f);
  __mmask16 norm = static_cast<__mmask16>(~_mm512_cmpeq_epi32_mask(
      _mm512_and_si512(fi.v, _mm512_set1_epi32(0x7f800000)), _mm512_set1_epi32(0x7f800000)));
  return _mm512_maskz_cvttps_epi32(norm, f.v);
}

DISPENSO_INLINE Avx512Int32 convert_to_int(Avx512Float f) {
  // _mm512_cvtps_epi32 uses round-to-nearest-even.
  // Use maskz to zero non-normal lanes and avoid undefined behavior.
  Avx512Int32 fi = bit_cast<Avx512Int32>(f);
  __mmask16 norm = static_cast<__mmask16>(~_mm512_cmpeq_epi32_mask(
      _mm512_and_si512(fi.v, _mm512_set1_epi32(0x7f800000)), _mm512_set1_epi32(0x7f800000)));
  return _mm512_maskz_cvtps_epi32(norm, f.v);
}

template <>
DISPENSO_INLINE Avx512Float min<Avx512Float>(Avx512Float x, Avx512Float mn) {
  // Ordering: if x is NaN, result is mn (second operand).
  return _mm512_min_ps(x.v, mn.v);
}

template <>
DISPENSO_INLINE Avx512Float
clamp_allow_nan<Avx512Float>(Avx512Float x, Avx512Float mn, Avx512Float mx) {
  return _mm512_max_ps(mn.v, _mm512_min_ps(mx.v, x.v));
}

template <>
DISPENSO_INLINE Avx512Float
clamp_no_nan<Avx512Float>(Avx512Float x, Avx512Float mn, Avx512Float mx) {
  return _mm512_max_ps(mn.v, _mm512_min_ps(x.v, mx.v));
}

template <>
DISPENSO_INLINE Avx512Float gather<Avx512Float>(const float* table, Avx512Int32 index) {
  // Note: AVX-512 gather has index as first arg, base as second.
  return _mm512_i32gather_ps(index.v, table, 4);
}

DISPENSO_INLINE Avx512Int32 int_div_by_3(Avx512Int32 i) {
  // Multiply each lane by 0x55555556 and take the high 32 bits.
  // Process even and odd lanes separately since _mm512_mul_epu32 only uses lanes 0,2,...
  __m512i multiplier = _mm512_set1_epi32(0x55555556);
  // Even lanes (0, 2, 4, ..., 14).
  __m512i even = _mm512_srli_epi64(_mm512_mul_epu32(i.v, multiplier), 32);
  // Odd lanes (1, 3, 5, ..., 15): shift right by 32 to put odd in even positions.
  __m512i i_odd = _mm512_srli_epi64(i.v, 32);
  __m512i odd = _mm512_srli_epi64(_mm512_mul_epu32(i_odd, multiplier), 32);
  odd = _mm512_slli_epi64(odd, 32);
  // Blend at 32-bit granularity: even indices from even, odd from odd.
  return _mm512_mask_blend_epi32(0xAAAA, even, odd); // 0xAAAA = odd lanes from 'odd'
}

// nonnormal/nonnormalOrZero: return Avx512Mask for AVX-512 types.
DISPENSO_INLINE Avx512Mask nonnormal(Avx512Int32 i) {
  return _mm512_cmpeq_epi32_mask(
      _mm512_and_si512(i.v, _mm512_set1_epi32(0x7f800000)), _mm512_set1_epi32(0x7f800000));
}

DISPENSO_INLINE Avx512Mask nonnormalOrZero(Avx512Int32 i) {
  __m512i masked = _mm512_and_si512(i.v, _mm512_set1_epi32(0x7f800000));
  __mmask16 isInfNan = _mm512_cmpeq_epi32_mask(masked, _mm512_set1_epi32(0x7f800000));
  __mmask16 isZero = _mm512_cmpeq_epi32_mask(masked, _mm512_setzero_si512());
  return static_cast<__mmask16>(isInfNan | isZero);
}

DISPENSO_INLINE Avx512Mask nonnormal(Avx512Float f) {
  return nonnormal(bit_cast<Avx512Int32>(f));
}

DISPENSO_INLINE Avx512Float signof(Avx512Float x) {
  Avx512Uint32 xi = bit_cast<Avx512Uint32>(x);
  return bit_cast<Avx512Float>((xi & 0x80000000u) | FloatTraits<Avx512Float>::kOne);
}

DISPENSO_INLINE Avx512Int32 signofi(Avx512Int32 i) {
  // Use mask blend: +1 for i >= 0, -1 for i < 0.
  __mmask16 neg = _mm512_cmplt_epi32_mask(i.v, _mm512_setzero_si512());
  return _mm512_mask_blend_epi32(neg, _mm512_set1_epi32(1), _mm512_set1_epi32(-1));
}

// nbool_as_one: 0 if mask is true, 1 if false.
template <>
DISPENSO_INLINE Avx512Float nbool_as_one<Avx512Float, Avx512Mask>(Avx512Mask b) {
  // Where mask is false (0 in mask bit), load 1.0f; where true, load 0.
  return _mm512_maskz_mov_ps(static_cast<__mmask16>(~b.m), _mm512_set1_ps(1.0f));
}

template <>
DISPENSO_INLINE Avx512Int32 nbool_as_one<Avx512Int32, Avx512Mask>(Avx512Mask b) {
  return _mm512_maskz_set1_epi32(static_cast<__mmask16>(~b.m), 1);
}

template <>
DISPENSO_INLINE Avx512Int32 nbool_as_one<Avx512Int32, Avx512Int32>(Avx512Int32 b) {
  // Lane-wide mask: non-zero → "true" → return 0; zero → "false" → return 1.
  __mmask16 isZero = _mm512_cmpeq_epi32_mask(b.v, _mm512_setzero_si512());
  return _mm512_maskz_set1_epi32(isZero, 1);
}

// bool_as_one: 1 if mask is true, 0 if false.
template <>
DISPENSO_INLINE Avx512Float bool_as_one<Avx512Float, Avx512Mask>(Avx512Mask b) {
  return _mm512_maskz_mov_ps(b.m, _mm512_set1_ps(1.0f));
}

template <>
DISPENSO_INLINE Avx512Int32 bool_as_one<Avx512Int32, Avx512Mask>(Avx512Mask b) {
  return _mm512_maskz_set1_epi32(b.m, 1);
}

// bool_as_mask: convert Avx512Mask to lane-wide Avx512Int32 mask.
template <>
DISPENSO_INLINE Avx512Int32 bool_as_mask<Avx512Int32, Avx512Mask>(Avx512Mask b) {
  return _mm512_maskz_set1_epi32(b.m, -1);
}

template <>
DISPENSO_INLINE Avx512Int32 bool_as_mask<Avx512Int32, Avx512Int32>(Avx512Int32 b) {
  return b;
}

template <>
DISPENSO_INLINE Avx512Int32 bool_as_mask<Avx512Int32, Avx512Uint32>(Avx512Uint32 b) {
  return b.v;
}

template <>
DISPENSO_INLINE Avx512Uint32 bool_as_mask<Avx512Uint32, Avx512Mask>(Avx512Mask b) {
  return _mm512_maskz_set1_epi32(b.m, -1);
}

template <>
DISPENSO_INLINE Avx512Uint32 bool_as_mask<Avx512Uint32, Avx512Uint32>(Avx512Uint32 b) {
  return b;
}

template <>
DISPENSO_INLINE Avx512Uint32 bool_as_mask<Avx512Uint32, Avx512Int32>(Avx512Int32 b) {
  return b.v;
}

// bool_apply_or_zero: specialized for Avx512Mask to use native masked operations.
template <>
DISPENSO_INLINE Avx512Float
bool_apply_or_zero<Avx512Float, Avx512Mask>(Avx512Mask b, Avx512Float val) {
  return _mm512_maskz_mov_ps(b.m, val.v);
}

template <>
DISPENSO_INLINE Avx512Int32
bool_apply_or_zero<Avx512Int32, Avx512Mask>(Avx512Mask b, Avx512Int32 val) {
  return _mm512_maskz_mov_epi32(b.m, val.v);
}

template <>
DISPENSO_INLINE Avx512Uint32
bool_apply_or_zero<Avx512Uint32, Avx512Mask>(Avx512Mask b, Avx512Uint32 val) {
  return _mm512_maskz_mov_epi32(b.m, val.v);
}

} // namespace fast_math
} // namespace dispenso

#endif // defined(__AVX512F__)
