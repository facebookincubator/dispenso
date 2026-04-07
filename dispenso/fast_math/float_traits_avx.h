/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__AVX2__)

#include <immintrin.h>

#include <cstdint>

#include "float_traits.h"
#include "util.h"

namespace dispenso {
namespace fast_math {

struct AvxInt32;
struct AvxUint32;

// 8-wide float SIMD wrapper around __m256.
struct AvxFloat {
  __m256 v;

  AvxFloat() = default;
  AvxFloat(__m256 vec) : v(vec) {}
  // Implicit broadcast from scalar — required for polynomial constant propagation.
  AvxFloat(float f) : v(_mm256_set1_ps(f)) {}
  // Explicit int→float conversion for (Flt)intExpr patterns.
  explicit AvxFloat(AvxInt32 i);

  // Implicit conversion back to raw intrinsic type.
  operator __m256() const {
    return v;
  }

  AvxFloat operator-() const {
    return _mm256_xor_ps(v, _mm256_set1_ps(-0.0f));
  }

  AvxFloat& operator+=(AvxFloat o) {
    v = _mm256_add_ps(v, o.v);
    return *this;
  }
  AvxFloat& operator-=(AvxFloat o) {
    v = _mm256_sub_ps(v, o.v);
    return *this;
  }
  AvxFloat& operator*=(AvxFloat o) {
    v = _mm256_mul_ps(v, o.v);
    return *this;
  }
  AvxFloat& operator&=(AvxFloat o) {
    v = _mm256_and_ps(v, o.v);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend AvxFloat operator+(AvxFloat a, AvxFloat b) {
    return _mm256_add_ps(a.v, b.v);
  }
  friend AvxFloat operator-(AvxFloat a, AvxFloat b) {
    return _mm256_sub_ps(a.v, b.v);
  }
  friend AvxFloat operator*(AvxFloat a, AvxFloat b) {
    return _mm256_mul_ps(a.v, b.v);
  }
  friend AvxFloat operator/(AvxFloat a, AvxFloat b) {
    return _mm256_div_ps(a.v, b.v);
  }

  // Comparisons return AvxFloat masks (all-ones or all-zeros per lane).
  friend AvxFloat operator<(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ);
  }
  friend AvxFloat operator>(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ);
  }
  friend AvxFloat operator<=(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ);
  }
  friend AvxFloat operator>=(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ);
  }
  friend AvxFloat operator==(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ);
  }
  friend AvxFloat operator!=(AvxFloat a, AvxFloat b) {
    return _mm256_cmp_ps(a.v, b.v, _CMP_NEQ_UQ);
  }

  // Logical NOT of a comparison mask.
  friend AvxFloat operator!(AvxFloat a) {
    return _mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
  }

  // Bitwise ops on float masks.
  friend AvxFloat operator&(AvxFloat a, AvxFloat b) {
    return _mm256_and_ps(a.v, b.v);
  }
  friend AvxFloat operator|(AvxFloat a, AvxFloat b) {
    return _mm256_or_ps(a.v, b.v);
  }
  friend AvxFloat operator^(AvxFloat a, AvxFloat b) {
    return _mm256_xor_ps(a.v, b.v);
  }
};

// 8-wide int32 SIMD wrapper around __m256i.
struct AvxInt32 {
  __m256i v;

  AvxInt32() = default;
  AvxInt32(__m256i vec) : v(vec) {}
  AvxInt32(int32_t i) : v(_mm256_set1_epi32(i)) {}
  // Reinterpret from AvxUint32 (no-op on __m256i).
  AvxInt32(AvxUint32 u);

  // Implicit conversion back to raw intrinsic type.
  operator __m256i() const {
    return v;
  }

  AvxInt32 operator-() const {
    return _mm256_sub_epi32(_mm256_setzero_si256(), v);
  }

  AvxInt32& operator+=(AvxInt32 o) {
    v = _mm256_add_epi32(v, o.v);
    return *this;
  }
  AvxInt32& operator-=(AvxInt32 o) {
    v = _mm256_sub_epi32(v, o.v);
    return *this;
  }
  AvxInt32& operator*=(AvxInt32 o) {
    v = _mm256_mullo_epi32(v, o.v);
    return *this;
  }
  AvxInt32& operator&=(AvxInt32 o) {
    v = _mm256_and_si256(v, o.v);
    return *this;
  }
  AvxInt32& operator|=(AvxInt32 o) {
    v = _mm256_or_si256(v, o.v);
    return *this;
  }

  // Shifts.
  AvxInt32 operator<<(int n) const {
    return _mm256_slli_epi32(v, n);
  }
  AvxInt32 operator>>(int n) const {
    return _mm256_srai_epi32(v, n); // Arithmetic shift right
  }
  AvxInt32& operator<<=(int n) {
    v = _mm256_slli_epi32(v, n);
    return *this;
  }
  AvxInt32& operator>>=(int n) {
    v = _mm256_srai_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend AvxInt32 operator+(AvxInt32 a, AvxInt32 b) {
    return _mm256_add_epi32(a.v, b.v);
  }
  friend AvxInt32 operator-(AvxInt32 a, AvxInt32 b) {
    return _mm256_sub_epi32(a.v, b.v);
  }
  friend AvxInt32 operator*(AvxInt32 a, AvxInt32 b) {
    return _mm256_mullo_epi32(a.v, b.v);
  }

  // Bitwise.
  friend AvxInt32 operator&(AvxInt32 a, AvxInt32 b) {
    return _mm256_and_si256(a.v, b.v);
  }
  friend AvxInt32 operator|(AvxInt32 a, AvxInt32 b) {
    return _mm256_or_si256(a.v, b.v);
  }
  friend AvxInt32 operator^(AvxInt32 a, AvxInt32 b) {
    return _mm256_xor_si256(a.v, b.v);
  }

  friend AvxInt32 operator~(AvxInt32 a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
  }

  // Comparisons return AvxInt32 masks.
  friend AvxInt32 operator==(AvxInt32 a, AvxInt32 b) {
    return _mm256_cmpeq_epi32(a.v, b.v);
  }
  friend AvxInt32 operator!=(AvxInt32 a, AvxInt32 b) {
    return _mm256_xor_si256(_mm256_cmpeq_epi32(a.v, b.v), _mm256_set1_epi32(-1));
  }
  friend AvxInt32 operator<(AvxInt32 a, AvxInt32 b) {
    return _mm256_cmpgt_epi32(b.v, a.v); // No _mm256_cmplt_epi32; use swapped cmpgt.
  }
  friend AvxInt32 operator>(AvxInt32 a, AvxInt32 b) {
    return _mm256_cmpgt_epi32(a.v, b.v);
  }

  friend AvxInt32 operator!(AvxInt32 a) {
    return _mm256_cmpeq_epi32(a.v, _mm256_setzero_si256());
  }
};

// 8-wide uint32 SIMD wrapper around __m256i.
struct AvxUint32 {
  __m256i v;

  AvxUint32() = default;
  AvxUint32(__m256i vec) : v(vec) {}
  AvxUint32(uint32_t u) : v(_mm256_set1_epi32(static_cast<int32_t>(u))) {}
  // Reinterpret from AvxInt32 (no-op on __m256i).
  AvxUint32(AvxInt32 i) : v(i.v) {}

  // Implicit conversion back to raw intrinsic type.
  operator __m256i() const {
    return v;
  }

  AvxUint32& operator+=(AvxUint32 o) {
    v = _mm256_add_epi32(v, o.v);
    return *this;
  }
  AvxUint32& operator-=(AvxUint32 o) {
    v = _mm256_sub_epi32(v, o.v);
    return *this;
  }
  AvxUint32& operator*=(AvxUint32 o) {
    v = _mm256_mullo_epi32(v, o.v);
    return *this;
  }
  AvxUint32& operator&=(AvxUint32 o) {
    v = _mm256_and_si256(v, o.v);
    return *this;
  }
  AvxUint32& operator|=(AvxUint32 o) {
    v = _mm256_or_si256(v, o.v);
    return *this;
  }

  // Shifts.
  AvxUint32 operator<<(int n) const {
    return _mm256_slli_epi32(v, n);
  }
  AvxUint32 operator>>(int n) const {
    return _mm256_srli_epi32(v, n); // Logical shift right
  }
  AvxUint32& operator<<=(int n) {
    v = _mm256_slli_epi32(v, n);
    return *this;
  }
  AvxUint32& operator>>=(int n) {
    v = _mm256_srli_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend AvxUint32 operator+(AvxUint32 a, AvxUint32 b) {
    return _mm256_add_epi32(a.v, b.v);
  }
  friend AvxUint32 operator-(AvxUint32 a, AvxUint32 b) {
    return _mm256_sub_epi32(a.v, b.v);
  }
  friend AvxUint32 operator*(AvxUint32 a, AvxUint32 b) {
    return _mm256_mullo_epi32(a.v, b.v);
  }

  // Bitwise.
  friend AvxUint32 operator&(AvxUint32 a, AvxUint32 b) {
    return _mm256_and_si256(a.v, b.v);
  }
  friend AvxUint32 operator|(AvxUint32 a, AvxUint32 b) {
    return _mm256_or_si256(a.v, b.v);
  }
  friend AvxUint32 operator^(AvxUint32 a, AvxUint32 b) {
    return _mm256_xor_si256(a.v, b.v);
  }

  friend AvxUint32 operator~(AvxUint32 a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
  }

  // Unsigned comparisons need XOR with sign bit to convert to signed domain.
  friend AvxUint32 operator==(AvxUint32 a, AvxUint32 b) {
    return _mm256_cmpeq_epi32(a.v, b.v);
  }
  friend AvxUint32 operator!=(AvxUint32 a, AvxUint32 b) {
    return _mm256_xor_si256(_mm256_cmpeq_epi32(a.v, b.v), _mm256_set1_epi32(-1));
  }
  friend AvxUint32 operator>(AvxUint32 a, AvxUint32 b) {
    __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
    return _mm256_cmpgt_epi32(_mm256_xor_si256(a.v, bias), _mm256_xor_si256(b.v, bias));
  }
  friend AvxUint32 operator<(AvxUint32 a, AvxUint32 b) {
    return b > a;
  }

  friend AvxUint32 operator!(AvxUint32 a) {
    return _mm256_cmpeq_epi32(a.v, _mm256_setzero_si256());
  }
};

// AvxFloat ↔ AvxInt32 conversion.
inline AvxFloat::AvxFloat(AvxInt32 i) : v(_mm256_cvtepi32_ps(i.v)) {}

// AvxInt32 ↔ AvxUint32 reinterpret (no-op on __m256i).
inline AvxInt32::AvxInt32(AvxUint32 u) : v(u.v) {}

// Map raw __m256 to AvxFloat for SimdTypeFor.
template <>
struct SimdTypeFor<__m256> {
  using type = AvxFloat;
};

// --- bit_cast specializations ---

template <>
inline AvxInt32 bit_cast<AvxInt32>(const AvxFloat& f) noexcept {
  return _mm256_castps_si256(f.v);
}
template <>
inline AvxUint32 bit_cast<AvxUint32>(const AvxFloat& f) noexcept {
  return _mm256_castps_si256(f.v);
}
template <>
inline AvxFloat bit_cast<AvxFloat>(const AvxInt32& i) noexcept {
  return _mm256_castsi256_ps(i.v);
}
template <>
inline AvxFloat bit_cast<AvxFloat>(const AvxUint32& u) noexcept {
  return _mm256_castsi256_ps(u.v);
}
template <>
inline AvxInt32 bit_cast<AvxInt32>(const AvxUint32& u) noexcept {
  return u.v;
}
template <>
inline AvxUint32 bit_cast<AvxUint32>(const AvxInt32& i) noexcept {
  return i.v;
}

// --- FloatTraits<AvxFloat> ---

template <>
struct FloatTraits<AvxFloat> {
  using IntType = AvxInt32;
  using UintType = AvxUint32;
  using BoolType = AvxFloat; // Float comparison masks

  static constexpr uint32_t kOne = 0x3f800000;
  static constexpr float kMagic = 12582912.f;
  static constexpr bool kBoolIsMask = true;

  static DISPENSO_INLINE AvxFloat sqrt(AvxFloat x) {
    return _mm256_sqrt_ps(x.v);
  }

  static DISPENSO_INLINE AvxFloat rcp(AvxFloat x) {
    return _mm256_rcp_ps(x.v);
  }

  static DISPENSO_INLINE AvxFloat fma(AvxFloat a, AvxFloat b, AvxFloat c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a.v, b.v, c.v);
#else
    return _mm256_add_ps(_mm256_mul_ps(a.v, b.v), c.v);
#endif
  }

  // conditional: select x where mask is true, y where false.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(AvxFloat mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(AvxInt32 mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg apply(AvxFloat mask, Arg x);

  static DISPENSO_INLINE AvxFloat min(AvxFloat a, AvxFloat b) {
    return _mm256_min_ps(a.v, b.v);
  }

  static DISPENSO_INLINE AvxFloat max(AvxFloat a, AvxFloat b) {
    return _mm256_max_ps(a.v, b.v);
  }
};

// conditional specializations.
template <>
inline AvxFloat FloatTraits<AvxFloat>::conditional(AvxFloat mask, AvxFloat x, AvxFloat y) {
  return _mm256_blendv_ps(y.v, x.v, mask.v);
}
template <>
inline AvxInt32 FloatTraits<AvxFloat>::conditional(AvxFloat mask, AvxInt32 x, AvxInt32 y) {
  return _mm256_castps_si256(
      _mm256_blendv_ps(_mm256_castsi256_ps(y.v), _mm256_castsi256_ps(x.v), mask.v));
}
template <>
inline AvxUint32 FloatTraits<AvxFloat>::conditional(AvxFloat mask, AvxUint32 x, AvxUint32 y) {
  return _mm256_castps_si256(
      _mm256_blendv_ps(_mm256_castsi256_ps(y.v), _mm256_castsi256_ps(x.v), mask.v));
}

template <>
inline AvxFloat FloatTraits<AvxFloat>::conditional(AvxInt32 mask, AvxFloat x, AvxFloat y) {
  return _mm256_blendv_ps(y.v, x.v, _mm256_castsi256_ps(mask.v));
}
template <>
inline AvxInt32 FloatTraits<AvxFloat>::conditional(AvxInt32 mask, AvxInt32 x, AvxInt32 y) {
  return _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(y.v), _mm256_castsi256_ps(x.v), _mm256_castsi256_ps(mask.v)));
}
template <>
inline AvxUint32 FloatTraits<AvxFloat>::conditional(AvxInt32 mask, AvxUint32 x, AvxUint32 y) {
  return _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(y.v), _mm256_castsi256_ps(x.v), _mm256_castsi256_ps(mask.v)));
}

// apply: mask & x (bitwise AND).
template <>
inline AvxFloat FloatTraits<AvxFloat>::apply(AvxFloat mask, AvxFloat x) {
  return _mm256_and_ps(mask.v, x.v);
}
template <>
inline AvxInt32 FloatTraits<AvxFloat>::apply(AvxFloat mask, AvxInt32 x) {
  return _mm256_and_si256(_mm256_castps_si256(mask.v), x.v);
}
template <>
inline AvxUint32 FloatTraits<AvxFloat>::apply(AvxFloat mask, AvxUint32 x) {
  return _mm256_and_si256(_mm256_castps_si256(mask.v), x.v);
}

template <>
struct FloatTraits<AvxInt32> {
  using IntType = AvxInt32;
};

template <>
struct FloatTraits<AvxUint32> {
  using IntType = AvxUint32;
};

// --- Util function overloads for AVX types ---

DISPENSO_INLINE AvxFloat floor_small(AvxFloat x) {
  return _mm256_floor_ps(x.v);
}

DISPENSO_INLINE AvxInt32 convert_to_int_trunc(AvxFloat f) {
  return _mm256_cvttps_epi32(f.v);
}

DISPENSO_INLINE AvxInt32 convert_to_int_trunc_safe(AvxFloat f) {
  AvxInt32 fi = bit_cast<AvxInt32>(f);
  AvxInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & AvxInt32(_mm256_cvttps_epi32(f.v));
}

DISPENSO_INLINE AvxInt32 convert_to_int(AvxFloat f) {
  // _mm256_cvtps_epi32 uses round-to-nearest-even.
  // Mask non-normals to 0 to avoid undefined behavior.
  AvxInt32 fi = bit_cast<AvxInt32>(f);
  AvxInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & AvxInt32(_mm256_cvtps_epi32(f.v));
}

template <>
DISPENSO_INLINE AvxFloat min<AvxFloat>(AvxFloat x, AvxFloat mn) {
  // Ordering: if x is NaN, result is mn (second operand).
  return _mm256_min_ps(x.v, mn.v);
}

template <>
DISPENSO_INLINE AvxFloat clamp_allow_nan<AvxFloat>(AvxFloat x, AvxFloat mn, AvxFloat mx) {
  return _mm256_max_ps(mn.v, _mm256_min_ps(mx.v, x.v));
}

template <>
DISPENSO_INLINE AvxFloat clamp_no_nan<AvxFloat>(AvxFloat x, AvxFloat mn, AvxFloat mx) {
  return _mm256_max_ps(mn.v, _mm256_min_ps(x.v, mx.v));
}

template <>
DISPENSO_INLINE AvxFloat gather<AvxFloat>(const float* table, AvxInt32 index) {
  return _mm256_i32gather_ps(table, index.v, 4);
}

DISPENSO_INLINE AvxInt32 int_div_by_3(AvxInt32 i) {
  // Multiply each lane by 0x55555556 and take the high 32 bits.
  // Process even and odd lanes separately since _mm256_mul_epu32 only uses lanes 0,2,4,6.
  __m256i multiplier = _mm256_set1_epi32(0x55555556);
  // Even lanes (0, 2, 4, 6).
  __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(i.v, multiplier), 32);
  // Odd lanes (1, 3, 5, 7): shift right by 32 bits to put odd lanes in even positions.
  __m256i i_odd = _mm256_srli_epi64(i.v, 32);
  __m256i odd = _mm256_srli_epi64(_mm256_mul_epu32(i_odd, multiplier), 32);
  odd = _mm256_slli_epi64(odd, 32);
  // Blend at 32-bit granularity: even indices from even, odd from odd.
  return _mm256_blend_epi32(even, odd, 0xAA); // 0xAA = 10101010b
}

// nonnormal/nonnormalOrZero: return AvxInt32 masks for AVX types.
DISPENSO_INLINE AvxInt32 nonnormal(AvxInt32 i) {
  return (i & 0x7f800000) == 0x7f800000;
}

DISPENSO_INLINE AvxInt32 nonnormalOrZero(AvxInt32 i) {
  auto m = i & 0x7f800000;
  return (m == 0x7f800000) | (m == 0);
}

DISPENSO_INLINE AvxInt32 nonnormal(AvxFloat f) {
  return nonnormal(bit_cast<AvxInt32>(f));
}

// any_true: reduce SIMD mask to scalar bool (true if any lane is set).
DISPENSO_INLINE bool any_true(AvxInt32 mask) {
  return _mm256_movemask_ps(_mm256_castsi256_ps(mask.v)) != 0;
}

DISPENSO_INLINE AvxFloat signof(AvxFloat x) {
  AvxUint32 xi = bit_cast<AvxUint32>(x);
  return bit_cast<AvxFloat>((xi & 0x80000000u) | FloatTraits<AvxFloat>::kOne);
}

DISPENSO_INLINE AvxInt32 signofi(AvxInt32 i) {
  return AvxInt32(1) - (AvxInt32(2) & (i < AvxInt32(0)));
}

// nbool_as_one: 0 if mask is true (all-ones), 1 if false (all-zeros).
template <>
DISPENSO_INLINE AvxFloat nbool_as_one<AvxFloat, AvxFloat>(AvxFloat b) {
  // ~mask & 1.0f bits
  return bit_cast<AvxFloat>(
      AvxInt32(_mm256_andnot_si256(_mm256_castps_si256(b.v), _mm256_set1_epi32(0x3f800000))));
}

template <>
DISPENSO_INLINE AvxInt32 nbool_as_one<AvxInt32, AvxFloat>(AvxFloat b) {
  return _mm256_andnot_si256(_mm256_castps_si256(b.v), _mm256_set1_epi32(1));
}

template <>
DISPENSO_INLINE AvxInt32 nbool_as_one<AvxInt32, AvxInt32>(AvxInt32 b) {
  return _mm256_andnot_si256(b.v, _mm256_set1_epi32(1));
}

// bool_as_one: 1 if mask is true, 0 if false.
template <>
DISPENSO_INLINE AvxFloat bool_as_one<AvxFloat, AvxFloat>(AvxFloat b) {
  return bit_cast<AvxFloat>(
      AvxInt32(_mm256_and_si256(_mm256_castps_si256(b.v), _mm256_set1_epi32(0x3f800000))));
}

template <>
DISPENSO_INLINE AvxInt32 bool_as_one<AvxInt32, AvxFloat>(AvxFloat b) {
  return _mm256_and_si256(_mm256_castps_si256(b.v), _mm256_set1_epi32(1));
}

// bool_as_mask: for SIMD masks, identity (already a mask).
template <>
DISPENSO_INLINE AvxInt32 bool_as_mask<AvxInt32, AvxFloat>(AvxFloat b) {
  return _mm256_castps_si256(b.v);
}

template <>
DISPENSO_INLINE AvxInt32 bool_as_mask<AvxInt32, AvxInt32>(AvxInt32 b) {
  return b;
}

template <>
DISPENSO_INLINE AvxInt32 bool_as_mask<AvxInt32, AvxUint32>(AvxUint32 b) {
  return b.v;
}

template <>
DISPENSO_INLINE AvxUint32 bool_as_mask<AvxUint32, AvxFloat>(AvxFloat b) {
  return _mm256_castps_si256(b.v);
}

template <>
DISPENSO_INLINE AvxUint32 bool_as_mask<AvxUint32, AvxUint32>(AvxUint32 b) {
  return b;
}

template <>
DISPENSO_INLINE AvxUint32 bool_as_mask<AvxUint32, AvxInt32>(AvxInt32 b) {
  return b.v;
}

} // namespace fast_math
} // namespace dispenso

#endif // defined(__AVX2__)
