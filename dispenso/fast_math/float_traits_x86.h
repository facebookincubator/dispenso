/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__SSE4_1__)

#include <immintrin.h>

#include <cstdint>

#include "float_traits.h"
#include "util.h"

namespace dispenso {
namespace fast_math {

struct SseInt32;
struct SseUint32;

// 4-wide float SIMD wrapper around __m128.
struct SseFloat {
  __m128 v;

  SseFloat() = default;
  SseFloat(__m128 vec) : v(vec) {}
  // Implicit broadcast from scalar — required for polynomial constant propagation.
  SseFloat(float f) : v(_mm_set1_ps(f)) {}
  // Explicit int→float conversion for (Flt)intExpr patterns.
  explicit SseFloat(SseInt32 i);

  // Implicit conversion back to raw intrinsic type.
  operator __m128() const {
    return v;
  }

  SseFloat operator-() const {
    return _mm_xor_ps(v, _mm_set1_ps(-0.0f));
  }

  SseFloat& operator+=(SseFloat o) {
    v = _mm_add_ps(v, o.v);
    return *this;
  }
  SseFloat& operator-=(SseFloat o) {
    v = _mm_sub_ps(v, o.v);
    return *this;
  }
  SseFloat& operator*=(SseFloat o) {
    v = _mm_mul_ps(v, o.v);
    return *this;
  }
  SseFloat& operator&=(SseFloat o) {
    v = _mm_and_ps(v, o.v);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend SseFloat operator+(SseFloat a, SseFloat b) {
    return _mm_add_ps(a.v, b.v);
  }
  friend SseFloat operator-(SseFloat a, SseFloat b) {
    return _mm_sub_ps(a.v, b.v);
  }
  friend SseFloat operator*(SseFloat a, SseFloat b) {
    return _mm_mul_ps(a.v, b.v);
  }
  friend SseFloat operator/(SseFloat a, SseFloat b) {
    return _mm_div_ps(a.v, b.v);
  }

  // Comparisons return SseFloat masks (all-ones or all-zeros per lane).
  friend SseFloat operator<(SseFloat a, SseFloat b) {
    return _mm_cmplt_ps(a.v, b.v);
  }
  friend SseFloat operator>(SseFloat a, SseFloat b) {
    return _mm_cmpgt_ps(a.v, b.v);
  }
  friend SseFloat operator<=(SseFloat a, SseFloat b) {
    return _mm_cmple_ps(a.v, b.v);
  }
  friend SseFloat operator>=(SseFloat a, SseFloat b) {
    return _mm_cmpge_ps(a.v, b.v);
  }
  friend SseFloat operator==(SseFloat a, SseFloat b) {
    return _mm_cmpeq_ps(a.v, b.v);
  }
  friend SseFloat operator!=(SseFloat a, SseFloat b) {
    return _mm_cmpneq_ps(a.v, b.v);
  }

  // Logical NOT of a comparison mask.
  friend SseFloat operator!(SseFloat a) {
    return _mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(-1)));
  }

  // Bitwise ops on float masks.
  friend SseFloat operator&(SseFloat a, SseFloat b) {
    return _mm_and_ps(a.v, b.v);
  }
  friend SseFloat operator|(SseFloat a, SseFloat b) {
    return _mm_or_ps(a.v, b.v);
  }
  friend SseFloat operator^(SseFloat a, SseFloat b) {
    return _mm_xor_ps(a.v, b.v);
  }
};

// 4-wide int32 SIMD wrapper around __m128i.
struct SseInt32 {
  __m128i v;

  SseInt32() = default;
  SseInt32(__m128i vec) : v(vec) {}
  SseInt32(int32_t i) : v(_mm_set1_epi32(i)) {}
  // Reinterpret from SseUint32 (no-op on __m128i).
  SseInt32(SseUint32 u);

  // Implicit conversion back to raw intrinsic type.
  operator __m128i() const {
    return v;
  }

  SseInt32 operator-() const {
    return _mm_sub_epi32(_mm_setzero_si128(), v);
  }

  SseInt32& operator+=(SseInt32 o) {
    v = _mm_add_epi32(v, o.v);
    return *this;
  }
  SseInt32& operator-=(SseInt32 o) {
    v = _mm_sub_epi32(v, o.v);
    return *this;
  }
  SseInt32& operator*=(SseInt32 o) {
    v = _mm_mullo_epi32(v, o.v);
    return *this;
  }
  SseInt32& operator&=(SseInt32 o) {
    v = _mm_and_si128(v, o.v);
    return *this;
  }
  SseInt32& operator|=(SseInt32 o) {
    v = _mm_or_si128(v, o.v);
    return *this;
  }

  // Shifts.
  SseInt32 operator<<(int n) const {
    return _mm_slli_epi32(v, n);
  }
  SseInt32 operator>>(int n) const {
    return _mm_srai_epi32(v, n); // Arithmetic shift right
  }
  SseInt32& operator<<=(int n) {
    v = _mm_slli_epi32(v, n);
    return *this;
  }
  SseInt32& operator>>=(int n) {
    v = _mm_srai_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend SseInt32 operator+(SseInt32 a, SseInt32 b) {
    return _mm_add_epi32(a.v, b.v);
  }
  friend SseInt32 operator-(SseInt32 a, SseInt32 b) {
    return _mm_sub_epi32(a.v, b.v);
  }
  friend SseInt32 operator*(SseInt32 a, SseInt32 b) {
    return _mm_mullo_epi32(a.v, b.v); // SSE4.1
  }

  // Bitwise.
  friend SseInt32 operator&(SseInt32 a, SseInt32 b) {
    return _mm_and_si128(a.v, b.v);
  }
  friend SseInt32 operator|(SseInt32 a, SseInt32 b) {
    return _mm_or_si128(a.v, b.v);
  }
  friend SseInt32 operator^(SseInt32 a, SseInt32 b) {
    return _mm_xor_si128(a.v, b.v);
  }

  friend SseInt32 operator~(SseInt32 a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }

  // Comparisons return SseInt32 masks.
  friend SseInt32 operator==(SseInt32 a, SseInt32 b) {
    return _mm_cmpeq_epi32(a.v, b.v);
  }
  friend SseInt32 operator!=(SseInt32 a, SseInt32 b) {
    return _mm_xor_si128(_mm_cmpeq_epi32(a.v, b.v), _mm_set1_epi32(-1));
  }
  friend SseInt32 operator<(SseInt32 a, SseInt32 b) {
    return _mm_cmplt_epi32(a.v, b.v);
  }
  friend SseInt32 operator>(SseInt32 a, SseInt32 b) {
    return _mm_cmpgt_epi32(a.v, b.v);
  }

  friend SseInt32 operator!(SseInt32 a) {
    return _mm_cmpeq_epi32(a.v, _mm_setzero_si128());
  }
};

// 4-wide uint32 SIMD wrapper around __m128i.
struct SseUint32 {
  __m128i v;

  SseUint32() = default;
  SseUint32(__m128i vec) : v(vec) {}
  SseUint32(uint32_t u) : v(_mm_set1_epi32(static_cast<int32_t>(u))) {}
  // Reinterpret from SseInt32 (no-op on __m128i).
  SseUint32(SseInt32 i) : v(i.v) {}

  // Implicit conversion back to raw intrinsic type.
  operator __m128i() const {
    return v;
  }

  SseUint32& operator+=(SseUint32 o) {
    v = _mm_add_epi32(v, o.v);
    return *this;
  }
  SseUint32& operator-=(SseUint32 o) {
    v = _mm_sub_epi32(v, o.v);
    return *this;
  }
  SseUint32& operator*=(SseUint32 o) {
    v = _mm_mullo_epi32(v, o.v);
    return *this;
  }
  SseUint32& operator&=(SseUint32 o) {
    v = _mm_and_si128(v, o.v);
    return *this;
  }
  SseUint32& operator|=(SseUint32 o) {
    v = _mm_or_si128(v, o.v);
    return *this;
  }

  // Shifts.
  SseUint32 operator<<(int n) const {
    return _mm_slli_epi32(v, n);
  }
  SseUint32 operator>>(int n) const {
    return _mm_srli_epi32(v, n); // Logical shift right
  }
  SseUint32& operator<<=(int n) {
    v = _mm_slli_epi32(v, n);
    return *this;
  }
  SseUint32& operator>>=(int n) {
    v = _mm_srli_epi32(v, n);
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend SseUint32 operator+(SseUint32 a, SseUint32 b) {
    return _mm_add_epi32(a.v, b.v);
  }
  friend SseUint32 operator-(SseUint32 a, SseUint32 b) {
    return _mm_sub_epi32(a.v, b.v);
  }
  friend SseUint32 operator*(SseUint32 a, SseUint32 b) {
    return _mm_mullo_epi32(a.v, b.v);
  }

  // Bitwise.
  friend SseUint32 operator&(SseUint32 a, SseUint32 b) {
    return _mm_and_si128(a.v, b.v);
  }
  friend SseUint32 operator|(SseUint32 a, SseUint32 b) {
    return _mm_or_si128(a.v, b.v);
  }
  friend SseUint32 operator^(SseUint32 a, SseUint32 b) {
    return _mm_xor_si128(a.v, b.v);
  }

  friend SseUint32 operator~(SseUint32 a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }

  // Unsigned comparisons need XOR with sign bit to convert to signed domain.
  friend SseUint32 operator==(SseUint32 a, SseUint32 b) {
    return _mm_cmpeq_epi32(a.v, b.v);
  }
  friend SseUint32 operator!=(SseUint32 a, SseUint32 b) {
    return _mm_xor_si128(_mm_cmpeq_epi32(a.v, b.v), _mm_set1_epi32(-1));
  }
  friend SseUint32 operator>(SseUint32 a, SseUint32 b) {
    __m128i bias = _mm_set1_epi32(static_cast<int32_t>(0x80000000u));
    return _mm_cmpgt_epi32(_mm_xor_si128(a.v, bias), _mm_xor_si128(b.v, bias));
  }
  friend SseUint32 operator<(SseUint32 a, SseUint32 b) {
    return b > a;
  }

  friend SseUint32 operator!(SseUint32 a) {
    return _mm_cmpeq_epi32(a.v, _mm_setzero_si128());
  }
};

// SseFloat ↔ SseInt32 conversion.
inline SseFloat::SseFloat(SseInt32 i) : v(_mm_cvtepi32_ps(i.v)) {}

// SseInt32 ↔ SseUint32 reinterpret (no-op on __m128i).
inline SseInt32::SseInt32(SseUint32 u) : v(u.v) {}

// Map raw __m128 to SseFloat for SimdTypeFor.
template <>
struct SimdTypeFor<__m128> {
  using type = SseFloat;
};

// --- bit_cast specializations ---

template <>
inline SseInt32 bit_cast<SseInt32>(const SseFloat& f) noexcept {
  return _mm_castps_si128(f.v);
}
template <>
inline SseUint32 bit_cast<SseUint32>(const SseFloat& f) noexcept {
  return _mm_castps_si128(f.v);
}
template <>
inline SseFloat bit_cast<SseFloat>(const SseInt32& i) noexcept {
  return _mm_castsi128_ps(i.v);
}
template <>
inline SseFloat bit_cast<SseFloat>(const SseUint32& u) noexcept {
  return _mm_castsi128_ps(u.v);
}
template <>
inline SseInt32 bit_cast<SseInt32>(const SseUint32& u) noexcept {
  return u.v;
}
template <>
inline SseUint32 bit_cast<SseUint32>(const SseInt32& i) noexcept {
  return i.v;
}

// --- FloatTraits<SseFloat> ---

template <>
struct FloatTraits<SseFloat> {
  using IntType = SseInt32;
  using UintType = SseUint32;
  using BoolType = SseFloat; // Float comparison masks

  static constexpr uint32_t kOne = 0x3f800000;
  static constexpr float kMagic = 12582912.f;
  static constexpr bool kBoolIsMask = true;

  static DISPENSO_INLINE SseFloat sqrt(SseFloat x) {
    return _mm_sqrt_ps(x.v);
  }

  static DISPENSO_INLINE SseFloat rcp(SseFloat x) {
    return _mm_rcp_ps(x.v);
  }

  static DISPENSO_INLINE SseFloat fma(SseFloat a, SseFloat b, SseFloat c) {
#if defined(__FMA__)
    return _mm_fmadd_ps(a.v, b.v, c.v);
#else
    return _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v);
#endif
  }

  // conditional: select x where mask is true, y where false.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(SseFloat mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(SseInt32 mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg apply(SseFloat mask, Arg x);

  static DISPENSO_INLINE SseFloat min(SseFloat a, SseFloat b) {
    return _mm_min_ps(a.v, b.v);
  }

  static DISPENSO_INLINE SseFloat max(SseFloat a, SseFloat b) {
    return _mm_max_ps(a.v, b.v);
  }
};

// conditional specializations.
template <>
inline SseFloat FloatTraits<SseFloat>::conditional(SseFloat mask, SseFloat x, SseFloat y) {
  return _mm_blendv_ps(y.v, x.v, mask.v);
}
template <>
inline SseInt32 FloatTraits<SseFloat>::conditional(SseFloat mask, SseInt32 x, SseInt32 y) {
  return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(y.v), _mm_castsi128_ps(x.v), mask.v));
}
template <>
inline SseUint32 FloatTraits<SseFloat>::conditional(SseFloat mask, SseUint32 x, SseUint32 y) {
  return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(y.v), _mm_castsi128_ps(x.v), mask.v));
}

template <>
inline SseFloat FloatTraits<SseFloat>::conditional(SseInt32 mask, SseFloat x, SseFloat y) {
  return _mm_blendv_ps(y.v, x.v, _mm_castsi128_ps(mask.v));
}
template <>
inline SseInt32 FloatTraits<SseFloat>::conditional(SseInt32 mask, SseInt32 x, SseInt32 y) {
  return _mm_castps_si128(
      _mm_blendv_ps(_mm_castsi128_ps(y.v), _mm_castsi128_ps(x.v), _mm_castsi128_ps(mask.v)));
}
template <>
inline SseUint32 FloatTraits<SseFloat>::conditional(SseInt32 mask, SseUint32 x, SseUint32 y) {
  return _mm_castps_si128(
      _mm_blendv_ps(_mm_castsi128_ps(y.v), _mm_castsi128_ps(x.v), _mm_castsi128_ps(mask.v)));
}

// apply: mask & x (bitwise AND).
template <>
inline SseFloat FloatTraits<SseFloat>::apply(SseFloat mask, SseFloat x) {
  return _mm_and_ps(mask.v, x.v);
}
template <>
inline SseInt32 FloatTraits<SseFloat>::apply(SseFloat mask, SseInt32 x) {
  return _mm_and_si128(_mm_castps_si128(mask.v), x.v);
}
template <>
inline SseUint32 FloatTraits<SseFloat>::apply(SseFloat mask, SseUint32 x) {
  return _mm_and_si128(_mm_castps_si128(mask.v), x.v);
}

template <>
struct FloatTraits<SseInt32> {
  using IntType = SseInt32;
};

template <>
struct FloatTraits<SseUint32> {
  using IntType = SseUint32;
};

// --- Util function overloads for SSE types ---

DISPENSO_INLINE SseFloat floor_small(SseFloat x) {
  return _mm_floor_ps(x.v);
}

DISPENSO_INLINE SseInt32 convert_to_int_trunc(SseFloat f) {
  return _mm_cvttps_epi32(f.v);
}

DISPENSO_INLINE SseInt32 convert_to_int_trunc_safe(SseFloat f) {
  SseInt32 fi = bit_cast<SseInt32>(f);
  SseInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & SseInt32(_mm_cvttps_epi32(f.v));
}

DISPENSO_INLINE SseInt32 convert_to_int(SseFloat f) {
  // _mm_cvtps_epi32 uses round-to-nearest-even.
  // Mask non-normals to 0 to avoid undefined behavior.
  SseInt32 fi = bit_cast<SseInt32>(f);
  SseInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & SseInt32(_mm_cvtps_epi32(f.v));
}

template <>
DISPENSO_INLINE SseFloat min<SseFloat>(SseFloat x, SseFloat mn) {
  // Ordering: if x is NaN, result is mn (second operand).
  return _mm_min_ps(x.v, mn.v);
}

template <>
DISPENSO_INLINE SseFloat clamp_allow_nan<SseFloat>(SseFloat x, SseFloat mn, SseFloat mx) {
  return _mm_max_ps(mn.v, _mm_min_ps(mx.v, x.v));
}

template <>
DISPENSO_INLINE SseFloat clamp_no_nan<SseFloat>(SseFloat x, SseFloat mn, SseFloat mx) {
  return _mm_max_ps(mn.v, _mm_min_ps(x.v, mx.v));
}

template <>
DISPENSO_INLINE SseFloat gather<SseFloat>(const float* table, SseInt32 index) {
  alignas(16) int32_t idx[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(idx), index.v);
  return _mm_set_ps(table[idx[3]], table[idx[2]], table[idx[1]], table[idx[0]]);
}

DISPENSO_INLINE SseInt32 int_div_by_3(SseInt32 i) {
  // Multiply each lane by 0x55555556 and take the high 32 bits.
  // Process even and odd lanes separately since _mm_mul_epu32 only uses lanes 0,2.
  __m128i multiplier = _mm_set1_epi32(0x55555556);
  // Even lanes (0, 2).
  __m128i even = _mm_srli_epi64(_mm_mul_epu32(i.v, multiplier), 32);
  // Odd lanes (1, 3): shift right by 32 bits to put odd lanes in even positions.
  __m128i i_odd = _mm_srli_epi64(i.v, 32);
  __m128i odd = _mm_srli_epi64(_mm_mul_epu32(i_odd, multiplier), 32);
  odd = _mm_slli_epi64(odd, 32);
  // Blend even and odd results.
  return _mm_blend_epi16(even, odd, 0xCC); // 0xCC = 11001100b selects odd from odd
}

// nonnormal/nonnormalOrZero: return SseInt32 masks for SSE types.
DISPENSO_INLINE SseInt32 nonnormal(SseInt32 i) {
  return (i & 0x7f800000) == 0x7f800000;
}

DISPENSO_INLINE SseInt32 nonnormalOrZero(SseInt32 i) {
  auto m = i & 0x7f800000;
  return (m == 0x7f800000) | (m == 0);
}

DISPENSO_INLINE SseInt32 nonnormal(SseFloat f) {
  return nonnormal(bit_cast<SseInt32>(f));
}

// any_true: reduce SIMD mask to scalar bool (true if any lane is set).
DISPENSO_INLINE bool any_true(SseInt32 mask) {
  return _mm_movemask_ps(_mm_castsi128_ps(mask.v)) != 0;
}

DISPENSO_INLINE SseFloat signof(SseFloat x) {
  SseUint32 xi = bit_cast<SseUint32>(x);
  return bit_cast<SseFloat>((xi & 0x80000000u) | FloatTraits<SseFloat>::kOne);
}

DISPENSO_INLINE SseInt32 signofi(SseInt32 i) {
  return SseInt32(1) - (SseInt32(2) & (i < SseInt32(0)));
}

// nbool_as_one: 0 if mask is true (all-ones), 1 if false (all-zeros).
template <>
DISPENSO_INLINE SseFloat nbool_as_one<SseFloat, SseFloat>(SseFloat b) {
  // ~mask & 1.0f bits
  return bit_cast<SseFloat>(
      SseInt32(_mm_andnot_si128(_mm_castps_si128(b.v), _mm_set1_epi32(0x3f800000))));
}

template <>
DISPENSO_INLINE SseInt32 nbool_as_one<SseInt32, SseFloat>(SseFloat b) {
  return _mm_andnot_si128(_mm_castps_si128(b.v), _mm_set1_epi32(1));
}

template <>
DISPENSO_INLINE SseInt32 nbool_as_one<SseInt32, SseInt32>(SseInt32 b) {
  return _mm_andnot_si128(b.v, _mm_set1_epi32(1));
}

// bool_as_one: 1 if mask is true, 0 if false.
template <>
DISPENSO_INLINE SseFloat bool_as_one<SseFloat, SseFloat>(SseFloat b) {
  return bit_cast<SseFloat>(
      SseInt32(_mm_and_si128(_mm_castps_si128(b.v), _mm_set1_epi32(0x3f800000))));
}

template <>
DISPENSO_INLINE SseInt32 bool_as_one<SseInt32, SseFloat>(SseFloat b) {
  return _mm_and_si128(_mm_castps_si128(b.v), _mm_set1_epi32(1));
}

// bool_as_mask: for SIMD masks, identity (already a mask).
template <>
DISPENSO_INLINE SseInt32 bool_as_mask<SseInt32, SseFloat>(SseFloat b) {
  return _mm_castps_si128(b.v);
}

template <>
DISPENSO_INLINE SseInt32 bool_as_mask<SseInt32, SseInt32>(SseInt32 b) {
  return b;
}

template <>
DISPENSO_INLINE SseInt32 bool_as_mask<SseInt32, SseUint32>(SseUint32 b) {
  return b.v;
}

template <>
DISPENSO_INLINE SseUint32 bool_as_mask<SseUint32, SseFloat>(SseFloat b) {
  return _mm_castps_si128(b.v);
}

template <>
DISPENSO_INLINE SseUint32 bool_as_mask<SseUint32, SseUint32>(SseUint32 b) {
  return b;
}

template <>
DISPENSO_INLINE SseUint32 bool_as_mask<SseUint32, SseInt32>(SseInt32 b) {
  return b.v;
}

} // namespace fast_math
} // namespace dispenso

#endif // defined(__SSE4_1__)
