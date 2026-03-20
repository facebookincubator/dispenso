/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Only include when Highway is available on the include path.
// Consumers must depend on the Highway library target.
#if __has_include("hwy/highway.h")

#include "hwy/highway.h"

#include <cstdint>

#include "float_traits.h"
#include "util.h"

namespace dispenso {
namespace fast_math {

namespace hn = hwy::HWY_NAMESPACE;

using HwyFloatTag = hn::ScalableTag<float>;
using HwyInt32Tag = hn::RebindToSigned<HwyFloatTag>;
using HwyUint32Tag = hn::RebindToUnsigned<HwyFloatTag>;

struct HwyInt32;
struct HwyUint32;

// N-wide float SIMD wrapper around Highway Vec<ScalableTag<float>>.
// Width is determined at compile time by Highway's static dispatch.
struct HwyFloat {
  hn::Vec<HwyFloatTag> v;

  HwyFloat() = default;
  HwyFloat(hn::Vec<HwyFloatTag> vec) : v(vec) {}
  // Implicit broadcast from scalar — required for polynomial constant propagation.
  HwyFloat(float f) : v(hn::Set(HwyFloatTag{}, f)) {}
  // Explicit int→float conversion for (Flt)intExpr patterns.
  explicit HwyFloat(HwyInt32 i);

  HwyFloat operator-() const {
    return hn::Neg(v);
  }

  HwyFloat& operator+=(HwyFloat o) {
    v = hn::Add(v, o.v);
    return *this;
  }
  HwyFloat& operator-=(HwyFloat o) {
    v = hn::Sub(v, o.v);
    return *this;
  }
  HwyFloat& operator*=(HwyFloat o) {
    v = hn::Mul(v, o.v);
    return *this;
  }
  HwyFloat& operator&=(HwyFloat o) {
    const HwyInt32Tag di;
    v = hn::BitCast(HwyFloatTag{}, hn::And(hn::BitCast(di, v), hn::BitCast(di, o.v)));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend HwyFloat operator+(HwyFloat a, HwyFloat b) {
    return hn::Add(a.v, b.v);
  }
  friend HwyFloat operator-(HwyFloat a, HwyFloat b) {
    return hn::Sub(a.v, b.v);
  }
  friend HwyFloat operator*(HwyFloat a, HwyFloat b) {
    return hn::Mul(a.v, b.v);
  }
  friend HwyFloat operator/(HwyFloat a, HwyFloat b) {
    return hn::Div(a.v, b.v);
  }

  // Comparisons return HwyFloat masks (all-ones or all-zeros per lane).
  friend HwyFloat operator<(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Lt(a.v, b.v));
  }
  friend HwyFloat operator>(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Gt(a.v, b.v));
  }
  friend HwyFloat operator<=(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Le(a.v, b.v));
  }
  friend HwyFloat operator>=(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Ge(a.v, b.v));
  }
  friend HwyFloat operator==(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Eq(a.v, b.v));
  }
  friend HwyFloat operator!=(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    return hn::VecFromMask(d, hn::Ne(a.v, b.v));
  }

  // Logical NOT of a comparison mask.
  friend HwyFloat operator!(HwyFloat a) {
    const HwyFloatTag d;
    const HwyInt32Tag di;
    return hn::BitCast(d, hn::Xor(hn::BitCast(di, a.v), hn::Set(di, -1)));
  }

  // Bitwise ops on float masks.
  friend HwyFloat operator&(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    const HwyInt32Tag di;
    return hn::BitCast(d, hn::And(hn::BitCast(di, a.v), hn::BitCast(di, b.v)));
  }
  friend HwyFloat operator|(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    const HwyInt32Tag di;
    return hn::BitCast(d, hn::Or(hn::BitCast(di, a.v), hn::BitCast(di, b.v)));
  }
  friend HwyFloat operator^(HwyFloat a, HwyFloat b) {
    const HwyFloatTag d;
    const HwyInt32Tag di;
    return hn::BitCast(d, hn::Xor(hn::BitCast(di, a.v), hn::BitCast(di, b.v)));
  }
};

// N-wide int32 SIMD wrapper around Highway Vec<RebindToSigned<ScalableTag<float>>>.
struct HwyInt32 {
  hn::Vec<HwyInt32Tag> v;

  HwyInt32() = default;
  HwyInt32(hn::Vec<HwyInt32Tag> vec) : v(vec) {}
  HwyInt32(int32_t i) : v(hn::Set(HwyInt32Tag{}, i)) {}
  // Reinterpret from HwyUint32 (no-op).
  HwyInt32(HwyUint32 u);

  HwyInt32 operator-() const {
    return hn::Neg(v);
  }

  HwyInt32& operator+=(HwyInt32 o) {
    v = hn::Add(v, o.v);
    return *this;
  }
  HwyInt32& operator-=(HwyInt32 o) {
    v = hn::Sub(v, o.v);
    return *this;
  }
  HwyInt32& operator*=(HwyInt32 o) {
    v = hn::Mul(v, o.v);
    return *this;
  }
  HwyInt32& operator&=(HwyInt32 o) {
    v = hn::And(v, o.v);
    return *this;
  }
  HwyInt32& operator|=(HwyInt32 o) {
    v = hn::Or(v, o.v);
    return *this;
  }

  // Shifts.
  HwyInt32 operator<<(int n) const {
    const HwyInt32Tag di;
    return hn::Shl(v, hn::BitCast(di, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n))));
  }
  HwyInt32 operator>>(int n) const {
    const HwyInt32Tag di;
    return hn::Shr(v, hn::BitCast(di, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n))));
  }
  HwyInt32& operator<<=(int n) {
    const HwyInt32Tag di;
    v = hn::Shl(v, hn::BitCast(di, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n))));
    return *this;
  }
  HwyInt32& operator>>=(int n) {
    const HwyInt32Tag di;
    v = hn::Shr(v, hn::BitCast(di, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n))));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend HwyInt32 operator+(HwyInt32 a, HwyInt32 b) {
    return hn::Add(a.v, b.v);
  }
  friend HwyInt32 operator-(HwyInt32 a, HwyInt32 b) {
    return hn::Sub(a.v, b.v);
  }
  friend HwyInt32 operator*(HwyInt32 a, HwyInt32 b) {
    return hn::Mul(a.v, b.v);
  }

  // Bitwise.
  friend HwyInt32 operator&(HwyInt32 a, HwyInt32 b) {
    return hn::And(a.v, b.v);
  }
  friend HwyInt32 operator|(HwyInt32 a, HwyInt32 b) {
    return hn::Or(a.v, b.v);
  }
  friend HwyInt32 operator^(HwyInt32 a, HwyInt32 b) {
    return hn::Xor(a.v, b.v);
  }

  friend HwyInt32 operator~(HwyInt32 a) {
    return hn::Not(a.v);
  }

  // Comparisons return HwyInt32 masks (via VecFromMask).
  friend HwyInt32 operator==(HwyInt32 a, HwyInt32 b) {
    const HwyInt32Tag di;
    return hn::VecFromMask(di, hn::Eq(a.v, b.v));
  }
  friend HwyInt32 operator!=(HwyInt32 a, HwyInt32 b) {
    const HwyInt32Tag di;
    return hn::VecFromMask(di, hn::Not(hn::Eq(a.v, b.v)));
  }
  friend HwyInt32 operator<(HwyInt32 a, HwyInt32 b) {
    const HwyInt32Tag di;
    return hn::VecFromMask(di, hn::Lt(a.v, b.v));
  }
  friend HwyInt32 operator>(HwyInt32 a, HwyInt32 b) {
    const HwyInt32Tag di;
    return hn::VecFromMask(di, hn::Gt(a.v, b.v));
  }

  friend HwyInt32 operator!(HwyInt32 a) {
    const HwyInt32Tag di;
    return hn::VecFromMask(di, hn::Eq(a.v, hn::Zero(di)));
  }
};

// N-wide uint32 SIMD wrapper around Highway Vec<RebindToUnsigned<ScalableTag<float>>>.
struct HwyUint32 {
  hn::Vec<HwyUint32Tag> v;

  HwyUint32() = default;
  HwyUint32(hn::Vec<HwyUint32Tag> vec) : v(vec) {}
  HwyUint32(uint32_t u) : v(hn::Set(HwyUint32Tag{}, u)) {}
  // Reinterpret from HwyInt32 (no-op).
  HwyUint32(HwyInt32 i) : v(hn::BitCast(HwyUint32Tag{}, i.v)) {}

  HwyUint32& operator+=(HwyUint32 o) {
    v = hn::Add(v, o.v);
    return *this;
  }
  HwyUint32& operator-=(HwyUint32 o) {
    v = hn::Sub(v, o.v);
    return *this;
  }
  HwyUint32& operator*=(HwyUint32 o) {
    v = hn::Mul(v, o.v);
    return *this;
  }
  HwyUint32& operator&=(HwyUint32 o) {
    v = hn::And(v, o.v);
    return *this;
  }
  HwyUint32& operator|=(HwyUint32 o) {
    v = hn::Or(v, o.v);
    return *this;
  }

  // Shifts.
  HwyUint32 operator<<(int n) const {
    return hn::Shl(v, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n)));
  }
  HwyUint32 operator>>(int n) const {
    return hn::Shr(v, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n)));
  }
  HwyUint32& operator<<=(int n) {
    v = hn::Shl(v, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n)));
    return *this;
  }
  HwyUint32& operator>>=(int n) {
    v = hn::Shr(v, hn::Set(HwyUint32Tag{}, static_cast<uint32_t>(n)));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend HwyUint32 operator+(HwyUint32 a, HwyUint32 b) {
    return hn::Add(a.v, b.v);
  }
  friend HwyUint32 operator-(HwyUint32 a, HwyUint32 b) {
    return hn::Sub(a.v, b.v);
  }
  friend HwyUint32 operator*(HwyUint32 a, HwyUint32 b) {
    return hn::Mul(a.v, b.v);
  }

  // Bitwise.
  friend HwyUint32 operator&(HwyUint32 a, HwyUint32 b) {
    return hn::And(a.v, b.v);
  }
  friend HwyUint32 operator|(HwyUint32 a, HwyUint32 b) {
    return hn::Or(a.v, b.v);
  }
  friend HwyUint32 operator^(HwyUint32 a, HwyUint32 b) {
    return hn::Xor(a.v, b.v);
  }

  friend HwyUint32 operator~(HwyUint32 a) {
    return hn::Not(a.v);
  }

  // Comparisons.
  friend HwyUint32 operator==(HwyUint32 a, HwyUint32 b) {
    const HwyUint32Tag du;
    return hn::VecFromMask(du, hn::Eq(a.v, b.v));
  }
  friend HwyUint32 operator!=(HwyUint32 a, HwyUint32 b) {
    const HwyUint32Tag du;
    return hn::VecFromMask(du, hn::Not(hn::Eq(a.v, b.v)));
  }
  friend HwyUint32 operator>(HwyUint32 a, HwyUint32 b) {
    const HwyUint32Tag du;
    return hn::VecFromMask(du, hn::Gt(a.v, b.v));
  }
  friend HwyUint32 operator<(HwyUint32 a, HwyUint32 b) {
    const HwyUint32Tag du;
    return hn::VecFromMask(du, hn::Lt(a.v, b.v));
  }

  friend HwyUint32 operator!(HwyUint32 a) {
    const HwyUint32Tag du;
    return hn::VecFromMask(du, hn::Eq(a.v, hn::Zero(du)));
  }
};

// --- Deferred inline definitions ---

// HwyFloat ↔ HwyInt32 conversion.
inline HwyFloat::HwyFloat(HwyInt32 i) : v(hn::ConvertTo(HwyFloatTag{}, i.v)) {}

// HwyInt32 ↔ HwyUint32 reinterpret (no-op).
inline HwyInt32::HwyInt32(HwyUint32 u) : v(hn::BitCast(HwyInt32Tag{}, u.v)) {}

// Map raw Highway vector to HwyFloat for SimdTypeFor.
template <>
struct SimdTypeFor<hn::Vec<HwyFloatTag>> {
  using type = HwyFloat;
};

// --- bit_cast specializations ---

template <>
inline HwyInt32 bit_cast<HwyInt32>(const HwyFloat& f) noexcept {
  return hn::BitCast(HwyInt32Tag{}, f.v);
}
template <>
inline HwyUint32 bit_cast<HwyUint32>(const HwyFloat& f) noexcept {
  return hn::BitCast(HwyUint32Tag{}, f.v);
}
template <>
inline HwyFloat bit_cast<HwyFloat>(const HwyInt32& i) noexcept {
  return hn::BitCast(HwyFloatTag{}, i.v);
}
template <>
inline HwyFloat bit_cast<HwyFloat>(const HwyUint32& u) noexcept {
  return hn::BitCast(HwyFloatTag{}, u.v);
}
template <>
inline HwyInt32 bit_cast<HwyInt32>(const HwyUint32& u) noexcept {
  return hn::BitCast(HwyInt32Tag{}, u.v);
}
template <>
inline HwyUint32 bit_cast<HwyUint32>(const HwyInt32& i) noexcept {
  return hn::BitCast(HwyUint32Tag{}, i.v);
}

// --- FloatTraits<HwyFloat> ---

template <>
struct FloatTraits<HwyFloat> {
  using IntType = HwyInt32;
  using UintType = HwyUint32;
  using BoolType = HwyFloat; // Float comparison masks (lane-wide)

  static constexpr uint32_t kOne = 0x3f800000;
  static constexpr float kMagic = 12582912.f;
  static constexpr bool kBoolIsMask = true;

  static DISPENSO_INLINE HwyFloat sqrt(HwyFloat x) {
    return hn::Sqrt(x.v);
  }

  // Highway always has FMA.
  static DISPENSO_INLINE HwyFloat fma(HwyFloat a, HwyFloat b, HwyFloat c) {
    return hn::MulAdd(a.v, b.v, c.v);
  }

  // conditional: select x where mask is true, y where false.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(HwyFloat mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(HwyInt32 mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg apply(HwyFloat mask, Arg x);

  static DISPENSO_INLINE HwyFloat min(HwyFloat a, HwyFloat b) {
    return hn::Min(a.v, b.v);
  }

  static DISPENSO_INLINE HwyFloat max(HwyFloat a, HwyFloat b) {
    return hn::Max(a.v, b.v);
  }
};

// conditional specializations (HwyFloat mask).
// MaskFromVec extracts MSB: all-ones → true, all-zeros → false.
template <>
inline HwyFloat FloatTraits<HwyFloat>::conditional(HwyFloat mask, HwyFloat x, HwyFloat y) {
  auto m = hn::MaskFromVec(mask.v);
  return hn::IfThenElse(m, x.v, y.v);
}
template <>
inline HwyInt32 FloatTraits<HwyFloat>::conditional(HwyFloat mask, HwyInt32 x, HwyInt32 y) {
  const HwyInt32Tag di;
  auto imask = hn::BitCast(di, mask.v);
  auto m = hn::MaskFromVec(imask);
  return hn::IfThenElse(m, x.v, y.v);
}
template <>
inline HwyUint32 FloatTraits<HwyFloat>::conditional(HwyFloat mask, HwyUint32 x, HwyUint32 y) {
  const HwyUint32Tag du;
  auto umask = hn::BitCast(du, mask.v);
  auto m = hn::MaskFromVec(umask);
  return hn::IfThenElse(m, x.v, y.v);
}

// conditional specializations (HwyInt32 lane-wide mask).
template <>
inline HwyFloat FloatTraits<HwyFloat>::conditional(HwyInt32 mask, HwyFloat x, HwyFloat y) {
  const HwyFloatTag d;
  auto fmask = hn::BitCast(d, mask.v);
  auto m = hn::MaskFromVec(fmask);
  return hn::IfThenElse(m, x.v, y.v);
}
template <>
inline HwyInt32 FloatTraits<HwyFloat>::conditional(HwyInt32 mask, HwyInt32 x, HwyInt32 y) {
  auto m = hn::MaskFromVec(mask.v);
  return hn::IfThenElse(m, x.v, y.v);
}
template <>
inline HwyUint32 FloatTraits<HwyFloat>::conditional(HwyInt32 mask, HwyUint32 x, HwyUint32 y) {
  const HwyUint32Tag du;
  auto umask = hn::BitCast(du, mask.v);
  auto m = hn::MaskFromVec(umask);
  return hn::IfThenElse(m, x.v, y.v);
}

// apply: mask & x (bitwise AND).
template <>
inline HwyFloat FloatTraits<HwyFloat>::apply(HwyFloat mask, HwyFloat x) {
  const HwyFloatTag d;
  const HwyInt32Tag di;
  return hn::BitCast(d, hn::And(hn::BitCast(di, mask.v), hn::BitCast(di, x.v)));
}
template <>
inline HwyInt32 FloatTraits<HwyFloat>::apply(HwyFloat mask, HwyInt32 x) {
  const HwyInt32Tag di;
  return hn::And(hn::BitCast(di, mask.v), x.v);
}
template <>
inline HwyUint32 FloatTraits<HwyFloat>::apply(HwyFloat mask, HwyUint32 x) {
  const HwyUint32Tag du;
  return hn::And(hn::BitCast(du, mask.v), x.v);
}

template <>
struct FloatTraits<HwyInt32> {
  using IntType = HwyInt32;
};

template <>
struct FloatTraits<HwyUint32> {
  using IntType = HwyUint32;
};

// --- Util function overloads for Highway types ---

DISPENSO_INLINE HwyFloat floor_small(HwyFloat x) {
  return hn::Floor(x.v);
}

DISPENSO_INLINE HwyInt32 convert_to_int(HwyFloat f) {
  const HwyInt32Tag di;
  // Round to nearest even, then convert to int.
  auto rounded = hn::Round(f.v);
  auto converted = hn::ConvertTo(di, rounded);
  // Mask non-normals to 0 to avoid undefined behavior.
  HwyInt32 fi = bit_cast<HwyInt32>(f);
  HwyInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & HwyInt32(converted);
}

template <>
DISPENSO_INLINE HwyFloat min<HwyFloat>(HwyFloat x, HwyFloat mn) {
  // If x is NaN, return mn. Explicit NaN handling for portability.
  auto is_num = hn::Eq(x.v, x.v); // true for non-NaN
  auto result = hn::Min(x.v, mn.v);
  return hn::IfThenElse(is_num, result, mn.v);
}

template <>
DISPENSO_INLINE HwyFloat clamp_allow_nan<HwyFloat>(HwyFloat x, HwyFloat mn, HwyFloat mx) {
  // NaN propagates: if x is NaN, result is NaN.
  auto is_nan = hn::Not(hn::Eq(x.v, x.v));
  auto clamped = hn::Max(mn.v, hn::Min(mx.v, x.v));
  return hn::IfThenElse(is_nan, x.v, clamped);
}

template <>
DISPENSO_INLINE HwyFloat clamp_no_nan<HwyFloat>(HwyFloat x, HwyFloat mn, HwyFloat mx) {
  // NaN suppressed: if x is NaN, result is mn.
  auto is_nan = hn::Not(hn::Eq(x.v, x.v));
  auto clamped = hn::Max(mn.v, hn::Min(x.v, mx.v));
  return hn::IfThenElse(is_nan, mn.v, clamped);
}

template <>
DISPENSO_INLINE HwyFloat gather<HwyFloat>(const float* table, HwyInt32 index) {
  const HwyFloatTag d;
  return hn::GatherIndex(d, table, index.v);
}

DISPENSO_INLINE HwyInt32 int_div_by_3(HwyInt32 i) {
  // Scalar fallback: extract, divide, reload.
  const HwyInt32Tag di;
  constexpr size_t kMaxLanes = HWY_MAX_BYTES / sizeof(int32_t);
  HWY_ALIGN int32_t in[kMaxLanes];
  HWY_ALIGN int32_t out[kMaxLanes];
  const size_t N = hn::Lanes(di);
  hn::StoreU(i.v, di, in);
  for (size_t j = 0; j < N; ++j) {
    out[j] = static_cast<int32_t>((uint64_t(in[j]) * 0x55555556) >> 32);
  }
  return hn::Load(di, out);
}

// nonnormal/nonnormalOrZero: return HwyInt32 masks.
DISPENSO_INLINE HwyInt32 nonnormal(HwyInt32 i) {
  return (i & 0x7f800000) == 0x7f800000;
}

DISPENSO_INLINE HwyInt32 nonnormalOrZero(HwyInt32 i) {
  auto m = i & 0x7f800000;
  return (m == 0x7f800000) | (m == 0);
}

DISPENSO_INLINE HwyInt32 nonnormal(HwyFloat f) {
  return nonnormal(bit_cast<HwyInt32>(f));
}

DISPENSO_INLINE HwyFloat signof(HwyFloat x) {
  HwyUint32 xi = bit_cast<HwyUint32>(x);
  return bit_cast<HwyFloat>((xi & 0x80000000u) | FloatTraits<HwyFloat>::kOne);
}

DISPENSO_INLINE HwyInt32 signofi(HwyInt32 i) {
  return HwyInt32(1) - (HwyInt32(2) & (i < HwyInt32(0)));
}

// nbool_as_one: 0 if mask is true (all-ones), 1 if false (all-zeros).
template <>
DISPENSO_INLINE HwyFloat nbool_as_one<HwyFloat, HwyFloat>(HwyFloat b) {
  const HwyFloatTag d;
  const HwyInt32Tag di;
  auto bi = hn::BitCast(di, b.v);
  auto one_bits = hn::Set(di, 0x3f800000);
  return hn::BitCast(d, hn::AndNot(bi, one_bits));
}

template <>
DISPENSO_INLINE HwyInt32 nbool_as_one<HwyInt32, HwyFloat>(HwyFloat b) {
  const HwyInt32Tag di;
  auto bi = hn::BitCast(di, b.v);
  return hn::AndNot(bi, hn::Set(di, 1));
}

template <>
DISPENSO_INLINE HwyInt32 nbool_as_one<HwyInt32, HwyInt32>(HwyInt32 b) {
  const HwyInt32Tag di;
  return hn::AndNot(b.v, hn::Set(di, 1));
}

// bool_as_one: 1 if mask is true, 0 if false.
template <>
DISPENSO_INLINE HwyFloat bool_as_one<HwyFloat, HwyFloat>(HwyFloat b) {
  const HwyFloatTag d;
  const HwyInt32Tag di;
  auto bi = hn::BitCast(di, b.v);
  return hn::BitCast(d, hn::And(bi, hn::Set(di, 0x3f800000)));
}

template <>
DISPENSO_INLINE HwyInt32 bool_as_one<HwyInt32, HwyFloat>(HwyFloat b) {
  const HwyInt32Tag di;
  auto bi = hn::BitCast(di, b.v);
  return hn::And(bi, hn::Set(di, 1));
}

// bool_as_mask: for SIMD masks, identity (already a mask).
template <>
DISPENSO_INLINE HwyInt32 bool_as_mask<HwyInt32, HwyFloat>(HwyFloat b) {
  return hn::BitCast(HwyInt32Tag{}, b.v);
}

template <>
DISPENSO_INLINE HwyInt32 bool_as_mask<HwyInt32, HwyInt32>(HwyInt32 b) {
  return b;
}

template <>
DISPENSO_INLINE HwyInt32 bool_as_mask<HwyInt32, HwyUint32>(HwyUint32 b) {
  return hn::BitCast(HwyInt32Tag{}, b.v);
}

template <>
DISPENSO_INLINE HwyUint32 bool_as_mask<HwyUint32, HwyFloat>(HwyFloat b) {
  return hn::BitCast(HwyUint32Tag{}, b.v);
}

template <>
DISPENSO_INLINE HwyUint32 bool_as_mask<HwyUint32, HwyUint32>(HwyUint32 b) {
  return b;
}

template <>
DISPENSO_INLINE HwyUint32 bool_as_mask<HwyUint32, HwyInt32>(HwyInt32 b) {
  return hn::BitCast(HwyUint32Tag{}, b.v);
}

} // namespace fast_math
} // namespace dispenso

#endif // __has_include("hwy/highway.h")
