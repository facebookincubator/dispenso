/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cstdint>

#include "float_traits.h"
#include "util.h"

namespace dispenso {
namespace fast_math {

struct NeonInt32;
struct NeonUint32;

// 4-wide float SIMD wrapper around float32x4_t (AArch64 NEON).
struct NeonFloat {
  float32x4_t v;

  NeonFloat() = default;
  NeonFloat(float32x4_t vec) : v(vec) {}
  // Implicit broadcast from scalar — required for polynomial constant propagation.
  NeonFloat(float f) : v(vdupq_n_f32(f)) {}
  // Explicit int→float conversion for (Flt)intExpr patterns.
  explicit NeonFloat(NeonInt32 i);

  NeonFloat operator-() const {
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000)));
  }

  NeonFloat& operator+=(NeonFloat o) {
    v = vaddq_f32(v, o.v);
    return *this;
  }
  NeonFloat& operator-=(NeonFloat o) {
    v = vsubq_f32(v, o.v);
    return *this;
  }
  NeonFloat& operator*=(NeonFloat o) {
    v = vmulq_f32(v, o.v);
    return *this;
  }
  NeonFloat& operator&=(NeonFloat o) {
    v = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v), vreinterpretq_u32_f32(o.v)));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend NeonFloat operator+(NeonFloat a, NeonFloat b) {
    return vaddq_f32(a.v, b.v);
  }
  friend NeonFloat operator-(NeonFloat a, NeonFloat b) {
    return vsubq_f32(a.v, b.v);
  }
  friend NeonFloat operator*(NeonFloat a, NeonFloat b) {
    return vmulq_f32(a.v, b.v);
  }
  friend NeonFloat operator/(NeonFloat a, NeonFloat b) {
    return vdivq_f32(a.v, b.v);
  }

  // Comparisons return NeonFloat masks (all-ones or all-zeros per lane).
  // NEON comparisons produce uint32x4_t; reinterpret to float32x4_t.
  friend NeonFloat operator<(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vcltq_f32(a.v, b.v));
  }
  friend NeonFloat operator>(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vcgtq_f32(a.v, b.v));
  }
  friend NeonFloat operator<=(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vcleq_f32(a.v, b.v));
  }
  friend NeonFloat operator>=(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vcgeq_f32(a.v, b.v));
  }
  friend NeonFloat operator==(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vceqq_f32(a.v, b.v));
  }
  friend NeonFloat operator!=(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a.v, b.v)));
  }

  // Logical NOT of a comparison mask.
  friend NeonFloat operator!(NeonFloat a) {
    return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a.v)));
  }

  // Bitwise ops on float masks.
  friend NeonFloat operator&(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a.v), vreinterpretq_u32_f32(b.v)));
  }
  friend NeonFloat operator|(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a.v), vreinterpretq_u32_f32(b.v)));
  }
  friend NeonFloat operator^(NeonFloat a, NeonFloat b) {
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a.v), vreinterpretq_u32_f32(b.v)));
  }
};

// 4-wide int32 SIMD wrapper around int32x4_t.
struct NeonInt32 {
  int32x4_t v;

  NeonInt32() = default;
  NeonInt32(int32x4_t vec) : v(vec) {}
  NeonInt32(int32_t i) : v(vdupq_n_s32(i)) {}
  // Reinterpret from NeonUint32 (no-op).
  NeonInt32(NeonUint32 u);

  NeonInt32 operator-() const {
    return vnegq_s32(v);
  }

  NeonInt32& operator+=(NeonInt32 o) {
    v = vaddq_s32(v, o.v);
    return *this;
  }
  NeonInt32& operator-=(NeonInt32 o) {
    v = vsubq_s32(v, o.v);
    return *this;
  }
  NeonInt32& operator*=(NeonInt32 o) {
    v = vmulq_s32(v, o.v);
    return *this;
  }
  NeonInt32& operator&=(NeonInt32 o) {
    v = vandq_s32(v, o.v);
    return *this;
  }
  NeonInt32& operator|=(NeonInt32 o) {
    v = vorrq_s32(v, o.v);
    return *this;
  }

  // Shifts (NEON uses vshlq with signed shift amount: positive=left, negative=right).
  NeonInt32 operator<<(int n) const {
    return vshlq_s32(v, vdupq_n_s32(n));
  }
  NeonInt32 operator>>(int n) const {
    return vshlq_s32(v, vdupq_n_s32(-n)); // Arithmetic right shift
  }
  NeonInt32& operator<<=(int n) {
    v = vshlq_s32(v, vdupq_n_s32(n));
    return *this;
  }
  NeonInt32& operator>>=(int n) {
    v = vshlq_s32(v, vdupq_n_s32(-n));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend NeonInt32 operator+(NeonInt32 a, NeonInt32 b) {
    return vaddq_s32(a.v, b.v);
  }
  friend NeonInt32 operator-(NeonInt32 a, NeonInt32 b) {
    return vsubq_s32(a.v, b.v);
  }
  friend NeonInt32 operator*(NeonInt32 a, NeonInt32 b) {
    return vmulq_s32(a.v, b.v);
  }

  // Bitwise.
  friend NeonInt32 operator&(NeonInt32 a, NeonInt32 b) {
    return vandq_s32(a.v, b.v);
  }
  friend NeonInt32 operator|(NeonInt32 a, NeonInt32 b) {
    return vorrq_s32(a.v, b.v);
  }
  friend NeonInt32 operator^(NeonInt32 a, NeonInt32 b) {
    return veorq_s32(a.v, b.v);
  }

  friend NeonInt32 operator~(NeonInt32 a) {
    return vmvnq_s32(a.v);
  }

  // Comparisons return NeonInt32 masks (reinterpreted from uint32x4_t).
  friend NeonInt32 operator==(NeonInt32 a, NeonInt32 b) {
    return vreinterpretq_s32_u32(vceqq_s32(a.v, b.v));
  }
  friend NeonInt32 operator!=(NeonInt32 a, NeonInt32 b) {
    return vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a.v, b.v)));
  }
  friend NeonInt32 operator<(NeonInt32 a, NeonInt32 b) {
    return vreinterpretq_s32_u32(vcltq_s32(a.v, b.v));
  }
  friend NeonInt32 operator>(NeonInt32 a, NeonInt32 b) {
    return vreinterpretq_s32_u32(vcgtq_s32(a.v, b.v));
  }

  friend NeonInt32 operator!(NeonInt32 a) {
    return vreinterpretq_s32_u32(vceqq_s32(a.v, vdupq_n_s32(0)));
  }
};

// 4-wide uint32 SIMD wrapper around uint32x4_t.
struct NeonUint32 {
  uint32x4_t v;

  NeonUint32() = default;
  NeonUint32(uint32x4_t vec) : v(vec) {}
  NeonUint32(uint32_t u) : v(vdupq_n_u32(u)) {}
  // Reinterpret from NeonInt32 (no-op).
  NeonUint32(NeonInt32 i) : v(vreinterpretq_u32_s32(i.v)) {}

  NeonUint32& operator+=(NeonUint32 o) {
    v = vaddq_u32(v, o.v);
    return *this;
  }
  NeonUint32& operator-=(NeonUint32 o) {
    v = vsubq_u32(v, o.v);
    return *this;
  }
  NeonUint32& operator*=(NeonUint32 o) {
    v = vmulq_u32(v, o.v);
    return *this;
  }
  NeonUint32& operator&=(NeonUint32 o) {
    v = vandq_u32(v, o.v);
    return *this;
  }
  NeonUint32& operator|=(NeonUint32 o) {
    v = vorrq_u32(v, o.v);
    return *this;
  }

  // Shifts.
  NeonUint32 operator<<(int n) const {
    return vshlq_u32(v, vdupq_n_s32(n));
  }
  NeonUint32 operator>>(int n) const {
    return vshlq_u32(v, vdupq_n_s32(-n)); // Logical right shift
  }
  NeonUint32& operator<<=(int n) {
    v = vshlq_u32(v, vdupq_n_s32(n));
    return *this;
  }
  NeonUint32& operator>>=(int n) {
    v = vshlq_u32(v, vdupq_n_s32(-n));
    return *this;
  }

  // Arithmetic (non-member friends for symmetric implicit conversions).
  friend NeonUint32 operator+(NeonUint32 a, NeonUint32 b) {
    return vaddq_u32(a.v, b.v);
  }
  friend NeonUint32 operator-(NeonUint32 a, NeonUint32 b) {
    return vsubq_u32(a.v, b.v);
  }
  friend NeonUint32 operator*(NeonUint32 a, NeonUint32 b) {
    return vmulq_u32(a.v, b.v);
  }

  // Bitwise.
  friend NeonUint32 operator&(NeonUint32 a, NeonUint32 b) {
    return vandq_u32(a.v, b.v);
  }
  friend NeonUint32 operator|(NeonUint32 a, NeonUint32 b) {
    return vorrq_u32(a.v, b.v);
  }
  friend NeonUint32 operator^(NeonUint32 a, NeonUint32 b) {
    return veorq_u32(a.v, b.v);
  }

  friend NeonUint32 operator~(NeonUint32 a) {
    return vmvnq_u32(a.v);
  }

  // NEON has native unsigned comparisons.
  friend NeonUint32 operator==(NeonUint32 a, NeonUint32 b) {
    return vceqq_u32(a.v, b.v);
  }
  friend NeonUint32 operator!=(NeonUint32 a, NeonUint32 b) {
    return vmvnq_u32(vceqq_u32(a.v, b.v));
  }
  friend NeonUint32 operator>(NeonUint32 a, NeonUint32 b) {
    return vcgtq_u32(a.v, b.v);
  }
  friend NeonUint32 operator<(NeonUint32 a, NeonUint32 b) {
    return vcltq_u32(a.v, b.v);
  }

  friend NeonUint32 operator!(NeonUint32 a) {
    return vceqq_u32(a.v, vdupq_n_u32(0));
  }
};

// --- Deferred inline definitions ---

// NeonFloat ↔ NeonInt32 conversion.
inline NeonFloat::NeonFloat(NeonInt32 i) : v(vcvtq_f32_s32(i.v)) {}

// NeonInt32 ↔ NeonUint32 reinterpret (no-op).
inline NeonInt32::NeonInt32(NeonUint32 u) : v(vreinterpretq_s32_u32(u.v)) {}

// Map raw float32x4_t to NeonFloat for SimdTypeFor.
template <>
struct SimdTypeFor<float32x4_t> {
  using type = NeonFloat;
};

// --- bit_cast specializations ---

template <>
inline NeonInt32 bit_cast<NeonInt32>(const NeonFloat& f) noexcept {
  return vreinterpretq_s32_f32(f.v);
}
template <>
inline NeonUint32 bit_cast<NeonUint32>(const NeonFloat& f) noexcept {
  return vreinterpretq_u32_f32(f.v);
}
template <>
inline NeonFloat bit_cast<NeonFloat>(const NeonInt32& i) noexcept {
  return vreinterpretq_f32_s32(i.v);
}
template <>
inline NeonFloat bit_cast<NeonFloat>(const NeonUint32& u) noexcept {
  return vreinterpretq_f32_u32(u.v);
}
template <>
inline NeonInt32 bit_cast<NeonInt32>(const NeonUint32& u) noexcept {
  return vreinterpretq_s32_u32(u.v);
}
template <>
inline NeonUint32 bit_cast<NeonUint32>(const NeonInt32& i) noexcept {
  return vreinterpretq_u32_s32(i.v);
}

// --- FloatTraits<NeonFloat> ---

template <>
struct FloatTraits<NeonFloat> {
  using IntType = NeonInt32;
  using UintType = NeonUint32;
  using BoolType = NeonFloat; // Float comparison masks (lane-wide, like SSE)

  static constexpr uint32_t kOne = 0x3f800000;
  static constexpr float kMagic = 12582912.f;
  static constexpr bool kBoolIsMask = true;

  static DISPENSO_INLINE NeonFloat sqrt(NeonFloat x) {
    return vsqrtq_f32(x.v);
  }

  // AArch64 always has FMA.
  static DISPENSO_INLINE NeonFloat fma(NeonFloat a, NeonFloat b, NeonFloat c) {
    return vfmaq_f32(c.v, a.v, b.v); // vfmaq_f32(acc, a, b) = acc + a*b
  }

  // conditional: select x where mask is true, y where false.
  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(NeonFloat mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg conditional(NeonInt32 mask, Arg x, Arg y);

  template <typename Arg>
  static DISPENSO_INLINE Arg apply(NeonFloat mask, Arg x);

  static DISPENSO_INLINE NeonFloat min(NeonFloat a, NeonFloat b) {
    return vminnmq_f32(a.v, b.v);
  }

  static DISPENSO_INLINE NeonFloat max(NeonFloat a, NeonFloat b) {
    return vmaxnmq_f32(a.v, b.v);
  }
};

// conditional specializations (NeonFloat mask).
// vbslq selects from first operand where mask bits are 1, second where 0.
template <>
inline NeonFloat FloatTraits<NeonFloat>::conditional(NeonFloat mask, NeonFloat x, NeonFloat y) {
  return vbslq_f32(vreinterpretq_u32_f32(mask.v), x.v, y.v);
}
template <>
inline NeonInt32 FloatTraits<NeonFloat>::conditional(NeonFloat mask, NeonInt32 x, NeonInt32 y) {
  return vreinterpretq_s32_u32(vbslq_u32(
      vreinterpretq_u32_f32(mask.v), vreinterpretq_u32_s32(x.v), vreinterpretq_u32_s32(y.v)));
}
template <>
inline NeonUint32 FloatTraits<NeonFloat>::conditional(NeonFloat mask, NeonUint32 x, NeonUint32 y) {
  return vbslq_u32(vreinterpretq_u32_f32(mask.v), x.v, y.v);
}

// conditional specializations (NeonInt32 lane-wide mask).
template <>
inline NeonFloat FloatTraits<NeonFloat>::conditional(NeonInt32 mask, NeonFloat x, NeonFloat y) {
  return vbslq_f32(vreinterpretq_u32_s32(mask.v), x.v, y.v);
}
template <>
inline NeonInt32 FloatTraits<NeonFloat>::conditional(NeonInt32 mask, NeonInt32 x, NeonInt32 y) {
  return vreinterpretq_s32_u32(vbslq_u32(
      vreinterpretq_u32_s32(mask.v), vreinterpretq_u32_s32(x.v), vreinterpretq_u32_s32(y.v)));
}
template <>
inline NeonUint32 FloatTraits<NeonFloat>::conditional(NeonInt32 mask, NeonUint32 x, NeonUint32 y) {
  return vbslq_u32(vreinterpretq_u32_s32(mask.v), x.v, y.v);
}

// apply: mask & x (bitwise AND).
template <>
inline NeonFloat FloatTraits<NeonFloat>::apply(NeonFloat mask, NeonFloat x) {
  return vreinterpretq_f32_u32(
      vandq_u32(vreinterpretq_u32_f32(mask.v), vreinterpretq_u32_f32(x.v)));
}
template <>
inline NeonInt32 FloatTraits<NeonFloat>::apply(NeonFloat mask, NeonInt32 x) {
  return vreinterpretq_s32_u32(
      vandq_u32(vreinterpretq_u32_f32(mask.v), vreinterpretq_u32_s32(x.v)));
}
template <>
inline NeonUint32 FloatTraits<NeonFloat>::apply(NeonFloat mask, NeonUint32 x) {
  return vandq_u32(vreinterpretq_u32_f32(mask.v), x.v);
}

template <>
struct FloatTraits<NeonInt32> {
  using IntType = NeonInt32;
};

template <>
struct FloatTraits<NeonUint32> {
  using IntType = NeonUint32;
};

// --- Util function overloads for NEON types ---

DISPENSO_INLINE NeonFloat floor_small(NeonFloat x) {
  return vrndmq_f32(x.v);
}

DISPENSO_INLINE NeonInt32 convert_to_int_trunc(NeonFloat f) {
  return vcvtq_s32_f32(f.v);
}

DISPENSO_INLINE NeonInt32 convert_to_int_trunc_safe(NeonFloat f) {
  NeonInt32 fi = bit_cast<NeonInt32>(f);
  NeonInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & NeonInt32(vcvtq_s32_f32(f.v));
}

DISPENSO_INLINE NeonInt32 convert_to_int(NeonFloat f) {
  // vcvtnq_s32_f32 uses round-to-nearest-even.
  // Mask non-normals to 0 to avoid undefined behavior.
  NeonInt32 fi = bit_cast<NeonInt32>(f);
  NeonInt32 norm = (fi & 0x7f800000) != 0x7f800000;
  return norm & NeonInt32(vcvtnq_s32_f32(f.v));
}

template <>
DISPENSO_INLINE NeonFloat min<NeonFloat>(NeonFloat x, NeonFloat mn) {
  // vminnmq_f32: if x is NaN, returns mn (the number). NaN-suppressing.
  return vminnmq_f32(x.v, mn.v);
}

template <>
DISPENSO_INLINE NeonFloat clamp_allow_nan<NeonFloat>(NeonFloat x, NeonFloat mn, NeonFloat mx) {
  // Use NaN-propagating min/max (FMIN/FMAX) so NaN passes through.
  return vmaxq_f32(mn.v, vminq_f32(mx.v, x.v));
}

template <>
DISPENSO_INLINE NeonFloat clamp_no_nan<NeonFloat>(NeonFloat x, NeonFloat mn, NeonFloat mx) {
  // Use NaN-suppressing min/max (FMINNM/FMAXNM) so NaN is replaced.
  return vmaxnmq_f32(mn.v, vminnmq_f32(x.v, mx.v));
}

template <>
DISPENSO_INLINE NeonFloat gather<NeonFloat>(const float* table, NeonInt32 index) {
  // No NEON gather instruction; extract indices and load scalar.
  alignas(16) int32_t idx[4];
  vst1q_s32(idx, index.v);
  float vals[4] = {table[idx[0]], table[idx[1]], table[idx[2]], table[idx[3]]};
  return vld1q_f32(vals);
}

DISPENSO_INLINE NeonInt32 int_div_by_3(NeonInt32 i) {
  // Multiply each lane by 0x55555556 and take the high 32 bits.
  // Process low (lanes 0,1) and high (lanes 2,3) halves separately.
  uint32x4_t ui = vreinterpretq_u32_s32(i.v);
  uint32x2_t mul = vdup_n_u32(0x55555556);
  // Low lanes (0, 1).
  uint64x2_t lo_prod = vmull_u32(vget_low_u32(ui), mul);
  uint32x2_t lo_result = vshrn_n_u64(lo_prod, 32);
  // High lanes (2, 3).
  uint64x2_t hi_prod = vmull_u32(vget_high_u32(ui), mul);
  uint32x2_t hi_result = vshrn_n_u64(hi_prod, 32);
  return vreinterpretq_s32_u32(vcombine_u32(lo_result, hi_result));
}

// nonnormal/nonnormalOrZero: return NeonInt32 masks for NEON types.
DISPENSO_INLINE NeonInt32 nonnormal(NeonInt32 i) {
  return (i & 0x7f800000) == 0x7f800000;
}

DISPENSO_INLINE NeonInt32 nonnormalOrZero(NeonInt32 i) {
  auto m = i & 0x7f800000;
  return (m == 0x7f800000) | (m == 0);
}

DISPENSO_INLINE NeonInt32 nonnormal(NeonFloat f) {
  return nonnormal(bit_cast<NeonInt32>(f));
}

DISPENSO_INLINE NeonFloat signof(NeonFloat x) {
  NeonUint32 xi = bit_cast<NeonUint32>(x);
  return bit_cast<NeonFloat>((xi & 0x80000000u) | FloatTraits<NeonFloat>::kOne);
}

DISPENSO_INLINE NeonInt32 signofi(NeonInt32 i) {
  return NeonInt32(1) - (NeonInt32(2) & (i < NeonInt32(0)));
}

// nbool_as_one: 0 if mask is true (all-ones), 1 if false (all-zeros).
template <>
DISPENSO_INLINE NeonFloat nbool_as_one<NeonFloat, NeonFloat>(NeonFloat b) {
  // ~mask & 1.0f bits
  return bit_cast<NeonFloat>(NeonInt32(
      vreinterpretq_s32_u32(vbicq_u32(vdupq_n_u32(0x3f800000), vreinterpretq_u32_f32(b.v)))));
}

template <>
DISPENSO_INLINE NeonInt32 nbool_as_one<NeonInt32, NeonFloat>(NeonFloat b) {
  return vreinterpretq_s32_u32(vbicq_u32(vdupq_n_u32(1), vreinterpretq_u32_f32(b.v)));
}

template <>
DISPENSO_INLINE NeonInt32 nbool_as_one<NeonInt32, NeonInt32>(NeonInt32 b) {
  return vreinterpretq_s32_u32(vbicq_u32(vdupq_n_u32(1), vreinterpretq_u32_s32(b.v)));
}

// bool_as_one: 1 if mask is true, 0 if false.
template <>
DISPENSO_INLINE NeonFloat bool_as_one<NeonFloat, NeonFloat>(NeonFloat b) {
  return bit_cast<NeonFloat>(NeonInt32(
      vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_f32(b.v), vdupq_n_u32(0x3f800000)))));
}

template <>
DISPENSO_INLINE NeonInt32 bool_as_one<NeonInt32, NeonFloat>(NeonFloat b) {
  return vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_f32(b.v), vdupq_n_u32(1)));
}

// bool_as_mask: for SIMD masks, identity (already a mask).
template <>
DISPENSO_INLINE NeonInt32 bool_as_mask<NeonInt32, NeonFloat>(NeonFloat b) {
  return vreinterpretq_s32_f32(b.v);
}

template <>
DISPENSO_INLINE NeonInt32 bool_as_mask<NeonInt32, NeonInt32>(NeonInt32 b) {
  return b;
}

template <>
DISPENSO_INLINE NeonInt32 bool_as_mask<NeonInt32, NeonUint32>(NeonUint32 b) {
  return vreinterpretq_s32_u32(b.v);
}

template <>
DISPENSO_INLINE NeonUint32 bool_as_mask<NeonUint32, NeonFloat>(NeonFloat b) {
  return vreinterpretq_u32_f32(b.v);
}

template <>
DISPENSO_INLINE NeonUint32 bool_as_mask<NeonUint32, NeonUint32>(NeonUint32 b) {
  return b;
}

template <>
DISPENSO_INLINE NeonUint32 bool_as_mask<NeonUint32, NeonInt32>(NeonInt32 b) {
  return vreinterpretq_u32_s32(b.v);
}

} // namespace fast_math
} // namespace dispenso

#endif // defined(__aarch64__)
