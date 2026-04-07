/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// DoubleVec<Flt>: a pair of double-precision vectors covering all lanes of Flt.
//
// Provides the minimal set of operations needed for the tableless
// double-precision pow core: widen, narrow, arithmetic, FMA, and
// exp2 range-reduction helpers.
//
// Specialized for each SIMD backend alongside its float type:
//   float     → double (scalar)
//   SseFloat  → __m128d lo, hi
//   AvxFloat  → __m256d lo, hi
//   Avx512Float → __m512d lo, hi
//   NeonFloat → float64x2_t lo, hi
//   HwyFloat  → Vec<Repartition<double, FloatTag>> lo, hi

#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

#include <dispenso/fast_math/util.h>

namespace dispenso {
namespace fast_math {
namespace detail {

// Primary template — undefined; each backend specializes.
template <typename Flt>
struct DoubleVec;

// ---------------------------------------------------------------------------
// Scalar: DoubleVec<float> is just a double.
// ---------------------------------------------------------------------------
template <>
struct DoubleVec<float> {
  double v;

  DoubleVec() = default;
  explicit DoubleVec(double d) : v(d) {}

  static DISPENSO_INLINE DoubleVec from_float(float f) {
    return DoubleVec{static_cast<double>(f)};
  }

  // Gather: load table[index] as a double.
  static DISPENSO_INLINE DoubleVec gather(const double* base, int32_t index) {
    return DoubleVec{base[index]};
  }

  DISPENSO_INLINE float to_float() const {
    return static_cast<float>(v);
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return DoubleVec{a.v + b.v};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return DoubleVec{a.v - b.v};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return DoubleVec{a.v * b.v};
  }

  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
    return DoubleVec{std::fma(a.v, b.v, c.v)};
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return DoubleVec{x.v < low.v ? low.v : (x.v > high.v ? high.v : x.v)};
  }
};

// exp2 range reduction: free function, defined after DoubleVec is complete.
// Splits x into integer n and fractional r ∈ [-0.5, 0.5].
// Returns {r, scale} where scale = 2^n.
// Boundary cases (from ylogx clamped to [-1022, 1023]):
//   n = 1024 → scale = +inf (0x7FF0...), result overflows to +inf — correct.
//   n = -1023 → scale = 0.0, result underflows to 0 — correct.
DISPENSO_INLINE std::pair<DoubleVec<float>, DoubleVec<float>> exp2_split(DoubleVec<float> x) {
  using DV = DoubleVec<float>;
  constexpr double kShift = 0x1.8p+52;
  constexpr uint64_t kShiftBits = 0x4338000000000000ULL;

  double shifted = x.v + kShift;
  uint64_t ki = bit_cast<uint64_t>(shifted);
  double nd = shifted - kShift;
  double r = x.v - nd;

  int64_t n = static_cast<int64_t>(ki - kShiftBits);
  double scale = bit_cast<double>(static_cast<uint64_t>(n + 1023) << 52);

  return {DV{r}, DV{scale}};
}

// ---------------------------------------------------------------------------
// SSE: DoubleVec<SseFloat> holds two __m128d (4 float lanes → 2×2 doubles).
// ---------------------------------------------------------------------------
#if defined(__SSE4_1__)

template <>
struct DoubleVec<SseFloat> {
  __m128d lo, hi; // lo = lanes 0,1; hi = lanes 2,3

  DoubleVec() = default;
  DoubleVec(__m128d l, __m128d h) : lo(l), hi(h) {}
  explicit DoubleVec(double d) : lo(_mm_set1_pd(d)), hi(_mm_set1_pd(d)) {}

  static DISPENSO_INLINE DoubleVec from_float(SseFloat f) {
    return {_mm_cvtps_pd(f.v), _mm_cvtps_pd(_mm_movehl_ps(f.v, f.v))};
  }

  // Gather: load base[idx[lane]] for each of 4 lanes via scalar lookups.
  // Used for small tables that fit in L1 (e.g. 16-entry pow log2 table).
  static DISPENSO_INLINE DoubleVec gather(const double* base, SseInt32 idx) {
    int32_t i0 = _mm_extract_epi32(idx.v, 0);
    int32_t i1 = _mm_extract_epi32(idx.v, 1);
    int32_t i2 = _mm_extract_epi32(idx.v, 2);
    int32_t i3 = _mm_extract_epi32(idx.v, 3);
    return {_mm_set_pd(base[i1], base[i0]), _mm_set_pd(base[i3], base[i2])};
  }

  DISPENSO_INLINE SseFloat to_float() const {
    __m128 lo_f = _mm_cvtpd_ps(lo);
    __m128 hi_f = _mm_cvtpd_ps(hi);
    return SseFloat{_mm_movelh_ps(lo_f, hi_f)};
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return {_mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return {_mm_sub_pd(a.lo, b.lo), _mm_sub_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return {_mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi)};
  }

  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
#if defined(__FMA__)
    return {_mm_fmadd_pd(a.lo, b.lo, c.lo), _mm_fmadd_pd(a.hi, b.hi, c.hi)};
#else
    return {_mm_add_pd(_mm_mul_pd(a.lo, b.lo), c.lo), _mm_add_pd(_mm_mul_pd(a.hi, b.hi), c.hi)};
#endif
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return {
        _mm_min_pd(_mm_max_pd(x.lo, low.lo), high.lo),
        _mm_min_pd(_mm_max_pd(x.hi, low.hi), high.hi)};
  }
};

DISPENSO_INLINE std::pair<DoubleVec<SseFloat>, DoubleVec<SseFloat>> exp2_split(
    DoubleVec<SseFloat> x) {
  using DV = DoubleVec<SseFloat>;
  __m128d kShift = _mm_set1_pd(0x1.8p+52);
  __m128i kShiftBits = _mm_set1_epi64x(0x4338000000000000LL);
  __m128i k1023 = _mm_set1_epi64x(1023);

  __m128d shifted_lo = _mm_add_pd(x.lo, kShift);
  __m128d shifted_hi = _mm_add_pd(x.hi, kShift);

  __m128i ki_lo = _mm_castpd_si128(shifted_lo);
  __m128i ki_hi = _mm_castpd_si128(shifted_hi);

  __m128d nd_lo = _mm_sub_pd(shifted_lo, kShift);
  __m128d nd_hi = _mm_sub_pd(shifted_hi, kShift);

  __m128d r_lo = _mm_sub_pd(x.lo, nd_lo);
  __m128d r_hi = _mm_sub_pd(x.hi, nd_hi);

  __m128i n_lo = _mm_sub_epi64(ki_lo, kShiftBits);
  __m128i n_hi = _mm_sub_epi64(ki_hi, kShiftBits);

  __m128d scale_lo = _mm_castsi128_pd(_mm_slli_epi64(_mm_add_epi64(n_lo, k1023), 52));
  __m128d scale_hi = _mm_castsi128_pd(_mm_slli_epi64(_mm_add_epi64(n_hi, k1023), 52));

  return {DV{r_lo, r_hi}, DV{scale_lo, scale_hi}};
}

#endif // __SSE4_1__

// ---------------------------------------------------------------------------
// AVX: DoubleVec<AvxFloat> holds two __m256d (8 float lanes → 2×4 doubles).
// ---------------------------------------------------------------------------
#if defined(__AVX2__)

template <>
struct DoubleVec<AvxFloat> {
  __m256d lo, hi; // lo = lanes 0-3; hi = lanes 4-7

  DoubleVec() = default;
  DoubleVec(__m256d l, __m256d h) : lo(l), hi(h) {}
  explicit DoubleVec(double d) : lo(_mm256_set1_pd(d)), hi(_mm256_set1_pd(d)) {}

  static DISPENSO_INLINE DoubleVec from_float(AvxFloat f) {
    return {
        _mm256_cvtps_pd(_mm256_castps256_ps128(f.v)),
        _mm256_cvtps_pd(_mm256_extractf128_ps(f.v, 1))};
  }

  // Gather: AVX2 has native 4-wide double gather from int32 indices.
  // Two gathers cover all 8 lanes. The table is 256 bytes (4 cachelines),
  // likely L1-hot after the first call.
  static DISPENSO_INLINE DoubleVec gather(const double* base, AvxInt32 idx) {
    __m128i lo4 = _mm256_castsi256_si128(idx.v);
    __m128i hi4 = _mm256_extracti128_si256(idx.v, 1);
    return {_mm256_i32gather_pd(base, lo4, 8), _mm256_i32gather_pd(base, hi4, 8)};
  }

  DISPENSO_INLINE AvxFloat to_float() const {
    __m128 lo_f = _mm256_cvtpd_ps(lo);
    __m128 hi_f = _mm256_cvtpd_ps(hi);
    return AvxFloat{_mm256_set_m128(hi_f, lo_f)};
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return {_mm256_add_pd(a.lo, b.lo), _mm256_add_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return {_mm256_sub_pd(a.lo, b.lo), _mm256_sub_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return {_mm256_mul_pd(a.lo, b.lo), _mm256_mul_pd(a.hi, b.hi)};
  }

  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
#if defined(__FMA__)
    return {_mm256_fmadd_pd(a.lo, b.lo, c.lo), _mm256_fmadd_pd(a.hi, b.hi, c.hi)};
#else
    return {
        _mm256_add_pd(_mm256_mul_pd(a.lo, b.lo), c.lo),
        _mm256_add_pd(_mm256_mul_pd(a.hi, b.hi), c.hi)};
#endif
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return {
        _mm256_min_pd(_mm256_max_pd(x.lo, low.lo), high.lo),
        _mm256_min_pd(_mm256_max_pd(x.hi, low.hi), high.hi)};
  }
};

DISPENSO_INLINE std::pair<DoubleVec<AvxFloat>, DoubleVec<AvxFloat>> exp2_split(
    DoubleVec<AvxFloat> x) {
  using DV = DoubleVec<AvxFloat>;
  __m256d kShift = _mm256_set1_pd(0x1.8p+52);
  __m256i kShiftBits = _mm256_set1_epi64x(0x4338000000000000LL);
  __m256i k1023 = _mm256_set1_epi64x(1023);

  __m256d shifted_lo = _mm256_add_pd(x.lo, kShift);
  __m256d shifted_hi = _mm256_add_pd(x.hi, kShift);

  __m256i ki_lo = _mm256_castpd_si256(shifted_lo);
  __m256i ki_hi = _mm256_castpd_si256(shifted_hi);

  __m256d nd_lo = _mm256_sub_pd(shifted_lo, kShift);
  __m256d nd_hi = _mm256_sub_pd(shifted_hi, kShift);

  __m256d r_lo = _mm256_sub_pd(x.lo, nd_lo);
  __m256d r_hi = _mm256_sub_pd(x.hi, nd_hi);

  __m256i n_lo = _mm256_sub_epi64(ki_lo, kShiftBits);
  __m256i n_hi = _mm256_sub_epi64(ki_hi, kShiftBits);

  __m256d scale_lo = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(n_lo, k1023), 52));
  __m256d scale_hi = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(n_hi, k1023), 52));

  return {DV{r_lo, r_hi}, DV{scale_lo, scale_hi}};
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// AVX-512: DoubleVec<Avx512Float> holds two __m512d (16 floats → 2×8 doubles).
// ---------------------------------------------------------------------------
#if defined(__AVX512F__)

template <>
struct DoubleVec<Avx512Float> {
  __m512d lo, hi; // lo = lanes 0-7; hi = lanes 8-15

  DoubleVec() = default;
  DoubleVec(__m512d l, __m512d h) : lo(l), hi(h) {}
  explicit DoubleVec(double d) : lo(_mm512_set1_pd(d)), hi(_mm512_set1_pd(d)) {}

  static DISPENSO_INLINE DoubleVec from_float(Avx512Float f) {
    __m256 lo8 = _mm512_castps512_ps256(f.v);
    // Extract upper 8 floats without requiring AVX-512 DQ:
    // cast to __m512i, extract upper 256-bit int lane, reinterpret as __m256.
    __m256 hi8 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(f.v), 1));
    return {_mm512_cvtps_pd(lo8), _mm512_cvtps_pd(hi8)};
  }

  // Gather: AVX-512 has native 8-wide double gather from int32 indices.
  // Two gathers cover all 16 lanes.
  static DISPENSO_INLINE DoubleVec gather(const double* base, Avx512Int32 idx) {
    __m256i lo8 = _mm512_castsi512_si256(idx.v);
    __m256i hi8 = _mm512_extracti64x4_epi64(idx.v, 1);
    return {_mm512_i32gather_pd(lo8, base, 8), _mm512_i32gather_pd(hi8, base, 8)};
  }

  DISPENSO_INLINE Avx512Float to_float() const {
    __m256 lo_f = _mm512_cvtpd_ps(lo);
    __m256 hi_f = _mm512_cvtpd_ps(hi);
    // Combine without AVX-512 DQ: cast to int, insert, cast back.
    __m512i combined = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_castps_si256(lo_f)), _mm256_castps_si256(hi_f), 1);
    return Avx512Float{_mm512_castsi512_ps(combined)};
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return {_mm512_add_pd(a.lo, b.lo), _mm512_add_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return {_mm512_sub_pd(a.lo, b.lo), _mm512_sub_pd(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return {_mm512_mul_pd(a.lo, b.lo), _mm512_mul_pd(a.hi, b.hi)};
  }

  // AVX-512 always has FMA.
  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
    return {_mm512_fmadd_pd(a.lo, b.lo, c.lo), _mm512_fmadd_pd(a.hi, b.hi, c.hi)};
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return {
        _mm512_min_pd(_mm512_max_pd(x.lo, low.lo), high.lo),
        _mm512_min_pd(_mm512_max_pd(x.hi, low.hi), high.hi)};
  }
};

DISPENSO_INLINE std::pair<DoubleVec<Avx512Float>, DoubleVec<Avx512Float>> exp2_split(
    DoubleVec<Avx512Float> x) {
  using DV = DoubleVec<Avx512Float>;
  __m512d kShift = _mm512_set1_pd(0x1.8p+52);
  __m512i kShiftBits = _mm512_set1_epi64(0x4338000000000000LL);
  __m512i k1023 = _mm512_set1_epi64(1023);

  __m512d shifted_lo = _mm512_add_pd(x.lo, kShift);
  __m512d shifted_hi = _mm512_add_pd(x.hi, kShift);

  __m512i ki_lo = _mm512_castpd_si512(shifted_lo);
  __m512i ki_hi = _mm512_castpd_si512(shifted_hi);

  __m512d nd_lo = _mm512_sub_pd(shifted_lo, kShift);
  __m512d nd_hi = _mm512_sub_pd(shifted_hi, kShift);

  __m512d r_lo = _mm512_sub_pd(x.lo, nd_lo);
  __m512d r_hi = _mm512_sub_pd(x.hi, nd_hi);

  __m512i n_lo = _mm512_sub_epi64(ki_lo, kShiftBits);
  __m512i n_hi = _mm512_sub_epi64(ki_hi, kShiftBits);

  __m512d scale_lo = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(n_lo, k1023), 52));
  __m512d scale_hi = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(n_hi, k1023), 52));

  return {DV{r_lo, r_hi}, DV{scale_lo, scale_hi}};
}

#endif // __AVX512F__

// ---------------------------------------------------------------------------
// NEON: DoubleVec<NeonFloat> holds two float64x2_t (4 floats → 2×2 doubles).
// ---------------------------------------------------------------------------
#if defined(__aarch64__)

template <>
struct DoubleVec<NeonFloat> {
  float64x2_t lo, hi; // lo = lanes 0,1; hi = lanes 2,3

  DoubleVec() = default;
  DoubleVec(float64x2_t l, float64x2_t h) : lo(l), hi(h) {}
  explicit DoubleVec(double d) : lo(vdupq_n_f64(d)), hi(vdupq_n_f64(d)) {}

  static DISPENSO_INLINE DoubleVec from_float(NeonFloat f) {
    return {vcvt_f64_f32(vget_low_f32(f.v)), vcvt_f64_f32(vget_high_f32(f.v))};
  }

  // Gather: load base[idx[lane]] for each of 4 lanes via scalar lookups.
  static DISPENSO_INLINE DoubleVec gather(const double* base, NeonInt32 idx) {
    int32_t i0 = vgetq_lane_s32(idx.v, 0);
    int32_t i1 = vgetq_lane_s32(idx.v, 1);
    int32_t i2 = vgetq_lane_s32(idx.v, 2);
    int32_t i3 = vgetq_lane_s32(idx.v, 3);
    float64x2_t lo_d = vsetq_lane_f64(base[i1], vdupq_n_f64(base[i0]), 1);
    float64x2_t hi_d = vsetq_lane_f64(base[i3], vdupq_n_f64(base[i2]), 1);
    return {lo_d, hi_d};
  }

  DISPENSO_INLINE NeonFloat to_float() const {
    float32x2_t lo_f = vcvt_f32_f64(lo);
    float32x2_t hi_f = vcvt_f32_f64(hi);
    return NeonFloat{vcombine_f32(lo_f, hi_f)};
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return {vaddq_f64(a.lo, b.lo), vaddq_f64(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return {vsubq_f64(a.lo, b.lo), vsubq_f64(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return {vmulq_f64(a.lo, b.lo), vmulq_f64(a.hi, b.hi)};
  }

  // AArch64 always has FMA.
  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
    return {vfmaq_f64(c.lo, a.lo, b.lo), vfmaq_f64(c.hi, a.hi, b.hi)};
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return {
        vminnmq_f64(vmaxnmq_f64(x.lo, low.lo), high.lo),
        vminnmq_f64(vmaxnmq_f64(x.hi, low.hi), high.hi)};
  }
};

DISPENSO_INLINE std::pair<DoubleVec<NeonFloat>, DoubleVec<NeonFloat>> exp2_split(
    DoubleVec<NeonFloat> x) {
  using DV = DoubleVec<NeonFloat>;
  float64x2_t kShift = vdupq_n_f64(0x1.8p+52);

  float64x2_t shifted_lo = vaddq_f64(x.lo, kShift);
  float64x2_t shifted_hi = vaddq_f64(x.hi, kShift);

  uint64x2_t ki_lo = vreinterpretq_u64_f64(shifted_lo);
  uint64x2_t ki_hi = vreinterpretq_u64_f64(shifted_hi);

  float64x2_t nd_lo = vsubq_f64(shifted_lo, kShift);
  float64x2_t nd_hi = vsubq_f64(shifted_hi, kShift);

  float64x2_t r_lo = vsubq_f64(x.lo, nd_lo);
  float64x2_t r_hi = vsubq_f64(x.hi, nd_hi);

  uint64x2_t kShiftBits = vdupq_n_u64(0x4338000000000000ULL);
  uint64x2_t k1023 = vdupq_n_u64(1023);

  uint64x2_t n_lo = vsubq_u64(ki_lo, kShiftBits);
  uint64x2_t n_hi = vsubq_u64(ki_hi, kShiftBits);

  float64x2_t scale_lo = vreinterpretq_f64_u64(vshlq_n_u64(vaddq_u64(n_lo, k1023), 52));
  float64x2_t scale_hi = vreinterpretq_f64_u64(vshlq_n_u64(vaddq_u64(n_hi, k1023), 52));

  return {DV{r_lo, r_hi}, DV{scale_lo, scale_hi}};
}

#endif // __aarch64__

// ---------------------------------------------------------------------------
// Highway: DoubleVec<HwyFloat> holds two Vec<Repartition<double, FloatTag>>.
// Width adapts to the compile-time target (SSE→2×2, AVX→2×4, AVX-512→2×8).
// ---------------------------------------------------------------------------
#if __has_include("hwy/highway.h")

#include <dispenso/fast_math/float_traits_hwy.h>

template <>
struct DoubleVec<HwyFloat> {
  using DTag = hn::Repartition<double, HwyFloatTag>;
  using DV = hn::Vec<DTag>;
  using ITag = hn::RebindToSigned<DTag>;

  DV lo, hi; // lo = lower half lanes, hi = upper half lanes

  DoubleVec() = default;
  DoubleVec(DV l, DV h) : lo(l), hi(h) {}
  explicit DoubleVec(double d) : lo(hn::Set(DTag{}, d)), hi(hn::Set(DTag{}, d)) {}

  static DISPENSO_INLINE DoubleVec from_float(HwyFloat f) {
    const DTag dd;
    return {hn::PromoteLowerTo(dd, f.v), hn::PromoteUpperTo(dd, f.v)};
  }

  // Gather: load base[idx[lane]] for each lane via scalar lookups.
  // Highway lacks a double-precision gather from int32 indices, so we
  // extract indices and load scalars. The table is small and L1-hot.
  static DISPENSO_INLINE DoubleVec gather(const double* base, HwyInt32 idx) {
    const DTag dd;
    constexpr size_t kMaxF = HWY_MAX_BYTES / sizeof(float);
    constexpr size_t kMaxD = kMaxF / 2; // half as many double lanes
    HWY_ALIGN int32_t ibuf[kMaxF];
    hn::StoreU(idx.v, HwyInt32Tag{}, ibuf);
    HWY_ALIGN double lo_buf[kMaxD];
    HWY_ALIGN double hi_buf[kMaxD];
    const size_t nd = hn::Lanes(dd);
    for (size_t j = 0; j < nd; ++j) {
      lo_buf[j] = base[ibuf[j]];
      hi_buf[j] = base[ibuf[nd + j]];
    }
    return {hn::Load(dd, lo_buf), hn::Load(dd, hi_buf)};
  }

  DISPENSO_INLINE HwyFloat to_float() const {
    const HwyFloatTag df;
    // DemoteTo narrows each double half to a half-width float vector.
    auto lo_f = hn::DemoteTo(hn::Rebind<float, DTag>{}, lo);
    auto hi_f = hn::DemoteTo(hn::Rebind<float, DTag>{}, hi);
    // Combine: lo_f and hi_f are each half-width Vec<float>; concatenate.
    return HwyFloat{hn::Combine(df, hi_f, lo_f)};
  }

  friend DISPENSO_INLINE DoubleVec operator+(DoubleVec a, DoubleVec b) {
    return {hn::Add(a.lo, b.lo), hn::Add(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator-(DoubleVec a, DoubleVec b) {
    return {hn::Sub(a.lo, b.lo), hn::Sub(a.hi, b.hi)};
  }
  friend DISPENSO_INLINE DoubleVec operator*(DoubleVec a, DoubleVec b) {
    return {hn::Mul(a.lo, b.lo), hn::Mul(a.hi, b.hi)};
  }

  friend DISPENSO_INLINE DoubleVec fma(DoubleVec a, DoubleVec b, DoubleVec c) {
    return {hn::MulAdd(a.lo, b.lo, c.lo), hn::MulAdd(a.hi, b.hi, c.hi)};
  }

  friend DISPENSO_INLINE DoubleVec clamp(DoubleVec x, DoubleVec low, DoubleVec high) {
    return {hn::Min(hn::Max(x.lo, low.lo), high.lo), hn::Min(hn::Max(x.hi, low.hi), high.hi)};
  }
};

DISPENSO_INLINE std::pair<DoubleVec<HwyFloat>, DoubleVec<HwyFloat>> exp2_split(
    DoubleVec<HwyFloat> x) {
  using DVH = DoubleVec<HwyFloat>;
  using DTag = DVH::DTag;
  using ITag = DVH::ITag;

  auto kShift = hn::Set(DTag{}, 0x1.8p+52);
  auto kShiftBits = hn::Set(ITag{}, 0x4338000000000000LL);
  auto k1023 = hn::Set(ITag{}, int64_t{1023});

  auto shifted_lo = hn::Add(x.lo, kShift);
  auto shifted_hi = hn::Add(x.hi, kShift);

  auto ki_lo = hn::BitCast(ITag{}, shifted_lo);
  auto ki_hi = hn::BitCast(ITag{}, shifted_hi);

  auto nd_lo = hn::Sub(shifted_lo, kShift);
  auto nd_hi = hn::Sub(shifted_hi, kShift);

  auto r_lo = hn::Sub(x.lo, nd_lo);
  auto r_hi = hn::Sub(x.hi, nd_hi);

  auto n_lo = hn::Sub(ki_lo, kShiftBits);
  auto n_hi = hn::Sub(ki_hi, kShiftBits);

  auto scale_lo = hn::BitCast(DTag{}, hn::ShiftLeft<52>(hn::Add(n_lo, k1023)));
  auto scale_hi = hn::BitCast(DTag{}, hn::ShiftLeft<52>(hn::Add(n_hi, k1023)));

  return {DVH{r_lo, r_hi}, DVH{scale_lo, scale_hi}};
}

#endif // __has_include("hwy/highway.h")

} // namespace detail
} // namespace fast_math
} // namespace dispenso
