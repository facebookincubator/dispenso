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

#include "detail/double_promote.h"
#include "detail/fast_math_impl.h"

namespace dispenso {
namespace fast_math {

namespace detail {

// AccuracyTraits adapter for internal pow use: always uses best polynomial,
// inherits kBoundsValues from the outer traits.
template <typename AccuracyTraits>
struct PowInternalTraits {
  static constexpr bool kMaxAccuracy = true;
  static constexpr bool kBoundsValues = AccuracyTraits::kBoundsValues;
};

// Scalar integer parity check on float bits.
// Returns 0 if y is not an integer, 1 if even integer, 2 if odd integer.
// Works regardless of sign bit (sign is masked off from exponent field).
// Branching version — used inside the rare negative-x path.
DISPENSO_INLINE int32_t checkint(uint32_t iy) {
  int32_t e = static_cast<int32_t>(iy >> 23) & 0xff;
  if (e < 127)
    return 0; // |y| < 1 → not integer (includes 0, subnormals)
  if (e > 127 + 23)
    return 1; // |y| >= 2^24 → ULP ≥ 2, only even integers representable
  // 127 <= e <= 150: check mantissa bits below the integer point.
  uint32_t shift = static_cast<uint32_t>(127 + 23 - e);
  if (iy & ((1u << shift) - 1u))
    return 0; // fractional mantissa bits → not integer
  if (iy & (1u << shift))
    return 2; // lowest integer bit set → odd
  return 1; // even integer
}

// Branchless integer parity check for the scalar non-bounds path.
// Returns 0 if not integer, 1 if even integer, 2 if odd integer.
// For the non-bounds default path, we want to avoid branches on the hot path.
// The negative-x case is rare, but mixed-sign workloads benefit from branchless.
DISPENSO_INLINE int32_t checkint_branchless(uint32_t iy) {
  int32_t e = static_cast<int32_t>(iy >> 23) & 0xff;
  // Clamp shift to [0, 31] to avoid UB from shifting >= 32. When e < 127
  // (|y| < 1, not integer), the clamped shift reads a garbage bit, but
  // is_int below is false so the garbage is never used.
  uint32_t shift = static_cast<uint32_t>(127 + 23 - e) & 31u;
  uint32_t has_frac = iy & ((1u << shift) - 1u);
  uint32_t odd_bit = (iy >> shift) & 1u;
  // is_int: e >= 127 and no fractional bits (or e > 150, always integer).
  // For e > 150: shift would be negative (clamped to garbage), but we force result = 1.
  int32_t is_int = (e >= 127) & (has_frac == 0);
  int32_t is_large = (e > 127 + 23); // always even
  // Result: 0 if not int, 1 if even int, 2 if odd int.
  // odd_bit is meaningful only when is_int && !is_large.
  return is_int + (is_int & static_cast<int32_t>(odd_bit) & ~is_large);
}

// Scalar double-precision pow core: exp2(y * log2(|x|)) entirely in double.
// Float inputs/output, double internal arithmetic.
//
// log2: 16-entry table of {1/c, log2(c)} eliminates division. Degree-4
//   polynomial (Sollya fpminimax) approximates log2(1+r)/r with pipelined
//   evaluation. Table centers at OFF + i*2^19 + 2^18, max |r| < 0.0297.
// exp2: 32-entry table of 2^(k/32) stored as uint64 with pre-subtracted
//   exponent bits. Degree-3 polynomial (Sollya fpminimax) on [-1/64, 1/64].
//
// Total: ~8 double muls + ~5 double adds in two short parallel chains.
// Achieves <1 ULP for float output.

// log2 table: 16 entries, each {1/c, log2(c)} where c is the center of
// the ith subinterval of [OFF, 2*OFF] (OFF = 0x3f330000).
struct PowLog2Entry {
  double invc;
  double logc;
};

// EXP2F_TABLE_BITS = 5 → 32 entries.
// tab[i] = uint64(2^(i/32)) - (i << (52-5)).
// To reconstruct 2^(k/32): double(tab[k%32] + (k << 47)).
constexpr uint64_t kPowExp2Tab[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};

constexpr PowLog2Entry kPowLog2Tab[16] = {
    {0x1.661ec6a5122f9p+0, -0x1.efec61b011f85p-2}, // i=0,  c=0.71484375
    {0x1.571ed3c506b3ap+0, -0x1.b0b67f4f46810p-2}, // i=1,  c=0.74609375
    {0x1.49539e3b2d067p+0, -0x1.7418acebbf18fp-2}, // i=2,  c=0.77734375
    {0x1.3c995a47babe7p+0, -0x1.39de8e1559f6fp-2}, // i=3,  c=0.80859375
    {0x1.30d190130d190p+0, -0x1.01d9bbcfa61d4p-2}, // i=4,  c=0.83984375
    {0x1.25e22708092f1p+0, -0x1.97c1cb13c7ec1p-3}, // i=5,  c=0.87109375
    {0x1.1bb4a4046ed29p+0, -0x1.2f9e32d5bfdd1p-3}, // i=6,  c=0.90234375
    {0x1.12358e75d3033p+0, -0x1.960caf9abb7cap-4}, // i=7,  c=0.93359375
    {0x1.0953f39010954p+0, -0x1.a6f9c377dd31bp-5}, // i=8,  c=0.96484375
    {0x1p+0, 0x0p+0}, // i=9,  c=1.0 (exact)
    {0x1.e573ac901e574p-1, 0x1.3aa2fdd27f1c3p-4}, // i=10, c=1.05468750
    {0x1.ca4b3055ee191p-1, 0x1.476a9f983f74dp-3}, // i=11, c=1.11718750
    {0x1.b2036406c80d9p-1, 0x1.e840be74e6a4dp-3}, // i=12, c=1.17968750
    {0x1.9c2d14ee4a102p-1, 0x1.406463b1b0449p-2}, // i=13, c=1.24218750
    {0x1.886e5f0abb04ap-1, 0x1.88e9c72e0b226p-2}, // i=14, c=1.30468750
    {0x1.767dce434a9b1p-1, 0x1.ce0a4923a587dp-2}, // i=15, c=1.36718750
};

// log2(1+r)/r polynomial coefficients (degree 4, Sollya fpminimax).
// Relative error < 2^-32 on |r| < 0.0297.
constexpr double kPowLog2Poly[5] = {
    0x1.27bc533354f16p-2, // P[0]: r^4 coeff
    -0x1.719a02b5d4c08p-2, // P[1]: r^3 coeff
    0x1.ec70982733e8fp-2, // P[2]: r^2 coeff
    -0x1.7154745c0af07p-1, // P[3]: r   coeff (approx -1/(2*ln2))
    0x1.71547652bc3ep0, // P[4]: r^0 coeff (approx 1/ln2)
};

// exp2 polynomial coefficients (degree 3 on [-1/64, 1/64], Sollya fpminimax).
// Relative error < 2^-27 on (2^r - 1)/r.
constexpr double kPowExp2Poly[3] = {
    0x1.c6b08d7047f43p-5, // C[0]: r^2 coeff
    0x1.ebfccc582d012p-3, // C[1]: r   coeff
    0x1.62e42fefe5286p-1, // C[2]: r^0 coeff (approx ln2)
};

DISPENSO_INLINE float pow_double_core(float ax, float y) {
  constexpr uint32_t kOff = 0x3f330000;

  // --- log2 ---
  uint32_t ix = bit_cast<uint32_t>(ax);
  uint32_t tmp = ix - kOff;
  int32_t i = (tmp >> (23 - 4)) & 15; // table index
  uint32_t top = tmp & 0xff800000u;
  uint32_t iz = ix - top;
  int32_t k = static_cast<int32_t>(top) >> 23; // exponent (arithmetic shift)
  double invc = kPowLog2Tab[i].invc;
  double logc = kPowLog2Tab[i].logc;
  double z = static_cast<double>(bit_cast<float>(iz));

  // log2(x) = log1p(z/c - 1)/ln2 + log2(c) + k
  double r = z * invc - 1.0;
  double y0 = logc + static_cast<double>(k);

  // Pipelined polynomial: two parallel chains merged via r4.
  double r2 = r * r;
  double p0 = kPowLog2Poly[0] * r + kPowLog2Poly[1];
  double p1 = kPowLog2Poly[2] * r + kPowLog2Poly[3];
  double r4 = r2 * r2;
  double p2 = kPowLog2Poly[4] * r + y0;
  p2 = p1 * r2 + p2;
  double logx = p0 * r4 + p2;

  // --- y * log2(x) ---
  double ylogx = static_cast<double>(y) * logx;

  // Overflow/underflow check.
  // Check if |ylogx| >= 126 using exponent bits.
  uint64_t ylogx_bits = bit_cast<uint64_t>(ylogx);
  if ((ylogx_bits >> 47 & 0xffff) >= (bit_cast<uint64_t>(126.0) >> 47)) {
    if (ylogx > 0x1.fffffffd1d571p+6) {
      return std::numeric_limits<float>::infinity();
    }
    if (ylogx <= -150.0) {
      return 0.0f;
    }
  }

  // --- exp2 ---
  // x = k/N + r with r in [-1/(2N), 1/(2N)], N=32.
  constexpr double kShift = 0x1.8p+52 / 32.0; // magic for round-to-nearest
  double kd = ylogx + kShift;
  uint64_t ki = bit_cast<uint64_t>(kd);
  kd -= kShift; // k/N, rounded
  double rd = ylogx - kd;

  // exp2(x) = 2^(k/N) * (C0*r^3 + C1*r^2 + C2*r + 1)
  //         = s * poly(r)
  uint64_t t = kPowExp2Tab[ki & 31];
  t += ki << (52 - 5); // add exponent
  double s = bit_cast<double>(t);

  double r2e = rd * rd;
  double pe = kPowExp2Poly[0] * rd + kPowExp2Poly[1];
  double qe = kPowExp2Poly[2] * rd + 1.0;
  qe = pe * r2e + qe;
  return static_cast<float>(qe * s);
}

// Hybrid double-precision pow core: table-assisted log2 + tableless exp2.
// Float inputs/output, double internal arithmetic via DoubleVec<Flt>.
//
// log2: SIMD range reduction (float/uint32) extracts a 4-bit table index per
//   lane. Scalar lookups fetch {1/c, log2(c)} from the 16-entry table, packed
//   into DoubleVec. The table narrows |r| to <0.03, enabling a degree-4
//   polynomial (vs degree-10 tableless). 8 scalar loads total (4 lanes × 2
//   values), always L1-hot (256 bytes).
// exp2: tableless degree-5 polynomial via exp2_split (same as poly core).
//
// Works for both scalar (Flt=float) and SIMD (Flt=SseFloat, etc.).
template <typename Flt>
DISPENSO_INLINE Flt pow_double_hybrid_core(Flt ax, Flt y) {
  using DV = DoubleVec<Flt>;
  using IntT = IntType_t<Flt>;
  using UintT = UintType_t<Flt>;

  // --- Range reduction (SIMD, same as pow_double_core) ---
  // Note: does NOT handle subnormal ax (ix < 0x00800000). Callers must
  // catch subnormals in is_special and fall back to pow_double_poly_core.
  constexpr uint32_t kOff = 0x3f330000;
  UintT ix = bit_cast<UintT>(ax);
  UintT tmp = ix - UintT(kOff);
  IntT i = (bit_cast<IntT>(tmp) >> 19) & IntT(15); // 4-bit table index
  UintT top = tmp & UintT(0xff800000u);
  UintT iz = ix - top;
  IntT k = bit_cast<IntT>(top) >> 23; // exponent (arithmetic shift)

  // --- Table lookups: 4 lanes × {invc, logc} = 8 scalar loads ---
  // Table is 16 × {double, double} = 256 bytes, stride 2 doubles per entry.
  const double* tab = reinterpret_cast<const double*>(kPowLog2Tab);
  IntT i2 = i + i; // stride: 2 doubles per entry
  DV invc = DV::gather(tab, i2); // tab[2*i + 0] = invc
  DV logc = DV::gather(tab, i2 + IntT(1)); // tab[2*i + 1] = logc

  // --- log2 in double ---
  DV z = DV::from_float(bit_cast<Flt>(iz));
  DV r = z * invc - DV(1.0);
  DV y0 = logc + DV::from_float(Flt(k));

  // Degree-4 Horner for log2 (chain too short for Estrin benefit).
  // log2(x) = y0 + r * P(r), |r| < 0.03.
  DV p(kPowLog2Poly[0]);
  p = fma(p, r, DV(kPowLog2Poly[1]));
  p = fma(p, r, DV(kPowLog2Poly[2]));
  p = fma(p, r, DV(kPowLog2Poly[3]));
  p = fma(p, r, DV(kPowLog2Poly[4]));
  DV logx = fma(p, r, y0);

  // --- y * log2(x) ---
  DV ylogx = DV::from_float(y) * logx;

  // Clamp to valid double exponent range.
  ylogx = clamp(ylogx, DV(-1022.0), DV(1023.0));

  // --- Tableless exp2 in double ---
  auto [re, scale] = exp2_split(ylogx);

  // Degree-5 Horner for exp2 (chain too short for Estrin benefit).
  DV q(0x1.4308fabc18decp-13);
  q = fma(q, re, DV(0x1.5f07b4611e454p-10));
  q = fma(q, re, DV(0x1.3b2b9fbd8970fp-7));
  q = fma(q, re, DV(0x1.c6af6e92be375p-5));
  q = fma(q, re, DV(0x1.ebfbdec29c82ap-3));
  q = fma(q, re, DV(0x1.62e4302eeb44dp-1));

  DV result = fma(scale, re * q, scale);
  return result.to_float();
}

// Tableless double-precision pow core: exp2(y * log2(|x|)) entirely in double.
// Float inputs/output, double internal arithmetic via DoubleVec<Flt>.
//
// log2: Uses logarithmSep range reduction (float/uint32) to get exponent i and
//   mantissa m in [sqrt(0.5), sqrt(2)]. Then t = (double)m - 1.0 (exact in
//   double) and log2(1+t) is evaluated as t * P(t) where P is a degree-10
//   Sollya fpminimax polynomial (~30 bits).
//   No tables, no division, no error tracking.
// exp2: Tableless double-precision polynomial. Range reduce to integer n and
//   fractional r ∈ [-0.5, 0.5], then 2^x = 2^n * (1 + r*Q(r)) where Q is a
//   degree-5 Sollya fpminimax polynomial (~27 bits). No tables.
//
// Works for both scalar (Flt=float) and SIMD (Flt=SseFloat, etc.).
template <typename Flt>
DISPENSO_INLINE Flt pow_double_poly_core(Flt ax, Flt y) {
  using DV = DoubleVec<Flt>;

  // --- Range reduction (float/uint32) ---
  // logarithmSep gives us exponent i and mantissa m in [sqrt(0.5), sqrt(2)].
  auto [xi, i_f, m] = logarithmSep<Flt, MaxAccuracyTraits>(ax);

  // --- Widen to double BEFORE subtraction ---
  // m - 1 in float loses ~17 bits when m ≈ 1 (catastrophic cancellation).
  // In double, (double)m - 1.0 is exact since every float is exactly
  // representable in double, and the result fits in 24 mantissa bits.
  DV t = DV::from_float(m) - DV(1.0);
  DV i_d = DV::from_float(i_f);

  // --- Double polynomial: log2(1+t) = t * P(t), degree 10 (Estrin) ---
  // Sollya fpminimax on [-0.293, 0.414], double precision.
  // Sup-norm error: 1.10e-9 (~30 bits).
  // Estrin's scheme evaluates independent sub-polynomials in parallel,
  // reducing latency vs Horner by exploiting ILP (multiple independent FMAs
  // at each level feed both FMA ports simultaneously).
  // P(t) = (c0+c1*t) + t²*(c2+c3*t) + t⁴*(c4+c5*t) + t⁶*(c6+c7*t)
  //       + t⁸*(c8+c9*t+c10*t²)
  //
  // Level 0: t², and five independent coefficient pairs.
  DV t2 = t * t;
  DV p01 = fma(DV(-0x1.7154762acdfffp-1), t, DV(0x1.71547656a2495p+0));
  DV p23 = fma(DV(-0x1.71548ef022a84p-2), t, DV(0x1.ec707f060e35ep-2));
  DV p45 = fma(DV(-0x1.ec7a94d1207adp-3), t, DV(0x1.278083c622fbcp-2));
  DV p67 = fma(DV(-0x1.6e5f6f8760a8ap-3), t, DV(0x1.a3ee210f50f0fp-3));
  DV p89 = fma(DV(-0x1.5a6abc244ec79p-3), t, DV(0x1.60552656955f2p-3));
  // Level 1: t⁴, combine pairs with t².
  DV t4 = t2 * t2;
  DV q03 = fma(p23, t2, p01);
  DV q47 = fma(p67, t2, p45);
  DV q8A = fma(DV(0x1.8e0e8c535162ap-4), t2, p89); // c10*t² + (c8+c9*t)
  // Level 2: t⁸, combine quads with t⁴.
  DV t8 = t4 * t4;
  DV r07 = fma(q47, t4, q03);
  // Level 3: final combination.
  DV p = fma(q8A, t8, r07);

  // log2(1+t) = t * P(t); log2(x) = i + log2(m)
  DV log2x = i_d + t * p;

  // --- y * log2(x) ---
  DV ylogx = DV::from_float(y) * log2x;

  // Clamp to valid double exponent range [-1022, 1023].
  // This prevents invalid bit patterns in exp2_split while preserving
  // correct float narrowing: 2^(-1022) → 0.0f, 2^1023 → inf.
  ylogx = clamp(ylogx, DV(-1022.0), DV(1023.0));

  // --- Tableless exp2 in double ---
  auto [r, scale] = exp2_split(ylogx);

  // 2^r = 1 + r * Q(r), degree-5 Sollya fpminimax on [-0.5, 0.5].
  // Q(r) ≈ (2^r - 1)/r. Sup-norm error: 8.88e-9 (~27 bits).
  DV q(0x1.4308fabc18decp-13); // e5
  q = fma(q, r, DV(0x1.5f07b4611e454p-10)); // e4
  q = fma(q, r, DV(0x1.3b2b9fbd8970fp-7)); // e3
  q = fma(q, r, DV(0x1.c6af6e92be375p-5)); // e2
  q = fma(q, r, DV(0x1.ebfbdec29c82ap-3)); // e1
  q = fma(q, r, DV(0x1.62e4302eeb44dp-1)); // e0

  // 2^ylogx = scale * (1 + r * q) = scale + scale * r * q
  DV result = fma(scale, r * q, scale);
  return result.to_float();
}

// --- Scalar pow IEEE 754 special-case helpers ---

// Handle y = 0, inf, or NaN for scalar pow bounds path.
// Returns true if the result is determined (stored in *out).
//
// Bit tricks used throughout:
//   2u * val          — left-shifts by 1, discarding the sign bit. This maps
//                       ±0 to 0, ±inf to 0xFF000000, and NaN to > 0xFF000000.
//   2u * val - 1u     — maps ±0 to UINT_MAX (wraps), normal floats to a
//                       positive range, and inf/NaN to >= 0xFF000000 - 1.
//                       So "2u*val - 1u >= 2u*0x7f800000u - 1u" tests whether
//                       val is zero, inf, or NaN in one branch-free comparison.
//   ix & 0x7FFFFFFFu  — clears the sign bit, giving |x| as raw bits.
DISPENSO_INLINE bool pow_scalar_y_special(float x, float y, uint32_t ix, uint32_t iy, float* out) {
  if (2u * iy == 0u) {
    *out = 1.0f;
    return true; // pow(x, ±0) = 1
  }
  if (ix == 0x3f800000u) {
    *out = 1.0f;
    return true; // pow(1, y) = 1 even for NaN y
  }
  // Either x or y is NaN → propagate via x + y (IEEE 754 NaN arithmetic).
  if (2u * (ix & 0x7fffffffu) > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u) {
    *out = x + y;
    return true;
  }
  // |x| == 1 (either +1 or -1) and y is ±inf → result is 1.
  if (2u * (ix & 0x7fffffffu) == 2u * 0x3f800000u) {
    *out = 1.0f;
    return true;
  }
  // y is ±inf (NaN cases handled above). Result is 0 or +inf depending on
  // whether |x| < 1 and the sign of y:
  //   |x| < 1, y = +inf → 0    |x| > 1, y = +inf → +inf
  //   |x| < 1, y = -inf → +inf |x| > 1, y = -inf → 0
  if ((2u * (ix & 0x7fffffffu) < 2u * 0x3f800000u) == !(iy & 0x80000000u)) {
    *out = 0.0f;
    return true;
  }
  *out = y * y; // +inf (y*y overflows to +inf for finite y, identity for ±inf)
  return true;
}

// Handle negative/zero/inf/subnormal x for scalar pow bounds path.
// Called when x is outside the normal positive range (the fast-path condition
// in the caller already filtered normal positive x). Returns true if the
// result is determined (stored in *out). Otherwise adjusts ix (to |x|, with
// subnormals normalized) and sign_bias for the core computation.
DISPENSO_INLINE bool
pow_scalar_x_special(uint32_t& ix, uint32_t iy, uint32_t& sign_bias, float* out) {
  if (ix & 0x80000000u) {
    // Negative x: check integer parity of y to determine sign of result.
    // checkint returns: 0 = not integer, 1 = even integer, 2 = odd integer.
    int32_t yint = checkint(iy);
    if (yint == 2)
      sign_bias = 0x80000000u; // odd integer y → negate final result
    ix &= 0x7fffffffu; // work with |x| from here
    if (ix == 0 || ix == 0x7f800000u) {
      // -0 or -inf: |x|^2 maps 0→0 and inf→inf. Then apply sign and
      // reciprocal (negative y inverts: pow(-0, -3) = -inf).
      float x2 = bit_cast<float>(ix) * bit_cast<float>(ix);
      if (iy & 0x80000000u)
        x2 = 1.0f / x2;
      *out = bit_cast<float>(bit_cast<uint32_t>(x2) ^ sign_bias);
      return true;
    }
    if (yint == 0) {
      *out = std::numeric_limits<float>::quiet_NaN(); // neg ^ non-integer = NaN
      return true;
    }
    // Negative finite nonzero x with integer y: fall through to core with |x|.
  }
  // Test for +0 or +inf. "2u*ix - 1u >= 2u*0x7f800000u - 1u" is true when
  // ix is 0 (wraps to UINT_MAX) or >= 0x7f800000 (inf/NaN, but NaN was
  // already handled by the caller's y-special path).
  if (2u * ix - 1u >= 2u * 0x7f800000u - 1u) {
    float x2 = bit_cast<float>(ix) * bit_cast<float>(ix);
    if (iy & 0x80000000u)
      x2 = 1.0f / x2;
    *out = x2;
    return true;
  }
  if (ix < 0x00800000u) {
    // Subnormal x: scale by 2^23 to normalize, then subtract 23 from the
    // biased exponent so the log2 core sees the correct value.
    ix = bit_cast<uint32_t>(bit_cast<float>(ix) * 0x1p23f);
    ix -= 23u << 23;
  }
  return false; // Fall through to core computation with adjusted ix.
}

// SIMD pow fixup for special inputs (zero, subnormal, inf, NaN).
// The double-precision cores produce garbage for these because logarithmSep's
// range reduction (exponent extraction via integer subtract) is undefined for
// zero/inf/NaN bit patterns. This cold-path fixup corrects only the affected
// lanes; the branch is almost never taken for normal workloads.
template <typename Flt>
DISPENSO_INLINE void pow_simd_special_fixup(Flt ax, Flt y, Flt& r) {
  using IntT = IntType_t<Flt>;
  using UintT = UintType_t<Flt>;
  IntT axi = bit_cast<IntT>(ax);
  // Detect special lanes: subnormal (exponent field 0, mantissa nonzero),
  // zero (all bits 0), inf (0x7F800000), NaN (> 0x7F800000).
  // nonnormal() checks exponent == 0xFF (inf/NaN). Combined with < 0x00800000
  // (subnormal/zero), this covers all non-normal inputs.
  auto is_special = (axi < IntT(0x00800000)) | nonnormal<Flt>(axi);
  if (DISPENSO_EXPECT(any_true(is_special), 0)) {
    // Subnormals: the hybrid core's range reduction (ix - kOff) can't handle
    // exponent-0 bit patterns. Recompute these lanes via poly core, whose
    // logarithmSep prescales subnormals by 2^23 before extraction.
    auto is_subnorm = (axi > IntT(0)) & (axi < IntT(0x00800000));
    if (any_true(is_subnorm)) {
      Flt r_sub = pow_double_poly_core(ax, y);
      r = FloatTraits<Flt>::conditional(is_subnorm, r_sub, r);
    }
    // Zero/inf/NaN lanes: use an integer trick that swaps 0↔inf in one op.
    // IEEE 754 bit layout: 0 = 0x00000000, +inf = 0x7F800000.
    // 0x7F800000 - 0x00000000 = 0x7F800000 (+inf)  → pow(0, pos) = 0, pow(0, neg) = inf
    // 0x7F800000 - 0x7F800000 = 0x00000000 (0)     → pow(inf, pos) = inf, pow(inf, neg) = 0
    // 0x7F800000 - (NaN bits) wraps negative, which reinterprets as NaN
    // (sign bit irrelevant for NaN). So:
    //   y > 0 → result = ax (0 stays 0, inf stays inf)
    //   y < 0 → result = flipped (0↔inf swap)
    //   NaN   → stays NaN either way
    auto is_zinf = (axi == IntT(0)) | nonnormal<Flt>(axi);
    UintT ax_bits = bit_cast<UintT>(ax);
    Flt flipped = bit_cast<Flt>(UintT(0x7F800000u) - ax_bits);
    Flt special_r = FloatTraits<Flt>::conditional(y < 0.0f, flipped, ax);
    r = FloatTraits<Flt>::conditional(is_zinf, special_r, r);
  }
}

// Scalar-exponent fast paths: exact results for common y values and
// binary squaring for small integer exponents. Sets y_is_int and y_is_odd
// as output for the caller's sign-handling path when returning false.
template <typename Flt>
DISPENSO_INLINE bool
pow_scalar_y_fast_path(Flt x, float y, bool& y_is_int, bool& y_is_odd, Flt& out) {
  using UintT = UintType_t<Flt>;
  if (y == 0.0f) {
    out = Flt(1.0f);
    return true;
  }
  if (y == 1.0f) {
    out = x;
    return true;
  }
  if (y == -1.0f) {
    out = Flt(1.0f) / x;
    return true;
  }
  if (y == 0.5f) {
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    auto negx = x < 0.0f;
    out = FloatTraits<Flt>::conditional(negx, Flt(kNaN), FloatTraits<Flt>::sqrt(x));
    return true;
  }
  if (y == 2.0f) {
    out = x * x;
    return true;
  }

  float ay = std::fabs(y);
  y_is_int = (std::floor(y) == y);
  y_is_odd = y_is_int && (std::fmod(ay, 2.0f) == 1.0f);

  if (y_is_int && ay < 64.0f) {
    // Binary squaring with uniform SIMD multiplies.
    int32_t n = static_cast<int32_t>(ay);
    Flt result(1.0f);
    Flt base = fabs(x);
    while (n > 0) {
      if (n & 1)
        result = result * base;
      base = base * base;
      n >>= 1;
    }
    if (y < 0.0f)
      result = Flt(1.0f) / result;
    if (y_is_odd) {
      UintT xsign = bit_cast<UintT>(x) & UintT(0x80000000u);
      result = bit_cast<Flt>(bit_cast<UintT>(result) ^ xsign);
    }
    out = result;
    return true;
  }
  return false;
}

} // namespace detail

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
 * @brief Decompose a float into mantissa and exponent (bit-accurate for normals).
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Accepted but ignored; result is always bit-accurate.
 * @param x Input value.
 * @param eptr Pointer to receive the exponent.
 * @return Mantissa in [0.5, 1) for normal values. Returns @p x unchanged for
 *   inf/NaN/zero/subnormal. Subnormals are passed through with *eptr = 0
 *   rather than normalized into [0.5, 1) as std::frexp does. This means the
 *   mantissa contract is violated for subnormals, but ldexp(frexp(x)) == x
 *   still holds (the round-trip is an identity). Code that depends on a
 *   normalized mantissa (e.g., counting significant bits) should not use
 *   subnormal inputs.
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
 * @tparam AccuracyTraits Default: 3 ULP. kMaxAccuracy: 2 ULP (degree-8 polynomial).
 * @param x Input value in [-1, 1].
 * @return Arc cosine of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt acos(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return acos<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    BoolType_t<Flt> xneg = x < 0.0f;
    // abs
    UintType_t<Flt> xi = bit_cast<UintType_t<Flt>>(x);
    xi &= 0x7fffffff;
    x = bit_cast<Flt>(xi);

    // acos(x) = sqrt(1-x) * P(x), where P approximates acos(x)/sqrt(1-x) on [0, 1).
    // Horner (not Estrin): Estrin regresses acos ULP from 3 to 4.
    Flt y;
    if constexpr (AccuracyTraits::kMaxAccuracy || AccuracyTraits::kBoundsValues) {
      // Sollya fpminimax(acos(x)/sqrt(1-x), 8, [|SG...|], [0, 1-1b-23]):
      //   sup-norm error < 2^-25.
      y = dispenso::fast_math::hornerEval(
          x,
          0x1.d129f2p-11f,
          -0x1.3ca11ap-8f,
          0x1.9a3376p-7f,
          -0x1.6a1a08p-6f,
          0x1.10b838p-5f,
          -0x1.a0365ep-5f,
          0x1.6cce04p-4f,
          -0x1.b7820cp-3f,
          0x1.921fb6p0f);
    } else {
      // Sollya fpminimax(acos(x)/sqrt(1-x), 7, [|SG...|], [0, 1-1b-23]):
      //   sup-norm error < 2^-24.
      y = dispenso::fast_math::hornerEval(
          x,
          -0x1.39a988p-10f,
          0x1.a50fdp-8f,
          -0x1.120206p-6f,
          0x1.f5a1e2p-6f,
          -0x1.9a1efp-5f,
          0x1.6c5d48p-4f,
          -0x1.b77e7ep-3f,
          0x1.921fb4p0f);
    }

    auto sqrt1mx = FloatTraits<Flt>::sqrt(1.0f - x);
    return FloatTraits<Flt>::conditional(
        xneg, FloatTraits<Flt>::fma(y, -sqrt1mx, Flt(kPi)), y * sqrt1mx);
  }
}

/**
 * @brief Arc sine approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: 2 ULP. kMaxAccuracy: 1 ULP (higher-degree polynomials).
 * @param x Input value in [-1, 1].
 * @return Arc sine of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt asin(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return asin<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;

    IntT xi = bit_cast<IntT>(x);
    IntT sgnbit = xi & 0x80000000;
    xi &= 0x7fffffff;
    x = bit_cast<Flt>(xi);

    Flt ret;
    if constexpr (std::is_same_v<Flt, float>) {
      if (xi > 0x3f000000) { // x > 0.5
        ret = detail::asin_pt5_1<Flt, AccuracyTraits>(x);

      } else {
        ret = detail::asin_0_pt5<Flt, AccuracyTraits>(x);
      }
    } else {
      // Use compensating sum for better rounding.
      auto y = detail::asin_pt5_1<Flt, AccuracyTraits>(x);
      auto z = detail::asin_0_pt5<Flt, AccuracyTraits>(x);
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
 * @tparam AccuracyTraits Default: 3 ULP. kMaxAccuracy: 2 ULP (degree-11 polynomial).
 * @param x Input value (all float domain).
 * @return Arc tangent of @p x in radians. Compatible with all SIMD backends.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt atan(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return atan<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
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

    auto y = detail::atan_poly<Flt, AccuracyTraits>(x);

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

    Flt y = dispenso::fast_math::hornerEval(
        x,
        0x1.c41242p-13f,
        0x1.4748aep-10f,
        0x1.3cfb6p-7f,
        0x1.c68b2ep-5f,
        0x1.ebfd38p-3f,
        0x1.62e42cp-1f,
        0x1p0f);

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
      Flt y = dispenso::fast_math::hornerEval(
          f,
          0x1.6850e4p-10f,
          0x1.123bccp-7f,
          0x1.555b98p-5f,
          0x1.55548ep-3f,
          0x1.fffff8p-2f,
          1.f,
          1.f);
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
          auto overflow = (x > 89.0f);
          if ((zero || overflow)) {
            Flt orbits = bit_cast<Flt>(bool_apply_or_zero(overflow, 0x7f800000));
            y = FloatTraits<Flt>::conditional(zero | overflow, orbits, y);
          }
        } else {
          auto zero = (x < -89.0f);
          auto overflow = (x > 89.0f);
          IntT orbits = bool_apply_or_zero<IntT>(overflow, IntT(0x7f800000));
          auto mask = bool_as_mask<IntT>(zero) | bool_as_mask<IntT>(overflow);
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
      Flt y = dispenso::fast_math::hornerEval(
          x,
          0x1.8014dp-7f,
          0x1.3f3846p-5f,
          0x1.57481ep-3f,
          0x1.ffd96ep-2f,
          0x1.000086p0f,
          0x1.fffffep-1f);

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
    Flt y = dispenso::fast_math::hornerEval(
        x,
        0x1.293a54p-2f,
        0x1.024feap-1f,
        0x1.2d9da2p0f,
        0x1.0459f6p1f,
        0x1.53534cp1f,
        0x1.26bb18p1f,
        0x1p0f);

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
    // Degree-9 polynomial with c0=0 absorbed: the Horner chain evaluates c9..c1,
    // then fma(y, m, m) applies the final y*m + m (since c0=0, c1 is folded into +m).
    Flt y = dispenso::fast_math::hornerEval(
        m,
        -1.09985352e-1f,
        1.86182275e-1f,
        -1.91066533e-1f,
        2.04593703e-1f,
        -2.39627063e-1f,
        2.88573444e-1f,
        -3.60695332e-1f,
        4.80897635e-1f,
        -7.21347392e-1f,
        4.42695051e-1f);
    y = FloatTraits<Flt>::fma(y, m, m); // c0=0 absorbed: y*m + m

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
    // Degree-9 polynomial with c0=0, c1=1 absorbed: y = P(m)*m² + m.
    // P(m) evaluates c9..c2 (8 coefficients), then fma(P, m², m) applies c1=1 and c0=0.
    Flt y = dispenso::fast_math::hornerEval(
        m,
        0.0924733654f,
        -0.14482744f,
        0.148145974f,
        -0.165455937f,
        0.199700251f,
        -0.250024557f,
        0.333337069f,
        -0.499999911f);
    y = FloatTraits<Flt>::fma(y, m * m, m); // c0=0, c1=1 absorbed

    y = FloatTraits<Flt>::fma(i, kLn2, y);

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
    // Degree-9 polynomial with c0=0 absorbed: y = P(m) * m, where
    // P(m) = c9*m^8 + ... + c1 (9 coefficients).
    Flt y = dispenso::fast_math::hornerEval(
                m,
                0.0326662175f,
                -0.0601400957f,
                0.0659840927f,
                -0.0723559037f,
                0.086600922f,
                -0.108560629f,
                0.144770443f,
                -0.217147425f,
                0.434294462f) *
        m;

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
 * @tparam AccuracyTraits Default: 3 ULP. kMaxAccuracy: 2 ULP (degree-11 polynomial).
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

    auto z = detail::atan_poly<Flt, AccuracyTraits>(y_x);

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

/**
 * @brief Power function: x^y.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~1 ULP for scalar float (double-precision core),
 *   error scales as |y*log2(x)|*ln2 ULP for SIMD (~1-2 for moderate y).
 *   kMaxAccuracy: uses extended-precision log2 for ~1-2 ULP even for large y (SIMD only;
 *   scalar float always uses the double-precision core which subsumes this).
 *   kBoundsValues: full IEEE 754 special-case handling (NaN, Inf, zero, negative bases).
 * @param x Base (all float domain, including negative).
 * @param y Exponent (SIMD vector).
 * @return x^y. Negative x: returns -|x|^y for odd integer y, NaN for non-integer y.
 *   Compatible with all SIMD backends.
 *
 * Scalar float uses a table-based double-precision core (~2.5ns).
 * SIMD uses float-precision exp2/log2 with FMA error-free transform.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt pow(Flt x, NonDeduced<Flt> y) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return pow<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x), SimdType_t<Flt>(y)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar float: double-precision pow core (table-based log2 + exp2).
    // Uses integer checkint() for parity (~5 insns vs ~40 for float_is_int/odd).
    // Branchless sign XOR. Table entry i=9 is {1.0, 0.0} so pow(1,y) = 1 exactly.
    uint32_t ix = bit_cast<uint32_t>(x);
    uint32_t iy = bit_cast<uint32_t>(y);
    // sign_bias is 0x80000000u for odd-integer y with negative x, else 0.
    // Always XORed into the result (no-op when 0).
    uint32_t sign_bias = 0;
    // neg_nan: set if negative x with non-integer y (domain error → NaN).
    // Only used by the non-bounds path (deferred after core computation).
    uint32_t neg_nan = 0;

    if constexpr (AccuracyTraits::kBoundsValues) {
      // Combined fast-path test: detects any x or y that needs special handling.
      // First term: ix - 0x00800000 >= 0x7F000000 is true when ix < 0x00800000
      // (zero, subnormal) or ix >= 0x7F800000 (inf, NaN) — but also when ix
      // has the sign bit set (negative), because unsigned subtraction wraps
      // to a huge value. This means negative x enters the special block too,
      // avoiding a separate negative-x check.
      // Second term: 2u*iy - 1u >= 0xFEFFFFFF tests y = ±0, ±inf, or NaN
      // (see bit trick comments in pow_scalar_y_special).
      if (DISPENSO_EXPECT(
              ix - 0x00800000u >= 0x7f800000u - 0x00800000u ||
                  2u * iy - 1u >= 2u * 0x7f800000u - 1u,
              0)) {
        float special;
        if (2u * iy - 1u >= 2u * 0x7f800000u - 1u) {
          detail::pow_scalar_y_special(x, y, ix, iy, &special);
          return special;
        }
        if (detail::pow_scalar_x_special(ix, iy, sign_bias, &special))
          return special;
        // Fall through to core computation with adjusted ix.
      }
    } else {
      // Non-bounds path: one branch for the common case (positive x),
      // branchless checkint in the else (rare negative-x path avoids further mispredicts).
      if (DISPENSO_EXPECT(!(ix & 0x80000000u), 1)) {
        // Positive x: nothing to do.
      } else {
        ix &= 0x7fffffffu;
        int32_t yint = detail::checkint_branchless(iy);
        sign_bias = static_cast<uint32_t>(yint == 2) * 0x80000000u;
        neg_nan = static_cast<uint32_t>(yint == 0);
      }
    }

    float r;
    if constexpr (AccuracyTraits::kMaxAccuracy) {
      // Tableless core: logarithmSep handles subnormals internally (pre-scales
      // by 2^23), so pass the original |x| — not the bounds-adjusted ix.
      r = detail::pow_double_poly_core(std::fabs(x), y);
    } else {
      // Table core: uses ix directly. The bounds path above already normalized
      // subnormals (scale by 2^23, subtract 23 from exponent) so ix is valid.
      r = detail::pow_double_core(bit_cast<float>(ix), y);
    }

    // Branchless sign flip (no-op when sign_bias == 0).
    r = bit_cast<float>(bit_cast<uint32_t>(r) ^ sign_bias);
    // Non-bounds: apply deferred NaN for neg^non-int, and pow(x,0)=1.
    if constexpr (!AccuracyTraits::kBoundsValues) {
      if (neg_nan)
        r = std::numeric_limits<float>::quiet_NaN();
      if (y == 0.0f)
        r = 1.0f;
    }

    return r;
  } else {
    // SIMD path: double-precision core for all accuracy levels (~1 ULP).
    using UintT = UintType_t<Flt>;

    Flt ax = fabs(x);
    Flt r;

    // Width-dependent dispatch: table gathers are cheap at 4 lanes but
    // scale poorly; polynomial cost amortizes over more lanes.
    if constexpr (sizeof(Flt) <= 16) {
      r = detail::pow_double_hybrid_core(ax, y);
    } else {
      r = detail::pow_double_poly_core(ax, y);
    }
    if constexpr (AccuracyTraits::kBoundsValues) {
      detail::pow_simd_special_fixup(ax, y, r);
    }

    // Sign handling for negative x (SIMD, branchless per-lane):
    // The double core computes pow(|x|, y). For negative x we need:
    //   - odd integer y:     negate result (e.g. pow(-2, 3) = -8)
    //   - even integer y:    keep positive (e.g. pow(-2, 2) = 4)
    //   - non-integer y:     result is NaN (domain error)
    // float_is_int/float_is_odd test this via mantissa bit inspection
    // (branchless SIMD comparisons, no scalar checkint needed).
    UintT xsign = bit_cast<UintT>(x) & UintT(0x80000000u);
    auto negx = x < 0.0f;
    auto y_int = float_is_int(y);
    auto y_odd = float_is_odd(y);
    // sign_mask is 0x80000000 in lanes where x < 0 AND y is odd, else 0.
    // XORing into r negates exactly those lanes.
    UintT sign_mask = xsign & bool_as_mask<UintT>(y_odd);
    r = bit_cast<Flt>(bit_cast<UintT>(r) ^ sign_mask);
    // Replace with NaN where x < 0 and y is not an integer.
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    r = FloatTraits<Flt>::conditional(negx & !y_int, Flt(kNaN), r);

    // pow(x, 0) = 1 for all x (IEEE 754).
    r = FloatTraits<Flt>::conditional(y == 0.0f, Flt(1.0f), r);

    if constexpr (AccuracyTraits::kBoundsValues) {
      // pow(1, y) = 1 even for NaN y.
      r = FloatTraits<Flt>::conditional(x == 1.0f, Flt(1.0f), r);
      // pow(-1, ±inf) = 1.
      constexpr float kInf = std::numeric_limits<float>::infinity();
      auto x_neg1 = (x == -1.0f);
      auto y_inf = fabs(y) == Flt(kInf);
      r = FloatTraits<Flt>::conditional(x_neg1 & y_inf, Flt(1.0f), r);
    }

    return r;
  }
}

/**
 * @brief Power function with scalar exponent: x^y.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Same as pow(Flt, Flt).
 * @param x Base (all float domain, including negative).
 * @param y Scalar exponent — enables scalar branching optimizations.
 * @return x^y. Same semantics as pow(Flt, Flt).
 *   Compatible with all SIMD backends.
 *
 * Scalar y enables fast paths: y=0→1, y=1→x, y=-1→1/x, y=0.5→sqrt,
 * y=2→x*x, integer y→binary squaring. General case uses exp2/log2.
 */
template <
    typename Flt,
    typename AccuracyTraits = DefaultAccuracyTraits,
    typename = std::enable_if_t<!std::is_same_v<Flt, float>>>
DISPENSO_INLINE Flt pow(Flt x, float y) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return pow<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x), y).v;
  } else {
    using UintT = UintType_t<Flt>;

    // Scalar fast paths: y=0→1, y=1→x, y=-1→1/x, y=0.5→sqrt, y=2→x*x,
    // small integer y → binary squaring.
    bool y_is_int = false, y_is_odd = false;
    Flt fast_result;
    if (detail::pow_scalar_y_fast_path(x, y, y_is_int, y_is_odd, fast_result))
      return fast_result;

    Flt ax = fabs(x);
    Flt r;
    Flt yv(y);

    // Double-precision core for all SIMD accuracy levels (~1 ULP).
    if constexpr (sizeof(Flt) <= 16) {
      r = detail::pow_double_hybrid_core(ax, yv);
    } else {
      r = detail::pow_double_poly_core(ax, yv);
    }
    if constexpr (AccuracyTraits::kBoundsValues) {
      detail::pow_simd_special_fixup(ax, yv, r);
    }

    // Sign handling for negative x.
    if (y_is_odd) {
      UintT xsign = bit_cast<UintT>(x) & UintT(0x80000000u);
      r = bit_cast<Flt>(bit_cast<UintT>(r) ^ xsign);
    }
    if (!y_is_int) {
      auto negx = x < 0.0f;
      constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
      r = FloatTraits<Flt>::conditional(negx, Flt(kNaN), r);
    }

    if constexpr (AccuracyTraits::kBoundsValues) {
      r = FloatTraits<Flt>::conditional(x == 1.0f, Flt(1.0f), r);
      constexpr float kInf = std::numeric_limits<float>::infinity();
      if (y == kInf || y == -kInf) {
        auto x_neg1 = (x == -1.0f);
        r = FloatTraits<Flt>::conditional(x_neg1, Flt(1.0f), r);
      }
    }

    return r;
  }
}

/**
 * @brief Compute exp(x) - 1 with precision near zero.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~2 ULP (1-step CW + degree-4 poly).
 *   MaxAccuracy: 1 ULP (2-step CW + degree-5 poly).
 *   kBoundsValues: 1 ULP + handles NaN/inf.
 * @param x Input value (all float domain).
 * @return exp(x) - 1. Compatible with all SIMD backends.
 *
 * Uses Cody-Waite range reduction: x = n*ln2 + r, |r| <= ln2/2.
 * Then expm1(x) = 2^n * expm1(r) + (2^n - 1), where expm1(r) is computed
 * via a Sollya fpminimax polynomial. Avoids the catastrophic cancellation of
 * exp(x) - 1 near zero, and provides uniform accuracy across the entire
 * domain (no polynomial/fallback transition).
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt expm1(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return expm1<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    using IntT = IntType_t<Flt>;
    using UintT = UintType_t<Flt>;
    auto fma_fn = FloatTraits<Flt>::fma;

    // Cody-Waite range reduction: x = n*ln2 + r, |r| <= ln2/2.
    // 2-step CW is required for all accuracy levels because the
    // reconstruction 2^n * expm1(r) amplifies reduction error by 2^n.
    constexpr float k1_ln2 = k1_Ln2;
    constexpr float kLn2hi = 6.93145752e-1f;
    constexpr float kLn2lo = 1.42860677e-6f;
    Flt r = x;
    Flt jf = detail::rangeReduce(r, k1_ln2, kLn2hi, kLn2lo);
    IntT n = convert_to_int(jf);

    Flt em1_r;

    if constexpr (AccuracyTraits::kMaxAccuracy || AccuracyTraits::kBoundsValues) {
      // Degree-5 Sollya fpminimax((exp(x)-1-x)/x^2, 5, [|SG...|], [-ln2/2, ln2/2]):
      //   sup-norm error < 2^-29 (~0.03 ULP at r = ln2/2).
      Flt p = dispenso::fast_math::hornerEval(
          r,
          0x1.a26762p-13f,
          0x1.6d2ep-10f,
          0x1.110ff2p-7f,
          0x1.555502p-5f,
          0x1.555556p-3f,
          0x1p-1f);
      em1_r = fma_fn(p, r * r, r);
    } else {
      // Degree-4 Sollya fpminimax((exp(x)-1-x)/x^2, 4, [|SG...|], [-ln2/2, ln2/2]):
      //   sup-norm error < 2^-24 (~1.2 ULP at r = ln2/2).
      Flt p = dispenso::fast_math::hornerEval(
          r, 0x1.6ca992p-10f, 0x1.120abep-7f, 0x1.55556cp-5f, 0x1.5554dep-3f, 0x1p-1f);
      em1_r = fma_fn(p, r * r, r);
    }

    // Reconstruction: expm1(x) = 2^n * expm1(r) + (2^n - 1).
    // 2^n is exact for integer n. 2^n - 1 is exact for |n| <= 23.
    // For |n| > 24, 2^n - 1 = 2^n in float, so expm1 ≈ exp which is correct.
    // Use fma for precision: fma(2^n, expm1(r), 2^n - 1).
    Flt two_n = bit_cast<Flt>(UintT(n + 127) << 23);
    Flt two_n_m1 = two_n - 1.0f;
    Flt result = fma_fn(two_n, em1_r, two_n_m1);

    if constexpr (std::is_same_v<Flt, float>) {
      // For n == 0 (|x| < ln2/2 ≈ 0.347): return polynomial directly.
      if (n == 0)
        return em1_r;
      // For x < -25*ln2 ≈ -17.3: expm1(x) rounds to -1 in float.
      if (x < -17.5f)
        return -1.0f;
      // For x > 89: exp(x) overflows, expm1(x) = inf.
      if (x > 89.0f)
        return std::numeric_limits<float>::infinity();
      // NaN propagation: rangeReduce maps NaN to finite r, producing a
      // garbage result.  x - x is 0 for finite, NaN for NaN.
      if constexpr (AccuracyTraits::kBoundsValues)
        return result + (x - x);
      return result;
    } else {
      // SIMD: blend n==0 path (direct polynomial) with reconstruction.
      auto zero_n = (n == IntT(0));
      result = FloatTraits<Flt>::conditional(zero_n, em1_r, result);
      // Clamp: x < -17.5 → -1, x > 89 → inf.
      result = FloatTraits<Flt>::conditional(x < -17.5f, Flt(-1.0f), result);
      constexpr float kInf = std::numeric_limits<float>::infinity();
      result = FloatTraits<Flt>::conditional(x > 89.0f, Flt(kInf), result);

      if constexpr (AccuracyTraits::kBoundsValues) {
        // NaN propagation: range reduction maps NaN to finite values.
        auto is_nan = (bit_cast<UintT>(x) & UintT(0x7fffffffu)) > UintT(0x7f800000u);
        result = FloatTraits<Flt>::conditional(is_nan, x, result);
      }

      return result;
    }
  }
}

/**
 * @brief Compute log(1 + x) with precision near zero.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~2 ULP. kBoundsValues: handles x = -1 (→ -inf), NaN.
 * @param x Input value in (-1, +inf).
 * @return log(1 + x). Compatible with all SIMD backends.
 *
 * For |x| < 0.25: direct Sollya polynomial avoids the log() call (~1 ULP).
 * For |x| >= 0.25: compensated-addition trick: u = 1 + x, c = x - (u - 1)
 * captures the rounding error. Then log(u) is computed inline via the same
 * range reduction as log() (logarithmSep), and the result is log(u) + c/u.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt log1p(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return log1p<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else {
    auto fma_fn = FloatTraits<Flt>::fma;

    // Direct polynomial for log1p(x) on [-0.25, 0.25]:
    //   log1p(x) = x - x²/2 + x³/3 - ... ≈ x + x² * q(x)
    // Sollya fpminimax((log(1+x)-x)/x^2, 6, [|SG...|], [-0.25, 0.25]):
    //   sup-norm error < 2^-23; at x=0.25 → ~0.25 ULP polynomial error.
    Flt q = dispenso::fast_math::hornerEval(
        x,
        -0x1.1957b2p-3f,
        0x1.3f347cp-3f,
        -0x1.5472d2p-3f,
        0x1.98bfaap-3f,
        -0x1.000112p-2f,
        0x1.555632p-2f,
        -0x1p-1f);
    Flt poly_r = fma_fn(q, x * x, x); // log1p(x) ≈ x + x² * q(x)

    if constexpr (std::is_same_v<Flt, float>) {
      // Scalar: branch on magnitude.
      if (std::fabs(x) < 0.25f)
        return poly_r;
    }

    // Compensated addition: u = float(1 + x), c = rounding error.
    // log(1 + x) = log(u + c) = log(u) + log(1 + c/u) ≈ log(u) + c/u.
    Flt u = Flt(1.0f) + x;
    Flt c = x - (u - 1.0f);

    // Inline log(u): range-reduce u via logarithmSep, evaluate polynomial.
    auto [xi, i, m] = detail::logarithmSep<Flt, AccuracyTraits>(u);
    m = m - 1.0f;

    // log(1+m) polynomial for m in [sqrt(0.5)-1, sqrt(2.0)-1].
    // Same coefficients as log(). Degree-9 with c0=0, c1=1 absorbed.
    // p(m) = m + m² * P(m). polyEval uses Estrin on CPU (~27% faster),
    // Horner on GPU (better for in-order pipelines).
    Flt inner = dispenso::fast_math::polyEval(
        m,
        0.0924733654f,
        -0.14482744f,
        0.148145974f,
        -0.165455937f,
        0.199700251f,
        -0.250024557f,
        0.333337069f,
        -0.499999911f);
    Flt y = fma_fn(inner, m * m, m); // c0=0, c1=1 absorbed

    y = fma_fn(i, kLn2, y);

    // Add compensation: log1p(x) = log(u) + c/u.
    Flt result = y + c / u;

    if constexpr (!std::is_same_v<Flt, float>) {
      // SIMD: blend polynomial for small |x|, compensated log otherwise.
      auto small = fabs(x) < 0.25f;
      result = FloatTraits<Flt>::conditional(small, poly_r, result);
    }

    if constexpr (AccuracyTraits::kBoundsValues) {
      // log1p(-1) → -inf, log1p(x < -1) → NaN, log1p(NaN) → NaN.
      result = detail::logarithmBounds(u, result, xi);
    }

    return result;
  }
}

/**
 * @brief Hyperbolic tangent approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~2 ULP. kBoundsValues: handles NaN.
 * @param x Input value (all float domain).
 * @return tanh(x) in [-1, 1]. Compatible with all SIMD backends.
 *
 * tanh(x) = expm1(2x) / (expm1(2x) + 2).
 * The range-reduced expm1 preserves precision near zero (expm1(2x) ≈ 2x),
 * so this formula naturally gives tanh(x) ≈ x for small x without a
 * separate polynomial. Input clamped to [-10, 10] since tanh(10) = 1.0f
 * exactly in float, and the clamp prevents expm1 overflow for large |x|.
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt tanh(Flt x) {
  assert_float_type<Flt>();
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return tanh<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar: Sollya fpminimax polynomial for |x| < 1.0 (degree 6 in u = x²).
    // tanh(x) = x * (1 + u * Q(u)), where Q is fitted to tanh(x)/x - 1.
    // Error: ~0.52 ULP at the boundary. 6 FMAs.
    float ax = std::fabs(x);
    if (!(ax >= 1.0f)) { // !(>=) so NaN enters polynomial path and propagates
      constexpr float t1 = -0x1.555482p-2f;
      constexpr float t2 = 0x1.10ef12p-3f;
      constexpr float t3 = -0x1.b66cecp-5f;
      constexpr float t4 = 0x1.4e3a6ap-6f;
      constexpr float t5 = -0x1.9c954p-8f;
      constexpr float t6 = 0x1.189ec2p-10f;
      float u = x * x;
      float poly =
          std::fma(std::fma(std::fma(std::fma(std::fma(t6, u, t5), u, t4), u, t3), u, t2), u, t1);
      return x * std::fma(poly, u, 1.0f);
    }
    // Large |x|: use expm1 formula (no cancellation, result far from 0).
    float x_safe = ax < 10.0f ? x : (x < 0.0f ? -10.0f : 10.0f);
    float em1 = expm1<float, AccuracyTraits>(x_safe + x_safe);
    return em1 / (em1 + 2.0f);
  } else {
    // SIMD: pure expm1 formula (branchless, uniform across all lanes).
    Flt x_safe = clamp_no_nan(x, Flt(-10.0f), Flt(10.0f));
    Flt em1 = expm1<Flt, AccuracyTraits>(x_safe + x_safe);
    Flt result = em1 / (em1 + 2.0f);

    if constexpr (AccuracyTraits::kBoundsValues) {
      // NaN propagation: clamp_no_nan may map NaN to a finite value.
      using UintT = UintType_t<Flt>;
      auto is_nan = (bit_cast<UintT>(x) & UintT(0x7fffffffu)) > UintT(0x7f800000u);
      result = FloatTraits<Flt>::conditional(is_nan, x, result);
    }

    return result;
  }
}

/**
 * @brief Error function approximation.
 * @tparam Flt float or SIMD float type.
 * @tparam AccuracyTraits Default: ~2 ULP. Not used for bounds (NaN propagates naturally).
 * @param x Input value (all float domain).
 * @return erf(x) in [-1, 1]. Compatible with all SIMD backends.
 *
 * Abramowitz & Stegun 7.1.26-inspired t-substitution with Sollya-optimized
 * coefficients. Uses p=0.45 (vs A&S's 0.3275911) for better polynomial fit.
 *
 * Two domains:
 *   |x| < 0.875:  erf(x) = x * (c0 + x² * Q(x²)),  pure polynomial.
 *   |x| ∈ [0.875, 3.92]:  erf(x) = 1 - t·P(t)·exp(-x²),  t = 1/(1+px).
 *   |x| >= 3.92:  erf(x) = ±1  (saturated in float).
 */
template <typename Flt, typename AccuracyTraits = DefaultAccuracyTraits>
DISPENSO_INLINE Flt erf(Flt x) {
  assert_float_type<Flt>();
  // AccuracyTraits accepted for API consistency; erf uses the same polynomial for all traits.
  (void)sizeof(AccuracyTraits);
  if constexpr (!std::is_same_v<Flt, SimdType_t<Flt>>) {
    return erf<SimdType_t<Flt>, AccuracyTraits>(SimdType_t<Flt>(x)).v;
  } else if constexpr (std::is_same_v<Flt, float>) {
    // Scalar path: branching for efficiency.
    // Save sign and work with |x|; restore sign at end via bit-OR.
    uint32_t sign = bit_cast<uint32_t>(x) & 0x80000000u;
    float ax = dispenso::fast_math::fabs(x);
    float result;
    if (ax >= 3.92f) {
      result = 1.0f;
    } else if (!(ax < 0.875f)) { // !(< ) so NaN goes to erfc path and propagates
      // erfc formula: erf(x) = 1 - t * P(t) * exp(-x²), t = 1/(1+p*x).
      constexpr float p = 0.45f;
      float t = 1.0f / std::fma(p, ax, 1.0f);

      // Sollya fpminimax degree 5 P(t), float coefficients.
      constexpr float c0 = 0x1.04873ep-2f;
      constexpr float c1 = 0x1.f81fc6p-3f;
      constexpr float c2 = 0x1.189f42p-2f;
      constexpr float c3 = 0x1.15aaa6p-5f;
      constexpr float c4 = 0x1.65d24ep-2f;
      constexpr float c5 = -0x1.4432a4p-3f;
      float poly = t *
          std::fma(std::fma(std::fma(std::fma(std::fma(c5, t, c4), t, c3), t, c2), t, c1), t, c0);

      // Inline exp(-x²) via Cody-Waite range reduction.
      float u = ax * ax;
      constexpr float kLog2e = 0x1.715476p+0f;
      constexpr float kLn2hi = 0x1.62e400p-1f;
      constexpr float kLn2lo = 0x1.7f7d1cp-20f;
      float k = std::floor(u * kLog2e);
      float f = std::fma(k, -kLn2hi, u);
      f = std::fma(k, -kLn2lo, f);
      // Degree-5 Horner for exp(-f), f in [0, ln2).
      constexpr float e0 = 0x1.fffffep-1f, e1 = -0x1.ffff1ep-1f;
      constexpr float e2 = 0x1.ffe314p-2f, e3 = -0x1.53f876p-3f;
      constexpr float e4 = 0x1.462f16p-5f, e5 = -0x1.80e5b2p-8f;
      float exp_neg_f =
          std::fma(std::fma(std::fma(std::fma(std::fma(e5, f, e4), f, e3), f, e2), f, e1), f, e0);
      int32_t ki = convert_to_int_trunc_safe(k);
      float pow2_neg_k = bit_cast<float>((127 - ki) << 23);

      result = 1.0f - pow2_neg_k * exp_neg_f * poly;
    } else {
      // Near-zero: erf(x) = x * (c0 + x² * Q(x²)).
      // Sollya fpminimax degree 5 Q(u), float coefficients.
      constexpr float c0 = 0x1.20dd76p+0f; // ≈ 2/√π
      constexpr float q0 = -0x1.812746p-2f, q1 = 0x1.ce2ec6p-4f, q2 = -0x1.b81edep-6f;
      constexpr float q3 = 0x1.556b48p-8f, q4 = -0x1.b0255p-11f, q5 = 0x1.7149c8p-14f;
      float u = ax * ax;
      float q =
          std::fma(std::fma(std::fma(std::fma(std::fma(q5, u, q4), u, q3), u, q2), u, q1), u, q0);
      result = ax * std::fma(q, u, c0);
    }
    return bit_cast<float>(bit_cast<uint32_t>(result) | sign);
  } else {
    // SIMD: branchless, compute both paths for all lanes and blend.
    using UintT = UintType_t<Flt>;
    using IntT = IntType_t<Flt>;
    auto fma_fn = FloatTraits<Flt>::fma;

    // clamp_allow_nan preserves NaN so it propagates through the computation.
    Flt ax = clamp_allow_nan(fabs(x), Flt(0.0f), Flt(3.92f));

    // --- Near-zero path: erf(x) = x * (c0 + x² * Q(x²)) ---
    Flt u_near = ax * ax;
    constexpr float c0_near = 0x1.20dd76p+0f;
    Flt q_near = hornerEval(
        u_near,
        0x1.7149c8p-14f,
        -0x1.b0255p-11f,
        0x1.556b48p-8f,
        -0x1.b81edep-6f,
        0x1.ce2ec6p-4f,
        -0x1.812746p-2f);
    Flt near_result = ax * fma_fn(q_near, u_near, Flt(c0_near));

    // --- erfc path: erf(x) = 1 - t * P(t) * exp(-x²) ---
    constexpr float p = 0.45f;
    Flt denom = fma_fn(Flt(p), ax, Flt(1.0f));
    Flt t = Flt(1.0f) / denom;

    Flt erfc_poly = t *
        hornerEval(t,
                   -0x1.4432a4p-3f,
                   0x1.65d24ep-2f,
                   0x1.15aaa6p-5f,
                   0x1.189f42p-2f,
                   0x1.f81fc6p-3f,
                   0x1.04873ep-2f);

    // Inline exp(-x²) via Cody-Waite range reduction.
    Flt u = ax * ax;
    constexpr float kLog2e = 0x1.715476p+0f;
    constexpr float kLn2hi = 0x1.62e400p-1f;
    constexpr float kLn2lo = 0x1.7f7d1cp-20f;
    Flt k = floor_small(u * kLog2e);
    Flt f = fma_fn(k, Flt(-kLn2hi), u);
    f = fma_fn(k, Flt(-kLn2lo), f);
    Flt exp_neg_f = hornerEval(
        f,
        -0x1.80e5b2p-8f,
        0x1.462f16p-5f,
        -0x1.53f876p-3f,
        0x1.ffe314p-2f,
        -0x1.ffff1ep-1f,
        0x1.fffffep-1f);
    IntT ki = convert_to_int(k);
    Flt pow2_neg_k = bit_cast<Flt>(UintT((IntT(127) - ki) << 23));

    Flt erfc_result = Flt(1.0f) - pow2_neg_k * exp_neg_f * erfc_poly;

    // Blend: near-zero for |x| < 0.875, erfc otherwise.
    auto small = ax < Flt(0.875f);
    Flt result = FloatTraits<Flt>::conditional(small, near_result, erfc_result);

    // Restore sign from original x (odd function). Result is non-negative, so OR stamps sign.
    result = bit_cast<Flt>(bit_cast<UintT>(result) | (bit_cast<UintT>(x) & UintT(0x80000000u)));

    return result;
  }
}

} // namespace fast_math
} // namespace dispenso
