/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Central SIMD feature detection for dispenso::fast_math.
//
// Compilers disagree about how they advertise x86 SIMD support:
//   - GCC and Clang define __SSE4_1__, __AVX2__ and __AVX512F__ only when the
//     matching -m flag is passed.
//   - MSVC defines __AVX__/__AVX2__/__AVX512F__ for /arch:AVX and above, but
//     never defines any __SSE*__ macro. On x64 it treats SSE2 as an
//     unconditional baseline and accepts any intrinsic you write regardless of
//     /arch, so an SSE4.1 intrinsic compiles cleanly into an SSE2-targeted
//     binary and faults at run time on hardware that lacks it.
//
// Gating on __SSE4_1__ alone therefore disables every x86 fast path on Windows
// silently. dispenso treats SSE4.1 (Penryn, 2007) as its x86 baseline, so on
// MSVC x64 we assume it is available. That assumption is deliberate: MSVC
// offers no compile-time signal that would let us detect a pre-SSE4.1 x64 CPU,
// so such a machine must opt out explicitly via
// DISPENSO_FORCE_TEST_SCALAR_PORTABLE_FALLBACK rather than being detected.
//
// DISPENSO_FORCE_TEST_SCALAR_PORTABLE_FALLBACK disables every SIMD backend. Besides being
// the escape hatch above, it exists so the scalar fallback can be tested at
// all: every machine that can run the test suite has SIMD, so those paths are
// otherwise unreachable and regress unnoticed.

#if defined(DISPENSO_FORCE_TEST_SCALAR_PORTABLE_FALLBACK) && \
    DISPENSO_FORCE_TEST_SCALAR_PORTABLE_FALLBACK

#define DISPENSO_FAST_MATH_HAS_SSE4_1 0
#define DISPENSO_FAST_MATH_HAS_AVX2 0
#define DISPENSO_FAST_MATH_HAS_AVX512F 0
#define DISPENSO_FAST_MATH_HAS_NEON 0
// Scalar FloatTraits<float>::fma uses std::fma, which is genuinely fused, so
// the portable path keeps FMA semantics even with every backend disabled.
#define DISPENSO_FAST_MATH_HAS_FMA 1

#else

// __AVX__ implies SSE4.2, which implies SSE4.1; _M_X64 is the MSVC baseline
// assumption described above.
#if defined(__SSE4_1__) || defined(__AVX__) || defined(_M_X64)
#define DISPENSO_FAST_MATH_SSE4_1_PRESENT 1
#else
#define DISPENSO_FAST_MATH_SSE4_1_PRESENT 0
#endif

#if defined(__AVX2__)
#define DISPENSO_FAST_MATH_HAS_AVX2 1
#else
#define DISPENSO_FAST_MATH_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
#define DISPENSO_FAST_MATH_HAS_AVX512F 1
#else
#define DISPENSO_FAST_MATH_HAS_AVX512F 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define DISPENSO_FAST_MATH_HAS_NEON 1
#else
#define DISPENSO_FAST_MATH_HAS_NEON 0
#endif

// Fused multiply-add. MSVC does not define __FMA__ either, so gating on it
// alone silently substitutes a separate multiply and add on Windows. That is
// not merely slower: the two-term Cody-Waite reductions in fast_math depend on
// fusion, because the unfused product rounds away exactly the bits the low
// correction term exists to cancel. AVX2 implies FMA3 on every part that has
// it and on both compilers, and AArch64 always has FMA.
#if defined(__FMA__) || DISPENSO_FAST_MATH_HAS_AVX2 || DISPENSO_FAST_MATH_HAS_NEON
#define DISPENSO_FAST_MATH_HAS_FMA 1
#else
#define DISPENSO_FAST_MATH_HAS_FMA 0
#endif

// The SSE backend additionally requires FMA. SSE4.1 is from 2007 and FMA3 from
// 2013, so the combination SSE4.1-without-FMA is a real configuration -- and
// not a usable one. The Cody-Waite reductions assume single rounding, and the
// trig constants are not truncated at all (kPi_2hi keeps 23 significant bits,
// so n*hi is exact only for |n| < 2). Measured on SSE4.1 without FMA, sin and
// cos exceed 3.9e6 and 5.6e6 ULP beyond +-pi -- garbage rather than degraded.
// Such hardware therefore uses the scalar path, where std::fma really is fused.
// AVX2, AVX-512 and NEON all imply FMA, so only SSE needs this guard.
#if DISPENSO_FAST_MATH_SSE4_1_PRESENT && DISPENSO_FAST_MATH_HAS_FMA
#define DISPENSO_FAST_MATH_HAS_SSE4_1 1
#else
#define DISPENSO_FAST_MATH_HAS_SSE4_1 0
#endif

#endif // DISPENSO_FORCE_TEST_SCALAR_PORTABLE_FALLBACK

// True when an x86 intrinsics header is required.
#if !defined(__CUDACC__) &&                                          \
    (DISPENSO_FAST_MATH_HAS_SSE4_1 || DISPENSO_FAST_MATH_HAS_AVX2 || \
     DISPENSO_FAST_MATH_HAS_AVX512F)
#define DISPENSO_FAST_MATH_X86_INTRIN 1
#else
#define DISPENSO_FAST_MATH_X86_INTRIN 0
#endif
