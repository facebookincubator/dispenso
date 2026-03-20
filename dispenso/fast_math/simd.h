/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Convenience header that provides DefaultSimdFloat — the widest available
// SIMD float type for the current platform. Include this header and use
// DefaultSimdFloat in generic code that should auto-vectorize to the best
// available SIMD width.
//
// Example:
//   #include <dispenso/fast_math/simd.h>
//   using F = dispenso::fast_math::DefaultSimdFloat;
//   F result = dispenso::fast_math::sin(F(1.0f));

#pragma once

#include <dispenso/fast_math/fast_math.h>

namespace dispenso {
namespace fast_math {

// On AArch64, prefer native NEON over Highway — NEON has lower abstraction
// overhead at the same 4-lane width. On x86, prefer Highway for runtime
// dispatch to the widest available ISA (up to AVX-512).
#if defined(__aarch64__)
using DefaultSimdFloat = NeonFloat;
#elif __has_include("hwy/highway.h")
using DefaultSimdFloat = HwyFloat;
#elif defined(__AVX512F__)
using DefaultSimdFloat = Avx512Float;
#elif defined(__AVX2__)
using DefaultSimdFloat = AvxFloat;
#elif defined(__SSE4_1__)
using DefaultSimdFloat = SseFloat;
#else
using DefaultSimdFloat = float;
#endif

} // namespace fast_math
} // namespace dispenso
