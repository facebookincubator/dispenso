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

// DefaultSimdFloat is defined in util.h (after SIMD backend includes).
// This header just pulls in fast_math.h which includes util.h.
#include <dispenso/fast_math/fast_math.h>
