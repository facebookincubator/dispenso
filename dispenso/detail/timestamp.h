/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// INTERNAL ONLY. Cheap monotonic timestamp wrapper for paths that want raw
// cycle counts without going through std::chrono. Serializing variant only —
// reorder-safe on x86 (lfence;rdtsc) and uses cntvct_el0 on aarch64.
// DISPENSO_HAS_TIMESTAMP is defined when a timestamp is available on the
// target platform.

#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif // _MSC_VER

namespace dispenso {
namespace detail {

#if defined(__x86_64__) || defined(_M_AMD64)
#define DISPENSO_HAS_TIMESTAMP

// Serializing: ~25-30 cycles; waits for prior instructions to retire before
// sampling. Use for calibration or interval measurement where bias from
// out-of-order execution matters.
#if defined(_MSC_VER)
inline uint64_t timestamp() {
  _mm_lfence();
  return __rdtsc();
}
#else
inline uint64_t timestamp() {
  uint32_t lo, hi;
  __asm__ volatile("lfence\n\trdtsc"
                   : /* outputs */ "=a"(lo), "=d"(hi)
                   : /* inputs */
                   : /* clobbers */ "memory");
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}
#endif

#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define DISPENSO_HAS_TIMESTAMP

// aarch64 virtual counter (cntvct_el0). No separate serializing variant.
inline uint64_t timestamp() {
  uint64_t val;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}

#endif // ARCH

} // namespace detail
} // namespace dispenso
