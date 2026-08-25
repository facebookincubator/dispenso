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
#elif defined(__x86_64__)
#include <cpuid.h>
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

// Whether the timestamp advances at a fixed rate: CPUID.80000007H:EDX[8]
// (Invariant TSC). Without it the counter tracks the core clock and changes
// rate with P-state and C-state transitions, so a frequency measured once at
// startup silently stops being correct. Callers that convert ticks to seconds
// must consult this and fall back to a kernel-maintained clock when it is
// false.
inline bool hasInvariantTimestamp() {
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 0x80000000);
  if (static_cast<uint32_t>(regs[0]) < 0x80000007u) {
    return false;
  }
  __cpuid(regs, 0x80000007);
  return (static_cast<uint32_t>(regs[3]) & (1u << 8)) != 0;
#else
  // gcc declares __get_cpuid_max as returning unsigned, clang as returning
  // int, so the comparison needs an explicit type to be warning-free on both.
  const uint32_t maxExtendedLeaf = static_cast<uint32_t>(__get_cpuid_max(0x80000000u, nullptr));
  if (maxExtendedLeaf < 0x80000007u) {
    return false;
  }
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid(0x80000007u, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
  return (edx & (1u << 8)) != 0;
#endif
}

#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define DISPENSO_HAS_TIMESTAMP

// aarch64 virtual counter (cntvct_el0). No separate serializing variant.
inline uint64_t timestamp() {
  uint64_t val = 0;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}

// cntvct_el0 advances at the fixed rate reported by cntfrq_el0, so unlike the
// x86 TSC there is no non-invariant case to guard against.
inline bool hasInvariantTimestamp() {
  return true;
}

#endif // ARCH

} // namespace detail
} // namespace dispenso
