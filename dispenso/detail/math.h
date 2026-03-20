/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <type_traits>

#if defined(_WIN32)
#include <intrin.h>
#endif //_WIN32

namespace dispenso {

namespace detail {

constexpr uint64_t nextPow2(uint64_t v) {
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v++;
  return v;
}

constexpr inline uint32_t log2const(uint64_t v) {
  constexpr uint64_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000, 0xFFFFFFFF00000000UL};
  constexpr uint32_t S[] = {1, 2, 4, 8, 16, 32};

  uint32_t r = 0;
  for (uint32_t i = 6; i--;) {
    if (v & b[i]) {
      v >>= S[i];
      r |= S[i];
    }
  }

  return r;
}

constexpr inline uint32_t log2const(uint32_t v) {
  constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  constexpr uint32_t S[] = {1, 2, 4, 8, 16};

  uint32_t r = 0;
  for (uint32_t i = 5; i--;) {
    if (v & b[i]) {
      v >>= S[i];
      r |= S[i];
    }
  }

  return r;
}

// On some platforms (e.g. macOS ARM64), size_t is 'unsigned long' which is a distinct type
// from both uint32_t and uint64_t, causing overload ambiguity. This template resolves it
// by delegating to the 64-bit overload; it is SFINAE-disabled on platforms where unsigned
// long already matches one of the fixed-width types (Linux, Windows), so no behavior change.
template <
    typename T = unsigned long,
    typename std::enable_if<
        !std::is_same<T, uint32_t>::value && !std::is_same<T, uint64_t>::value,
        int>::type = 0>
constexpr inline uint32_t log2const(T v) {
  return log2const(static_cast<uint64_t>(v));
}

// --- 64-bit log2 ---

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
inline uint32_t log2(uint64_t v) {
  uint64_t result;
  __asm__("bsrq %1, %0" : "=r"(result) : "r"(v));
  return static_cast<uint32_t>(result);
}
#elif (defined(__GNUC__) || defined(__clang__))
inline uint32_t log2(uint64_t v) {
  return static_cast<uint32_t>(63 - __builtin_clzll(v));
}
#elif defined(_WIN64)
inline uint32_t log2(uint64_t v) {
  unsigned long index;
  _BitScanReverse64(&index, v);
  return static_cast<uint32_t>(index);
}
#elif defined(_WIN32)
inline uint32_t log2(uint64_t v) {
  unsigned long index;
  uint32_t hi = static_cast<uint32_t>(v >> 32);
  if (hi != 0) {
    _BitScanReverse(&index, hi);
    return static_cast<uint32_t>(index + 32);
  }
  _BitScanReverse(&index, static_cast<uint32_t>(v));
  return static_cast<uint32_t>(index);
}
#else
inline uint32_t log2(uint64_t v) {
  return log2const(v);
}
#endif // 64-bit

template <
    typename T = unsigned long,
    typename std::enable_if<
        !std::is_same<T, uint32_t>::value && !std::is_same<T, uint64_t>::value,
        int>::type = 0>
inline uint32_t log2(T v) {
  return log2(static_cast<uint64_t>(v));
}

// --- 32-bit log2 ---

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
inline uint32_t log2(uint32_t v) {
  uint32_t result;
  __asm__("bsrl %1, %0" : "=r"(result) : "r"(v));
  return result;
}
#elif (defined(__GNUC__) || defined(__clang__))
inline uint32_t log2(uint32_t v) {
  return static_cast<uint32_t>(31 - __builtin_clz(v));
}
#elif defined(_WIN32)
inline uint32_t log2(uint32_t v) {
  unsigned long index;
  _BitScanReverse(&index, v);
  return static_cast<uint32_t>(index);
}
#else
inline uint32_t log2(uint32_t v) {
  return log2const(v);
}
#endif // 32-bit

} // namespace detail
} // namespace dispenso
