// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <stdint.h>

#if defined(_WIN32)
#include <intrin.h>
#endif //_WIN32

namespace dispenso {

namespace detail {

constexpr uint64_t nextPow2(uint64_t v) {
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
  for (size_t i = 6; i--;) {
    if (v & b[i]) {
      v >>= S[i];
      r |= S[i];
    }
  }

  return r;
}

#if (defined(__GNUC__) || defined(__clang__))
inline int log2(uint64_t v) {
  return 63 - __builtin_clzll(v);
}
#elif defined(_WIN32)
inline int log2(uint64_t v) {
  return 63 - __lzcnt64(v);
}
#else
inline uint32_t log2(uint64_t v) {
  return log2const(v);
}

#endif // PLATFORM

} // namespace detail
} // namespace dispenso
