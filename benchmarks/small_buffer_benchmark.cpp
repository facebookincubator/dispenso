// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/small_buffer_allocator.h>

#include "benchmark_common.h"

constexpr size_t kSmallSize = 32;
constexpr size_t kMediumSize = 128;
constexpr size_t kLargeSize = 256;

template <typename Alloc, typename Free>
void run(benchmark::State& state, Alloc alloc, Free dealloc) {
  std::vector<char*> ptrs(state.range(0));
  for (auto UNUSED_VAR : state) {
    for (char*& p : ptrs) {
      p = alloc();
    }
    for (char* p : ptrs) {
      dealloc(p);
    }
  }
}

template <size_t kSize>
void BM_newdelete(benchmark::State& state) {
  run(
      state, []() { return new char[kSize]; }, [](char* buf) { delete[](buf); });
}

template <size_t kSize>
void BM_small_buffer_allocator(benchmark::State& state) {
  run(
      state,
      []() { return dispenso::allocSmallBuffer<kSize>(); },
      [](char* buf) { dispenso::deallocSmallBuffer<kSize>(buf); });
}

BENCHMARK_TEMPLATE(BM_newdelete, kSmallSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kSmallSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_newdelete, kMediumSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kMediumSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_newdelete, kLargeSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kLargeSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_newdelete, kSmallSize)->Threads(16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kSmallSize)->Threads(16)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_newdelete, kMediumSize)->Threads(16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kMediumSize)->Threads(16)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_newdelete, kLargeSize)->Threads(16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_small_buffer_allocator, kLargeSize)->Threads(16)->Range(1 << 13, 1 << 15);

BENCHMARK_MAIN();
