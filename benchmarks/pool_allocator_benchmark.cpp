/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/pool_allocator.h>

#include <deque>

#include <dispenso/task_set.h>

#include "benchmark_common.h"

constexpr size_t kSmallSize = 1024;
constexpr size_t kMediumSize = 8192;
constexpr size_t kLargeSize = 65536;

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
void BM_mallocfree(benchmark::State& state) {
  run(
      state,
      []() { return reinterpret_cast<char*>(::malloc(kSize)); },
      [](char* buf) { ::free(buf); });
}

template <size_t kSize>
void BM_pool_allocator(benchmark::State& state) {
  dispenso::PoolAllocator allocator(kSize, kSize * 32, ::malloc, ::free);
  run(
      state,
      [&allocator]() { return allocator.alloc(); },
      [&allocator](char* buf) { allocator.dealloc(buf); });
}

template <size_t kSize>
void BM_nl_pool_allocator(benchmark::State& state) {
  dispenso::NoLockPoolAllocator allocator(kSize, kSize * 32, ::malloc, ::free);
  run(
      state,
      [&allocator]() { return allocator.alloc(); },
      [&allocator](char* buf) { allocator.dealloc(buf); });
}

template <size_t kThreads, typename Alloc, typename Free>
void runThreaded(benchmark::State& state, Alloc alloc, Free dealloc) {
  dispenso::resizeGlobalThreadPool(kThreads);
  std::vector<char*> ptrsArray[kThreads];
  for (auto& ptrs : ptrsArray) {
    ptrs.resize(state.range(0));
  }
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(dispenso::globalThreadPool());
    for (size_t i = 0; i < kThreads; ++i) {
      tasks.schedule([alloc, dealloc, &ptrs = ptrsArray[i]]() {
        for (char*& p : ptrs) {
          p = alloc();
        }
        for (char* p : ptrs) {
          dealloc(p);
        }
      });
    }
  }
}

template <size_t kSize, size_t kThreads>
void BM_mallocfree_threaded(benchmark::State& state) {
  runThreaded<kThreads>(
      state,
      []() { return reinterpret_cast<char*>(::malloc(kSize)); },
      [](char* buf) { ::free(buf); });
}

template <size_t kSize, size_t kThreads>
void BM_pool_allocator_threaded(benchmark::State& state) {
  dispenso::PoolAllocator allocator(kSize, (1 << 20), ::malloc, ::free);
  runThreaded<kThreads>(
      state,
      [&allocator]() { return allocator.alloc(); },
      [&allocator](char* buf) { allocator.dealloc(buf); });
}

BENCHMARK_TEMPLATE(BM_mallocfree, kSmallSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_pool_allocator, kSmallSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kSmallSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_mallocfree, kMediumSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_pool_allocator, kMediumSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kMediumSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE(BM_mallocfree, kLargeSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_pool_allocator, kLargeSize)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kLargeSize)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 2)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 2)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 2)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 2)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 2)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 2)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 8)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 8)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 8)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 8)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 8)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 8)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 16)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 16)->Range(1 << 13, 1 << 15);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 16)->Range(1 << 13, 1 << 15);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 16)->Range(1 << 13, 1 << 15);

BENCHMARK_MAIN();
