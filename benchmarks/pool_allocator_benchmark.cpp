/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/pool_allocator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>

#include <dispenso/task_set.h>

#include "benchmark_common.h"

// Portable physical-memory query. These benchmarks also build standalone via
// CMake for open-source users, so we cannot rely on folly/arvr portability
// helpers here and query the OS directly.
#if defined(_WIN32)
// windows.h defines min/max as function-like macros, which would mangle the
// std::min below; guard as dispenso/cpu_set.cpp does.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

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

template <typename PoolAlloc>
void runArena(benchmark::State& state, PoolAlloc& allocator) {
  std::vector<char*> ptrs(state.range(0));
  for (auto UNUSED_VAR : state) {
    for (char*& p : ptrs) {
      p = allocator.alloc();
    }
    allocator.clear();
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

template <size_t kSize>
void BM_nl_pool_allocator_arena(benchmark::State& state) {
  dispenso::NoLockPoolAllocator allocator(kSize, kSize * 32, ::malloc, ::free);
  runArena(state, allocator);
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

namespace {

// Total physical RAM in bytes, or 0 if it cannot be determined. Byte counts use
// uint64_t rather than size_t: the thresholds below exceed the range of a
// 32-bit size_t, which would silently disable the cap on 32-bit builds.
uint64_t totalSystemMemoryBytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? static_cast<uint64_t>(status.ullTotalPhys) : 0;
#elif defined(__APPLE__)
  int64_t bytes = 0;
  size_t len = sizeof(bytes);
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  return sysctl(mib, 2, &bytes, &len, nullptr, 0) == 0 ? static_cast<uint64_t>(bytes) : 0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long pageSize = sysconf(_SC_PAGE_SIZE);
  return (pages > 0 && pageSize > 0)
      ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize)
      : 0;
#endif
}

// Peak live memory for these benchmarks is count * size * threads. Machines
// with ample RAM run the full sweep; memory-constrained devices (phones,
// embedded) cap the peak working set so the largest configurations don't
// exhaust the allocator and abort mid-run. Returns 0 for "no cap".
constexpr uint64_t kGiB = uint64_t(1) << 30;
constexpr uint64_t kUncappedMemoryThreshold = 24 * kGiB; // full sweep at/above this
constexpr uint64_t kConstrainedWorkingSetBudget = 1 * kGiB; // cap below the threshold

uint64_t allocWorkingSetBudget() {
  static const uint64_t budget = [] {
    const uint64_t total = totalSystemMemoryBytes();
    return (total == 0 || total >= kUncappedMemoryThreshold) ? uint64_t(0)
                                                             : kConstrainedWorkingSetBudget;
  }();
  return budget;
}

// Drop-in for ->Range(1 << 13, 1 << 15) that omits configurations whose peak
// working set (count * kSize * kThreads) would exceed the memory budget.
template <size_t kSize, size_t kThreads>
void allocRange(benchmark::internal::Benchmark* b) {
  constexpr int64_t kLo = 1 << 13;
  constexpr int64_t kHi = 1 << 15;
  constexpr uint64_t kPerItem = uint64_t(kSize) * kThreads;
  const uint64_t budget = allocWorkingSetBudget();
  bool added = false;
  for (int64_t count = kLo;; count *= 8) {
    const int64_t arg = std::min(count, kHi);
    if (budget == 0 || static_cast<uint64_t>(arg) * kPerItem <= budget) {
      b->Arg(arg);
      added = true;
    }
    if (count >= kHi) {
      break;
    }
  }
  // A benchmark registered with zero args crashes on state.range(0); when even
  // the smallest standard count exceeds the budget, fall back to the largest
  // count that fits so the case still reports a (smaller) data point.
  if (!added) {
    const int64_t fit = static_cast<int64_t>(budget / kPerItem);
    b->Arg(fit > 0 ? fit : 1);
  }
}

} // namespace

BENCHMARK_TEMPLATE(BM_mallocfree, kSmallSize)->Apply(allocRange<kSmallSize, 1>);
BENCHMARK_TEMPLATE(BM_pool_allocator, kSmallSize)->Apply(allocRange<kSmallSize, 1>);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kSmallSize)->Apply(allocRange<kSmallSize, 1>);

BENCHMARK_TEMPLATE(BM_mallocfree, kMediumSize)->Apply(allocRange<kMediumSize, 1>);
BENCHMARK_TEMPLATE(BM_pool_allocator, kMediumSize)->Apply(allocRange<kMediumSize, 1>);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kMediumSize)->Apply(allocRange<kMediumSize, 1>);

BENCHMARK_TEMPLATE(BM_mallocfree, kLargeSize)->Apply(allocRange<kLargeSize, 1>);
BENCHMARK_TEMPLATE(BM_pool_allocator, kLargeSize)->Apply(allocRange<kLargeSize, 1>);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator, kLargeSize)->Apply(allocRange<kLargeSize, 1>);

BENCHMARK_TEMPLATE(BM_nl_pool_allocator_arena, kSmallSize)->Apply(allocRange<kSmallSize, 1>);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator_arena, kMediumSize)->Apply(allocRange<kMediumSize, 1>);
BENCHMARK_TEMPLATE(BM_nl_pool_allocator_arena, kLargeSize)->Apply(allocRange<kLargeSize, 1>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 2)->Apply(allocRange<kSmallSize, 2>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 2)->Apply(allocRange<kSmallSize, 2>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 2)->Apply(allocRange<kMediumSize, 2>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 2)->Apply(allocRange<kMediumSize, 2>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 2)->Apply(allocRange<kLargeSize, 2>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 2)->Apply(allocRange<kLargeSize, 2>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 8)->Apply(allocRange<kSmallSize, 8>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 8)->Apply(allocRange<kSmallSize, 8>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 8)->Apply(allocRange<kMediumSize, 8>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 8)->Apply(allocRange<kMediumSize, 8>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 8)->Apply(allocRange<kLargeSize, 8>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 8)->Apply(allocRange<kLargeSize, 8>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kSmallSize, 16)->Apply(allocRange<kSmallSize, 16>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kSmallSize, 16)->Apply(allocRange<kSmallSize, 16>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kMediumSize, 16)->Apply(allocRange<kMediumSize, 16>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kMediumSize, 16)
    ->Apply(allocRange<kMediumSize, 16>);

BENCHMARK_TEMPLATE2(BM_mallocfree_threaded, kLargeSize, 16)->Apply(allocRange<kLargeSize, 16>);
BENCHMARK_TEMPLATE2(BM_pool_allocator_threaded, kLargeSize, 16)->Apply(allocRange<kLargeSize, 16>);

BENCHMARK_MAIN();
