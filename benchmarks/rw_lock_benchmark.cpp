// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/rw_lock.h>

#include <map>
#include <shared_mutex>

#include <dispenso/task_set.h>

#include "thread_benchmark_common.h"

constexpr size_t kNumValues = 1 << 20;

// Precondition: Start < writePeriod.  Note that this is enforced in BM_serial and BM_parallel
template <typename MtxType>
int64_t iterate(MtxType& mtx, std::vector<int64_t>& values, int start, int writePeriod) {
  int64_t total = 0;
  int w = start;
  for (auto& p : values) {
    if (w++ == writePeriod) {
      std::lock_guard<MtxType> lk(mtx);
      ++p;
      w = 0;
    } else {
      std::shared_lock<MtxType> lk(mtx);
      total += p;
    }
  }
  return total;
}

struct NopMutex {
  void lock() {}
  void unlock() {}
  void lock_shared() {}
  void unlock_shared() {}
};

template <typename MutexT>
void BM_serial(benchmark::State& state) {
  int writePeriod = state.range(0);
  std::vector<int64_t> values(kNumValues);
  int64_t total = 0;
  MutexT mtx;
  int start = 0;
  for (auto UNUSED_VAR : state) {
    total += iterate(mtx, values, start++, writePeriod);
    if (start == writePeriod) {
      start = 0;
    }
  }

  benchmark::DoNotOptimize(total);
}

static void CustomArgumentsSerial(benchmark::internal::Benchmark* b) {
  for (int j : {2, 8, 32, 128, 512}) {
    b->Args({j});
  }
}

template <typename MutexT>
void BM_parallel(benchmark::State& state) {
  int concurrency = state.range(0);
  int writePeriod = state.range(1);
  std::vector<int64_t> values(kNumValues);
  std::atomic<int64_t> total(0);
  MutexT mtx;
  int start = 0;

  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  for (auto UNUSED_VAR : state) {
    for (int c = 0; c < concurrency; ++c) {
      tasks.schedule([&total, start, &mtx, &values, writePeriod]() {
        total.fetch_add(iterate(mtx, values, start, writePeriod), std::memory_order_acq_rel);
      });
      if (++start == writePeriod) {
        start = 0;
      }
    }
    tasks.wait();
  }

  benchmark::DoNotOptimize(total.load(std::memory_order_acquire));
}

static void CustomArgumentsParallel(benchmark::internal::Benchmark* b) {
  for (int j : {2, 8, 32, 128, 512}) {
    for (int s : {1, 2, 4, 8, 16, 32}) {
      if (s > static_cast<int>(std::thread::hardware_concurrency())) {
        break;
      }
      b->Args({s, j});
    }
  }
}

BENCHMARK_TEMPLATE(BM_serial, NopMutex)->Apply(CustomArgumentsSerial)->UseRealTime();

BENCHMARK_TEMPLATE(BM_serial, std::shared_mutex)->Apply(CustomArgumentsSerial)->UseRealTime();

BENCHMARK_TEMPLATE(BM_serial, dispenso::RWLock)->Apply(CustomArgumentsSerial)->UseRealTime();

BENCHMARK_TEMPLATE(BM_parallel, std::shared_mutex)->Apply(CustomArgumentsParallel)->UseRealTime();

BENCHMARK_TEMPLATE(BM_parallel, dispenso::RWLock)->Apply(CustomArgumentsParallel)->UseRealTime();
