/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmarks for thread pool behavior under low/idle workloads.
 * Tests CPU usage and responsiveness when threads are mostly waiting.
 */

#include <chrono>

#include <dispenso/task_set.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include <tbb/task_group.h>
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

using namespace std::chrono_literals;

static constexpr int kSmallSize = 1000;
static constexpr int kMediumSize = 10000;
static constexpr int kLargeSize = 1000000;

struct alignas(64) Work {
  size_t count = 0;

  void operator+=(size_t o) {
    count += o;
  }
};

Work g_work[1025];
std::atomic<int> g_tCounter{0};
inline int testTid() {
  static DISPENSO_THREAD_LOCAL int t = -1;
  if (t < 0) {
    t = g_tCounter.fetch_add(1, std::memory_order_acq_rel);
  }
  return t;
}

inline Work& work() {
  static DISPENSO_THREAD_LOCAL Work* w = nullptr;

  if (!w) {
    if (testTid() == 0) {
      w = g_work + 1024;
    } else {
      w = g_work + (testTid() & 1023);
    }
  }
  return *w;
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_mostly_idle(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  struct Recurse {
    void operator()() const {
      work() += i;
      if (i < num_elements) {
        ++i;
        g->run(*this);
      }
    }

    mutable int i;
    mutable tbb::task_group* g;
    int num_elements;
  };

  startRusage();

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    tbb::task_group g;
    Recurse rec;
    rec.i = 0;
    rec.g = &g;
    rec.num_elements = num_elements;
    rec();
    g.wait();
  }
  endRusage(state);
}

void BM_tbb_very_idle(benchmark::State& state) {
  const int num_threads = state.range(0);

  startRusage();

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    tbb::task_group g;
    g.run([]() {});
    std::this_thread::sleep_for(100ms);
    g.run([]() {});
    g.wait();
  }
  endRusage(state);
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_mostly_idle(benchmark::State& state) {
  const int num_threads = std::max<int>(1, state.range(0) - 1);
  const int num_elements = state.range(1);

  struct Recurse {
    void operator()() {
      work() += i;
      if (i < num_elements) {
        ++i;
        pool->schedule(*this);
      }
    }

    int i;
    dispenso::ThreadPool* pool;
    int num_elements;
  };

  startRusage();

  for (auto UNUSED_VAR : state) {
    dispenso::ThreadPool pool(num_threads);
    Recurse rec;
    rec.i = 0;
    rec.pool = &pool;
    rec.num_elements = num_elements;
    rec();
  }

  endRusage(state);
}

void BM_dispenso_very_idle(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  startRusage();

  for (auto UNUSED_VAR : state) {
    dispenso::ThreadPool pool(num_threads);
    pool.schedule([]() {});
    std::this_thread::sleep_for(100ms);
    pool.schedule([]() {});
  }

  endRusage(state);
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int s : pow2HalfStepThreads()) {
      b->Args({s, j});
    }
  }
}

static void CustomArgumentsVeryIdle(benchmark::internal::Benchmark* b) {
  for (int s : pow2HalfStepThreads()) {
    b->Args({s});
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_mostly_idle)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_tbb_very_idle)
    ->Apply(CustomArgumentsVeryIdle)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

BENCHMARK(BM_dispenso_mostly_idle)
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK(BM_dispenso_very_idle)
    ->Apply(CustomArgumentsVeryIdle)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_MAIN();
