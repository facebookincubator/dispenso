// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <chrono>
#include <cmath>

#include <dispenso/task_set.h>

#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"

#include "thread_benchmark_common.h"

using namespace std::chrono_literals;

static constexpr int kSmallSize = 1000;
static constexpr int kMediumSize = 10000;
static constexpr int kLargeSize = 1000000;

struct alignas(64) Work {
  size_t count;

  void operator+=(size_t o) {
    count += o;
  }
};

Work g_work[1024];
std::atomic<int> g_tCounter {0};
inline int tid() {
  static DISPENSO_THREAD_LOCAL int t = -1;
  if (t < 0) {
    t = g_tCounter++;
  }
  return t;
}

void BM_dispenso(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  dispenso::ThreadPool pool(num_threads);

  for (auto _ : state) {
    dispenso::TaskSet tasks(pool);
    for (int i = 0; i < num_elements; ++i) {
      auto* work = g_work;
      tasks.schedule([i, work]() { work[tid() & 1023] += i; });
    }
  }
}

void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  tbb::task_scheduler_init initsched(num_threads);

  for (auto _ : state) {
    tbb::task_group g;
    for (int i = 0; i < num_elements; ++i) {
      auto* work = g_work;
      g.run([i, work]() { work[tid() & 1023] += i; });
    }
    g.wait();
  }
}

void BM_dispenso2(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  for (auto _ : state) {
    dispenso::ThreadPool pool(num_threads);
    for (int i = 0; i < num_elements; ++i) {
      pool.schedule([&pool, num_elements]() {
        int num = std::sqrt(num_elements);
        dispenso::TaskSet tasks(pool);
        for (int j = 0; j < num; ++j) {
          auto* work = g_work;
          tasks.schedule([j, work]() { work[tid() & 1023] += j; });
        }
      });
    }
  }
}

void BM_tbb2(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  for (auto _ : state) {
    tbb::task_scheduler_init initsched(num_threads);
    tbb::task_group g;
    for (int i = 0; i < num_elements; ++i) {
      g.run([num_elements]() {
        int num = std::sqrt(num_elements);
        tbb::task_group g2;
        for (int j = 0; j < num; ++j) {
          auto* work = g_work;
          g2.run([j, work]() { work[tid() & 1023] += j; });
        }
        g2.wait();
      });
    }
    g.wait();
  }
}

void BM_dispenso_mostly_idle(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  struct Recurse {
    void operator()() {
      work[tid() & 1023] += i;
      if (i < num_elements) {
        ++i;
        pool->schedule(*this);
      }
    }

    int i;
    Work* work;
    dispenso::ThreadPool* pool;
    int num_elements;
  };

  startRusage();

  for (auto _ : state) {
    dispenso::ThreadPool pool(num_threads);
    Recurse rec;
    rec.i = 0;
    rec.work = g_work;
    rec.pool = &pool;
    rec.num_elements = num_elements;
    rec();
  }

  endRusage(state);
}

void BM_tbb_mostly_idle(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  struct Recurse {
    void operator()() const {
      work[tid() & 1023] += i;
      if (i < num_elements) {
        ++i;
        g->run(*this);
      }
    }

    mutable int i;
    mutable Work* work;
    mutable tbb::task_group* g;
    int num_elements;
  };

  startRusage();

  for (auto _ : state) {
    tbb::task_scheduler_init initsched(num_threads);
    tbb::task_group g;
    Recurse rec;
    rec.i = 0;
    rec.work = g_work;
    rec.g = &g;
    rec.num_elements = num_elements;
    rec();
    g.wait();
  }
  endRusage(state);
}

void BM_dispenso_very_idle(benchmark::State& state) {
  const int num_threads = state.range(0);
  startRusage();

  for (auto _ : state) {
    dispenso::ThreadPool pool(num_threads);
    pool.schedule([]() {});
    std::this_thread::sleep_for(100ms);
    pool.schedule([]() {});
  }

  endRusage(state);
}

void BM_tbb_very_idle(benchmark::State& state) {
  const int num_threads = state.range(0);

  startRusage();

  for (auto _ : state) {
    tbb::task_scheduler_init initsched(num_threads);
    tbb::task_group g;
    g.run([]() {});
    std::this_thread::sleep_for(100ms);
    g.run([]() {});
    g.wait();
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

BENCHMARK(BM_tbb)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_dispenso)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_tbb2)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_dispenso2)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_tbb_mostly_idle)->Apply(CustomArguments)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_dispenso_mostly_idle)
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK(BM_tbb_very_idle)
    ->Apply(CustomArgumentsVeryIdle)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK(BM_dispenso_very_idle)
    ->Apply(CustomArgumentsVeryIdle)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_MAIN();
