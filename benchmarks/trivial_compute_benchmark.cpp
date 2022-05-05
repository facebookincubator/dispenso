/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <future>

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

static constexpr int kSmallSize = 100;
static constexpr int kMediumSize = 1000000;
static constexpr int kLargeSize = 100000000;

uint32_t getInputs(int num_elements) {
  srand(num_elements);
  return rand() & 127;
}

inline uint64_t calculate(uint64_t input, uint64_t index, size_t foo) {
  return std::cos(std::log(
      std::sin(std::exp(std::sqrt(static_cast<double>((input ^ index) - 3 * foo * input))))));
}

void checkResults(uint32_t input, uint64_t actual, int foo, size_t num_elements) {
  if (!foo)
    return;
  if (input != getInputs(num_elements)) {
    std::cerr << "Failed to recover input!" << std::endl;
    abort();
  }
  uint64_t expected = 0;
  for (size_t i = 0; i < num_elements; ++i) {
    expected += calculate(input, i, foo);
  }
  if (expected != actual) {
    std::cerr << "FAIL! " << expected << " vs " << actual << std::endl;
    abort();
  }
}

template <int num_elements>
void BM_serial(benchmark::State& state) {
  auto input = getInputs(num_elements);
  uint64_t sum = 0;
  int foo = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
    for (size_t i = 0; i < num_elements; ++i) {
      sum += calculate(input, i, foo);
    }
  }
  checkResults(input, sum, foo, num_elements);
}

void BM_dispenso(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  dispenso::ThreadPool pool(num_threads);

  uint64_t sum = 0;
  int foo = 0;

  auto input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);

    std::vector<uint64_t> sums;
    sums.reserve(num_threads);
    ++foo;
    dispenso::parallel_for(
        tasks,
        sums,
        []() { return uint64_t{0}; },
        dispenso::makeChunkedRange(0, num_elements, dispenso::ParForChunking::kStatic),
        [input, foo](uint64_t& lsumStore, size_t i, size_t end) {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += calculate(input, i, foo);
          }
          lsumStore += lsum;
        });
    sum = 0;
    for (auto s : sums) {
      sum += s;
    }
  }

  checkResults(input, sum, foo, num_elements);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  omp_set_num_threads(num_threads);

  uint64_t sum = 0;

  int foo = 0;

  auto input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < num_elements; ++i) {
      sum += calculate(input, i, foo);
    }
  }
  checkResults(input, sum, foo, num_elements);
}
#endif /* defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  uint64_t sum = 0;

  int foo = 0;

  auto input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    tbb::task_scheduler_init initsched(num_threads);
    ++foo;
    sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, num_elements),
        uint64_t{0},
        [input, foo](const tbb::blocked_range<size_t>& r, uint64_t init) -> uint64_t {
          for (size_t a = r.begin(); a != r.end(); ++a)
            init += calculate(input, a, foo);
          return init;
        },
        [](uint64_t x, uint64_t y) -> uint64_t { return x + y; });
  }
  checkResults(input, sum, foo, num_elements);
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_async(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  uint64_t sum = 0;
  int foo = 0;

  auto input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    std::vector<uint64_t> sums;
    ++foo;

    size_t chunkSize = (num_elements + num_threads - 1) / num_threads;

    std::vector<std::future<uint64_t>> futures;

    for (int i = 0; i < num_elements; i += chunkSize) {
      futures.push_back(
          std::async([input, foo, i, end = std::min<int>(num_elements, i + chunkSize)]() mutable {
            uint64_t lsum = 0;
            for (; i != end; ++i) {
              lsum += calculate(input, i, foo);
            }
            return lsum;
          }));
    }
    sum = 0;
    for (auto& s : futures) {
      sum += s.get();
    }
  }

  checkResults(input, sum, foo, num_elements);
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int i : pow2HalfStepThreads()) {
      b->Args({i, j});
    }
  }
}

BENCHMARK_TEMPLATE(BM_serial, kSmallSize);
BENCHMARK_TEMPLATE(BM_serial, kMediumSize);
BENCHMARK_TEMPLATE(BM_serial, kLargeSize);

#if defined(_OPENMP)
BENCHMARK(BM_omp)->Apply(CustomArguments)->UseRealTime();
#endif // OPENMP
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_async)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_dispenso)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
