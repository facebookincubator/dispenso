/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <future>
#include <unordered_map>

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

static uint32_t kSeed(8);
static constexpr int kSmallSize = 1000;
static constexpr int kMediumSize = 1000000;
static constexpr int kLargeSize = 100000000;

const std::vector<int>& getInputs(int num_elements) {
  static std::unordered_map<int, std::vector<int>> vecs;
  auto it = vecs.find(num_elements);
  if (it != vecs.end()) {
    return it->second;
  }
  // No need to use a high-quality rng for this test.
  srand(kSeed);
  std::vector<int> values;
  values.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    values.push_back((rand() & 255) - 127);
  }
  auto res = vecs.emplace(num_elements, std::move(values));
  assert(res.second);
  return res.first->second;
}

void checkResults(const std::vector<int>& inputs, int64_t actual, int foo) {
  int64_t expected = 0;
  for (auto v : inputs) {
    expected += v * v - 3 * foo * v;
  }
  if (expected != actual) {
    std::cerr << "FAIL! " << expected << " vs " << actual << std::endl;
    abort();
  }
}

template <int num_elements>
void BM_serial(benchmark::State& state) {
  auto& input = getInputs(num_elements);
  int64_t sum = 0;
  int foo = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
    for (size_t i = 0; i < num_elements; ++i) {
      sum += input[i] * input[i] - 3 * foo * input[i];
    }
  }
  checkResults(input, sum, foo);
}

void BM_dispenso(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  dispenso::ThreadPool pool(num_threads);

  int64_t sum = 0;
  int foo = 0;

  dispenso::ParForOptions options;
  options.minItemsPerChunk = 50000;

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);

    std::vector<int64_t> sums;
    sums.reserve(num_threads + 1);
    ++foo;
    dispenso::parallel_for(
        tasks,
        sums,
        []() { return int64_t{0}; },
        dispenso::makeChunkedRange(0, num_elements, dispenso::ParForChunking::kAuto),
        [&input, foo](int64_t& lsumStore, size_t i, size_t end) {
          int64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += input[i] * input[i] - 3 * foo * input[i];
          }
          lsumStore += lsum;
        },
        options);
    sum = 0;
    for (auto s : sums) {
      sum += s;
    }
  }

  checkResults(input, sum, foo);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  omp_set_num_threads(num_threads);

  int64_t sum = 0;

  int foo = 0;

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < num_elements; ++i) {
      sum += input[i] * input[i] - 3 * foo * input[i];
    }
  }
  checkResults(input, sum, foo);
}
#endif /*defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  int64_t sum = 0;

  int foo = 0;

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    tbb::task_scheduler_init initsched(num_threads);
    ++foo;
    sum = tbb::parallel_reduce(
        tbb::blocked_range<const int*>(&input[0], &input[0] + num_elements),
        int64_t{0},
        [foo](const tbb::blocked_range<const int*>& r, int64_t init) -> int64_t {
          for (const int* a = r.begin(); a != r.end(); ++a)
            init += *a * *a - 3 * foo * *a;
          return init;
        },
        [](int64_t x, int64_t y) -> int64_t { return x + y; });
  }
  checkResults(input, sum, foo);
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_async(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);
  int64_t sum = 0;
  int foo = 0;

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    std::vector<int64_t> sums;
    ++foo;

    size_t chunkSize = (num_elements + num_threads - 1) / num_threads;

    std::vector<std::future<int64_t>> futures;

    for (int i = 0; i < num_elements; i += chunkSize) {
      futures.push_back(
          std::async([&input, foo, i, end = std::min<int>(num_elements, i + chunkSize)]() mutable {
            int64_t lsum = 0;
            for (; i != end; ++i) {
              lsum += input[i] * input[i] - 3 * foo * input[i];
            }
            return lsum;
          }));
    }
    sum = 0;
    for (auto& s : futures) {
      sum += s.get();
    }
  }

  checkResults(input, sum, foo);
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
