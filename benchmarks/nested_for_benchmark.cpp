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

namespace {
constexpr int kWorkMultiplier = 4;
constexpr int kSmallSize = 10;
constexpr int kMediumSize = 500;
constexpr int kLargeSize = 3000;

int g_numThreads = 0;
} // namespace

uint32_t getInputs(int numElements) {
  srand(numElements);
  return rand() & 127;
}

inline uint64_t calculate(uint64_t input, uint64_t index, size_t foo) {
  return std::cos(std::log(
      std::sin(std::exp(std::sqrt(static_cast<double>((input ^ index) - 3 * foo * input))))));
}

uint64_t calculateInnerSerial(uint64_t input, size_t foo, int numElements) {
  uint64_t sum = 0;
  for (size_t i = 0; i < kWorkMultiplier * numElements; ++i) {
    sum += calculate(input, i, foo);
  }
  return sum;
}

void checkResults(uint32_t input, uint64_t actual, int foo, size_t numElements) {
  if (!foo)
    return;
  if (input != getInputs(numElements)) {
    std::cerr << "Failed to recover input!" << std::endl;
    abort();
  }
  uint64_t expected = 0;
  for (size_t i = 0; i < numElements; ++i) {
    expected += calculateInnerSerial(input, foo, numElements);
  }
  if (expected != actual) {
    std::cerr << "FAIL! " << expected << " vs " << actual << std::endl;
    abort();
  }
}

template <int numElements>
void BM_serial(benchmark::State& state) {
  auto input = getInputs(numElements);
  uint64_t sum = 0;
  int foo = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
    for (size_t j = 0; j < numElements; ++j) {
      sum += calculateInnerSerial(input, foo, numElements);
    }
  }
  checkResults(input, sum, foo, numElements);
}

uint64_t calculateInnerDispenso(uint64_t input, size_t foo, int numElements) {
  std::vector<uint64_t> sums;
  sums.reserve(g_numThreads);
  dispenso::parallel_for(
      sums,
      []() { return uint64_t{0}; },
      dispenso::makeChunkedRange(0, kWorkMultiplier * numElements, dispenso::ParForChunking::kAuto),
      [input, foo](uint64_t& lsumStore, size_t i, size_t end) {
        uint64_t lsum = 0;
        for (; i != end; ++i) {
          lsum += calculate(input, i, foo);
        }
        lsumStore += lsum;
      });
  uint64_t sum = 0;
  for (auto s : sums) {
    sum += s;
  }
  return sum;
}

void BM_dispenso(benchmark::State& state) {
  g_numThreads = state.range(0);
  const int numElements = state.range(1);

  dispenso::resizeGlobalThreadPool(g_numThreads);

  uint64_t sum = 0;
  int foo = 0;

  auto input = getInputs(numElements);
  for (auto UNUSED_VAR : state) {
    std::vector<uint64_t> sums;
    sums.reserve(g_numThreads);
    ++foo;
    dispenso::parallel_for(
        sums,
        []() { return uint64_t{0}; },
        dispenso::makeChunkedRange(0, numElements, dispenso::ParForChunking::kAuto),
        [numElements, input, foo](uint64_t& lsumStore, size_t j, size_t end) {
          uint64_t lsum = 0;
          for (; j != end; ++j) {
            lsum += calculateInnerDispenso(input, foo, numElements);
          }
          lsumStore += lsum;
        });
    sum = 0;
    for (auto s : sums) {
      sum += s;
    }
  }

  checkResults(input, sum, foo, numElements);
}

#if defined(_OPENMP)
uint64_t calculateInnerOmp(uint64_t input, size_t foo, int numElements) {
  uint64_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < kWorkMultiplier * numElements; ++i) {
    sum += calculate(input, i, foo);
  }
  return sum;
}

void BM_omp(benchmark::State& state) {
  g_numThreads = state.range(0);
  const int numElements = state.range(1);

  omp_set_num_threads(g_numThreads);

  uint64_t sum = 0;

  int foo = 0;

  auto input = getInputs(numElements);
  for (auto UNUSED_VAR : state) {
    sum = 0;
    ++foo;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < numElements; ++i) {
      sum += calculateInnerOmp(input, foo, numElements);
    }
  }
  checkResults(input, sum, foo, numElements);
}
#endif /*defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
uint64_t calculateInnerTbb(uint64_t input, size_t foo, int numElements) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, kWorkMultiplier * numElements),
      uint64_t{0},
      [input, foo](const tbb::blocked_range<size_t>& r, uint64_t init) -> uint64_t {
        for (size_t a = r.begin(); a != r.end(); ++a)
          init += calculate(input, a, foo);
        return init;
      },
      [](uint64_t x, uint64_t y) -> uint64_t { return x + y; });
}

void BM_tbb(benchmark::State& state) {
  g_numThreads = state.range(0);
  const int numElements = state.range(1);

  uint64_t sum = 0;

  int foo = 0;

  auto input = getInputs(numElements);
  for (auto UNUSED_VAR : state) {
    tbb::task_scheduler_init initsched(g_numThreads);
    ++foo;
    sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, numElements),
        uint64_t{0},
        [numElements, input, foo](const tbb::blocked_range<size_t>& r, uint64_t init) -> uint64_t {
          for (size_t a = r.begin(); a != r.end(); ++a)
            init += calculateInnerTbb(input, foo, numElements);
          return init;
        },
        [](uint64_t x, uint64_t y) -> uint64_t { return x + y; });
  }
  checkResults(input, sum, foo, numElements);
}
#endif // !BENCHMARK_WITHOUT_TBB

uint64_t calculateInnerAsync(uint64_t input, size_t foo, int numElements) {
  size_t chunkSize = (numElements + g_numThreads - 1) / g_numThreads;

  std::vector<std::future<uint64_t>> futures;

  for (int i = 0; i < kWorkMultiplier * numElements; i += chunkSize) {
    futures.push_back(
        std::async([input, foo, i, end = std::min<int>(numElements, i + chunkSize)]() mutable {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += calculate(input, i, foo);
          }
          return lsum;
        }));
  }
  uint64_t sum = 0;
  for (auto& s : futures) {
    sum += s.get();
  }
  return sum;
}

void BM_async(benchmark::State& state) {
  g_numThreads = state.range(0);
  const int numElements = state.range(1);
  uint64_t sum = 0;
  int foo = 0;

  auto input = getInputs(numElements);
  for (auto UNUSED_VAR : state) {
    std::vector<uint64_t> sums;
    ++foo;

    size_t chunkSize = (numElements + g_numThreads - 1) / g_numThreads;

    std::vector<std::future<uint64_t>> futures;

    for (int i = 0; i < numElements; i += chunkSize) {
      futures.push_back(std::async(
          [numElements, input, foo, i, end = std::min<int>(numElements, i + chunkSize)]() mutable {
            uint64_t lsum = 0;
            for (; i != end; ++i) {
              lsum += calculateInnerAsync(input, foo, numElements);
            }
            return lsum;
          }));
    }
    sum = 0;
    for (auto& s : futures) {
      sum += s.get();
    }
  }

  checkResults(input, sum, foo, numElements);
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
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

BENCHMARK(BM_async)->Apply(CustomArguments)->UseRealTime();

BENCHMARK(BM_dispenso)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
