/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/parallel_for.h>
#include <dispenso/thread_pool.h>
#include <dispenso/timing.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <random>
#include <unordered_map>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

namespace {

using namespace std::chrono_literals;

uint32_t kSeed(8);
constexpr int kSize = 50000;
constexpr auto kSleep = 30ms;
} // namespace

// Adapted from Google gtest examples
// Returns true iff n is a prime number.
bool isPrime(int n) {
  // Trivial case 1: small numbers
  if (n <= 1)
    return false;

  // Trivial case 2: even numbers
  if (n % 2 == 0)
    return n == 2;

  // Now, we have that n is odd and n >= 3.

  // Try to divide n by every odd number i, starting from 3
  for (int i = 3;; i += 2) {
    // We only have to try i up to the squre root of n
    if (i > n / i)
      break;

    // Now, we have i <= n/i < n.
    // If n is divisible by i, n is not prime.
    if (n % i == 0)
      return false;
  }

  // n has no integer factor in the range (1, n), and thus is prime.
  return true;
}

const std::vector<int>& getInputs(int numElements) {
  static std::unordered_map<int, std::vector<int>> vecs;
  auto it = vecs.find(numElements);
  if (it != vecs.end()) {
    return it->second;
  }

  std::mt19937_64 gen64(kSeed);
  std::uniform_int_distribution<> distribution(100000, 1000000);
  std::vector<int> values;
  values.reserve(numElements);
  for (int i = 0; i < numElements; ++i) {
    values.push_back(distribution(gen64));
  }
  auto res = vecs.emplace(numElements, std::move(values));
  assert(res.second);
  return res.first->second;
}

double getMean(const std::vector<double>& data) {
  double sum = 0.0;
  for (auto d : data) {
    sum += d;
  }
  return sum / data.size();
}

double getStddev(double mean, const std::vector<double>& data) {
  double sumsq = 0.0;
  for (auto d : data) {
    auto dev = mean - d;
    sumsq += dev * dev;
  }
  return std::sqrt(sumsq / data.size());
}

void doStats(const std::vector<double>& times, benchmark::State& state) {
  double mean = getMean(times);
  state.counters["mean"] = mean;
  state.counters["stddev"] = getStddev(mean, times);
}

void BM_serial(benchmark::State& state) {
  std::vector<int> output(kSize, 0);
  auto& input = getInputs(kSize);

  std::vector<double> times;
  times.reserve(1000);

  for (auto UNUSED_VAR : state) {
    std::this_thread::sleep_for(kSleep);
    times.push_back(dispenso::getTime());
    for (size_t i = 0; i < kSize; ++i) {
      output[i] = isPrime(input[i]);
    }
    times.back() = dispenso::getTime() - times.back();
  }

  doStats(times, state);
}

void BM_dispenso(benchmark::State& state) {
  const int numThreads = state.range(0) - 1;

  std::vector<int> output(kSize, 0);
  dispenso::resizeGlobalThreadPool(numThreads);

  std::vector<double> times;
  times.reserve(1000);

  auto& input = getInputs(kSize);
  for (auto UNUSED_VAR : state) {
    std::this_thread::sleep_for(kSleep);
    times.push_back(dispenso::getTime());
    dispenso::parallel_for(
        dispenso::makeChunkedRange(0, kSize), [&input, &output](size_t i, size_t e) {
          for (; i != e; ++i) {
            output[i] = isPrime(input[i]);
          }
        });
    times.back() = dispenso::getTime() - times.back();
  }

  doStats(times, state);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int numThreads = state.range(0);

  std::vector<int> output(kSize, 0);
  omp_set_numThreads(numThreads);

  std::vector<double> times;
  times.reserve(1000);

  auto& input = getInputs(kSize);
  for (auto UNUSED_VAR : state) {
    std::this_thread::sleep_for(kSleep);
    times.push_back(dispenso::getTime());
#pragma omp parallel for
    for (int i = 0; i < kSize; ++i) {
      output[i] = isPrime(input[i]);
    }
    times.back() = dispenso::getTime() - times.back();
  }
  doStats(times, state);
}
#endif /*defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int numThreads = state.range(0);

  std::vector<int> output(kSize, 0);

  tbb::task_scheduler_init initsched(numThreads);

  std::vector<double> times;
  times.reserve(1000);

  auto& input = getInputs(kSize);
  for (auto UNUSED_VAR : state) {
    std::this_thread::sleep_for(kSleep);
    times.push_back(dispenso::getTime());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, kSize),
        [&input, &output](const tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            output[i] = isPrime(input[i]);
          }
        });
    times.back() = dispenso::getTime() - times.back();
  }
  doStats(times, state);
}
#endif // !BENCHMARK_WITHOUT_TBB

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i : pow2HalfStepThreads()) {
    b->Arg(i);
  }
}

BENCHMARK(BM_serial)->UseRealTime();

#if defined(_OPENMP)
BENCHMARK(BM_omp)->Apply(CustomArguments)->UseRealTime();
#endif // OPENMP
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

BENCHMARK(BM_dispenso)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
