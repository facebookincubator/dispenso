/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmarks for dispenso::for_each_n / dispenso::for_each.
 * Tests scheduling overhead across different container/iterator types:
 *   - vector (random-access)
 *   - deque (random-access)
 *   - list (bidirectional)
 *   - set (bidirectional, const elements)
 */

#include <dispenso/for_each.h>

#include <deque>
#include <list>
#include <set>
#include <unordered_map>

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

void checkResults(const std::vector<int>& input, const std::vector<int>& output) {
  for (size_t i = 0; i < input.size(); ++i) {
    if (output[i] != input[i] * input[i] - 3 * input[i]) {
      std::cerr << "FAIL! " << output[i] << " vs " << input[i] * input[i] - 3 * input[i]
                << std::endl;
      abort();
    }
  }
}

template <int num_elements>
void BM_serial(benchmark::State& state) {
  std::vector<int> output(num_elements, 0);
  auto& input = getInputs(num_elements);

  for (auto UNUSED_VAR : state) {
    for (size_t i = 0; i < num_elements; ++i) {
      output[i] = input[i] * input[i] - 3 * input[i];
    }
  }
}

void BM_for_each_n(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  std::vector<int> output(num_elements, 0);
  dispenso::ThreadPool pool(num_threads);

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::for_each_n(
        tasks, input.begin(), static_cast<size_t>(num_elements), [&output, &input](const int& val) {
          size_t idx = static_cast<size_t>(&val - input.data());
          output[idx] = val * val - 3 * val;
        });
  }
  checkResults(input, output);
}

void BM_for_each_n_deque(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  auto& input = getInputs(num_elements);
  std::deque<int> deq(input.begin(), input.end());
  dispenso::ThreadPool pool(num_threads);

  std::atomic<int64_t> sum(0);
  for (auto UNUSED_VAR : state) {
    sum.store(0, std::memory_order_relaxed);
    dispenso::TaskSet tasks(pool);
    dispenso::for_each_n(
        tasks, deq.begin(), static_cast<size_t>(num_elements), [&sum](const int& val) {
          sum.fetch_add(val * val - 3 * val, std::memory_order_relaxed);
        });
  }
  benchmark::DoNotOptimize(sum.load());
}

void BM_for_each_n_list(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  auto& input = getInputs(num_elements);
  std::list<int> lst(input.begin(), input.end());
  dispenso::ThreadPool pool(num_threads);

  std::atomic<int64_t> sum(0);
  for (auto UNUSED_VAR : state) {
    sum.store(0, std::memory_order_relaxed);
    dispenso::TaskSet tasks(pool);
    dispenso::for_each_n(
        tasks, lst.begin(), static_cast<size_t>(num_elements), [&sum](const int& val) {
          sum.fetch_add(val * val - 3 * val, std::memory_order_relaxed);
        });
  }
  benchmark::DoNotOptimize(sum.load());
}

void BM_for_each_n_set(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  auto& input = getInputs(num_elements);
  std::set<int> s(input.begin(), input.end());
  // set deduplicates, so actual size may be smaller
  size_t actual_size = s.size();
  dispenso::ThreadPool pool(num_threads);

  std::atomic<int64_t> sum(0);
  for (auto UNUSED_VAR : state) {
    sum.store(0, std::memory_order_relaxed);
    dispenso::TaskSet tasks(pool);
    dispenso::for_each_n(tasks, s.begin(), actual_size, [&sum](const int& val) {
      sum.fetch_add(val * val - 3 * val, std::memory_order_relaxed);
    });
  }
  benchmark::DoNotOptimize(sum.load());
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int i : pow2HalfStepThreads()) {
      b->Args({i, j});
    }
  }
}

// Smaller argument set for containers where 100M elements is impractical
static void SmallArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize}) {
    for (int i : pow2HalfStepThreads()) {
      b->Args({i, j});
    }
  }
}

BENCHMARK_TEMPLATE(BM_serial, kSmallSize);
BENCHMARK_TEMPLATE(BM_serial, kMediumSize);
BENCHMARK_TEMPLATE(BM_serial, kLargeSize);

BENCHMARK(BM_for_each_n)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_for_each_n_deque)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_for_each_n_list)->Apply(SmallArguments)->UseRealTime();
BENCHMARK(BM_for_each_n_set)->Apply(SmallArguments)->UseRealTime();

BENCHMARK_MAIN();
