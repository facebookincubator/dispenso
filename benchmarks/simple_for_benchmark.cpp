/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <unordered_map>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
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

void checkResults(const std::vector<int>& input, const std::vector<int>& output) {
  for (size_t i = 0; i < input.size(); ++i) {
    if (output[i] != input[i] * input[i] - 3 * input[i]) {
      std::cerr << "FAIL! " << output[i] << " vs " << input[i] * input[i] - 3 * input[i]
                << std::endl;
      abort();
    }
  }
}

void BM_dispenso(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int num_elements = state.range(1);

  std::vector<int> output(num_elements, 0);
  dispenso::ThreadPool pool(num_threads);

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::parallel_for(tasks, 0, num_elements, [&input, &output](size_t i) {
      output[i] = input[i] * input[i] - 3 * input[i];
    });
  }
  checkResults(input, output);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  std::vector<int> output(num_elements, 0);
  omp_set_num_threads(num_threads);

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
#pragma omp parallel for
    for (int i = 0; i < num_elements; ++i) {
      output[i] = input[i] * input[i] - 3 * input[i];
    }
  }
  checkResults(input, output);
}
#endif /*defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_elements = state.range(1);

  std::vector<int> output(num_elements, 0);

  auto& input = getInputs(num_elements);
  for (auto UNUSED_VAR : state) {
    tbb::task_scheduler_init initsched(num_threads);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_elements),
        [&input, &output](const tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            output[i] = input[i] * input[i] - 3 * input[i];
          }
        });
  }
  checkResults(input, output);
}
#endif // !BENCHMARK_WITHOUT_TBB

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

BENCHMARK(BM_dispenso)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
