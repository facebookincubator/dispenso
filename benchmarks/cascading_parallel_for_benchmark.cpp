/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark demonstrating dispenso's cascading parallel_for advantage.
 *
 * When multiple independent parallel_for loops need to run, dispenso can
 * overlap them on a shared TaskSet using ParForOptions{.wait = false}.
 * TBB and OpenMP each impose an implicit barrier per parallel_for call,
 * forcing sequential execution of independent loops.
 */

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <array>
#include <unordered_map>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_group.h"
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

static constexpr int32_t kSmallSize = 1000;
static constexpr int32_t kMediumSize = 100000;
static constexpr int32_t kLargeSize = 10000000;

static constexpr int32_t kNumLoops = 8;

// Minimum work per chunk to amortize scheduling overhead for cheap lambdas.
// With trivial compute (~4 integer ops ≈ 2ns/element), 512 elements ≈ 1µs
// of work, comfortably covering task dispatch cost on Windows.
static constexpr uint32_t kMinItemsPerChunk = 512;

static uint32_t kSeed(42);

inline int32_t compute(int32_t x) {
  return x * x - 3 * x + 7;
}

inline int32_t fuse(const std::array<int32_t, kNumLoops>& values) {
  int32_t result = 0;
  for (int32_t k = 0; k < kNumLoops; ++k) {
    result += values[static_cast<size_t>(k)];
  }
  return result;
}

struct BenchArrays {
  std::array<std::vector<int32_t>, kNumLoops> inputs;
  std::array<std::vector<int32_t>, kNumLoops> outputs;
  std::vector<int32_t> result;
};

BenchArrays& getArrays(int32_t numElements) {
  static std::unordered_map<int32_t, BenchArrays> arrays;
  auto it = arrays.find(numElements);
  if (it != arrays.end()) {
    return it->second;
  }
  srand(kSeed);
  BenchArrays ba;
  for (int32_t k = 0; k < kNumLoops; ++k) {
    ba.inputs[static_cast<size_t>(k)].reserve(static_cast<size_t>(numElements));
    for (int32_t i = 0; i < numElements; ++i) {
      ba.inputs[static_cast<size_t>(k)].push_back((rand() & 255) - 127);
    }
    ba.outputs[static_cast<size_t>(k)].resize(static_cast<size_t>(numElements), 0);
  }
  ba.result.resize(static_cast<size_t>(numElements), 0);
  auto res = arrays.emplace(numElements, std::move(ba));
  assert(res.second);
  return res.first->second;
}

void checkResults(BenchArrays& ba, int32_t numElements) {
  for (int32_t i = 0; i < numElements; ++i) {
    auto idx = static_cast<size_t>(i);
    std::array<int32_t, kNumLoops> expected;
    for (int32_t k = 0; k < kNumLoops; ++k) {
      expected[static_cast<size_t>(k)] = compute(ba.inputs[static_cast<size_t>(k)][idx]);
    }
    int32_t expectedFused = fuse(expected);
    if (ba.result[idx] != expectedFused) {
      std::cerr << "FAIL at index " << i << ": got " << ba.result[idx] << " expected "
                << expectedFused << std::endl;
      abort();
    }
  }
}

void BM_serial(benchmark::State& state) {
  const int32_t numElements = state.range(0);
  auto& ba = getArrays(numElements);

  for (auto UNUSED_VAR : state) {
    for (int32_t k = 0; k < kNumLoops; ++k) {
      auto kk = static_cast<size_t>(k);
      for (int32_t i = 0; i < numElements; ++i) {
        ba.outputs[kk][static_cast<size_t>(i)] = compute(ba.inputs[kk][static_cast<size_t>(i)]);
      }
    }
    for (int32_t i = 0; i < numElements; ++i) {
      auto idx = static_cast<size_t>(i);
      std::array<int32_t, kNumLoops> vals;
      for (int32_t k = 0; k < kNumLoops; ++k) {
        vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
      }
      ba.result[idx] = fuse(vals);
    }
  }
  checkResults(ba, numElements);
}

void BM_dispenso_blocking(benchmark::State& state) {
  const int32_t numThreads = state.range(0) - 1;
  const int32_t numElements = state.range(1);
  auto& ba = getArrays(numElements);

  dispenso::ThreadPool pool(numThreads);
  dispenso::ParForOptions opts;
  opts.minItemsPerChunk = kMinItemsPerChunk;

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    for (int32_t k = 0; k < kNumLoops; ++k) {
      auto kk = static_cast<size_t>(k);
      dispenso::parallel_for(
          tasks,
          0,
          numElements,
          [&inputs = ba.inputs[kk], &outputs = ba.outputs[kk]](int32_t i) {
            outputs[static_cast<size_t>(i)] = compute(inputs[static_cast<size_t>(i)]);
          },
          opts);
    }
    dispenso::parallel_for(
        tasks,
        0,
        numElements,
        [&ba](int32_t i) {
          auto idx = static_cast<size_t>(i);
          std::array<int32_t, kNumLoops> vals;
          for (int32_t k = 0; k < kNumLoops; ++k) {
            vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
          }
          ba.result[idx] = fuse(vals);
        },
        opts);
  }
  checkResults(ba, numElements);
}

void BM_dispenso_cascaded(benchmark::State& state) {
  const int32_t numThreads = state.range(0) - 1;
  const int32_t numElements = state.range(1);
  auto& ba = getArrays(numElements);

  dispenso::ThreadPool pool(numThreads);
  dispenso::ParForOptions noWait;
  noWait.wait = false;
  noWait.minItemsPerChunk = kMinItemsPerChunk;
  dispenso::ParForOptions opts;
  opts.minItemsPerChunk = kMinItemsPerChunk;

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    // First N-1 loops: non-blocking, returns immediately
    for (int32_t k = 0; k < kNumLoops - 1; ++k) {
      auto kk = static_cast<size_t>(k);
      dispenso::parallel_for(
          tasks,
          0,
          numElements,
          [&inputs = ba.inputs[kk], &outputs = ba.outputs[kk]](int32_t i) {
            outputs[static_cast<size_t>(i)] = compute(inputs[static_cast<size_t>(i)]);
          },
          noWait);
    }
    // Last independent loop: blocking — calling thread participates,
    // implicitly waits for all prior non-blocking loops too
    {
      constexpr auto kk = static_cast<size_t>(kNumLoops - 1);
      dispenso::parallel_for(
          tasks,
          0,
          numElements,
          [&inputs = ba.inputs[kk], &outputs = ba.outputs[kk]](int32_t i) {
            outputs[static_cast<size_t>(i)] = compute(inputs[static_cast<size_t>(i)]);
          },
          opts);
    }
    // Fusion: blocking (depends on all outputs being complete)
    dispenso::parallel_for(
        tasks,
        0,
        numElements,
        [&ba](int32_t i) {
          auto idx = static_cast<size_t>(i);
          std::array<int32_t, kNumLoops> vals;
          for (int32_t k = 0; k < kNumLoops; ++k) {
            vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
          }
          ba.result[idx] = fuse(vals);
        },
        opts);
  }
  checkResults(ba, numElements);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t numElements = state.range(1);
  auto& ba = getArrays(numElements);

  omp_set_num_threads(numThreads);

  for (auto UNUSED_VAR : state) {
    for (int32_t k = 0; k < kNumLoops; ++k) {
      auto kk = static_cast<size_t>(k);
#pragma omp parallel for
      for (int32_t i = 0; i < numElements; ++i) {
        ba.outputs[kk][static_cast<size_t>(i)] = compute(ba.inputs[kk][static_cast<size_t>(i)]);
      }
    }
#pragma omp parallel for
    for (int32_t i = 0; i < numElements; ++i) {
      auto idx = static_cast<size_t>(i);
      std::array<int32_t, kNumLoops> vals;
      for (int32_t k = 0; k < kNumLoops; ++k) {
        vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
      }
      ba.result[idx] = fuse(vals);
    }
  }
  checkResults(ba, numElements);
}
#endif /*defined(_OPENMP)*/

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t numElements = state.range(1);
  auto& ba = getArrays(numElements);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(numThreads);

    for (int32_t k = 0; k < kNumLoops; ++k) {
      auto kk = static_cast<size_t>(k);
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, numElements),
          [&inputs = ba.inputs[kk],
           &outputs = ba.outputs[kk]](const tbb::blocked_range<int32_t>& r) {
            for (int32_t i = r.begin(); i != r.end(); ++i) {
              outputs[static_cast<size_t>(i)] = compute(inputs[static_cast<size_t>(i)]);
            }
          });
    }
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, numElements), [&ba](const tbb::blocked_range<int32_t>& r) {
          for (int32_t i = r.begin(); i != r.end(); ++i) {
            auto idx = static_cast<size_t>(i);
            std::array<int32_t, kNumLoops> vals;
            for (int32_t k = 0; k < kNumLoops; ++k) {
              vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
            }
            ba.result[idx] = fuse(vals);
          }
        });
  }
  checkResults(ba, numElements);
}

// TBB with task_group: launch all independent parallel_for loops concurrently
// via task_group, then wait — emulating dispenso's cascading behavior.
void BM_tbb_task_group(benchmark::State& state) {
  const int32_t numThreads = state.range(0);
  const int32_t numElements = state.range(1);
  auto& ba = getArrays(numElements);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(numThreads);
    tbb::task_group tg;

    for (int32_t k = 0; k < kNumLoops; ++k) {
      auto kk = static_cast<size_t>(k);
      tg.run([&inputs = ba.inputs[kk], &outputs = ba.outputs[kk], numElements]() {
        tbb::parallel_for(
            tbb::blocked_range<int32_t>(0, numElements),
            [&inputs, &outputs](const tbb::blocked_range<int32_t>& r) {
              for (int32_t i = r.begin(); i != r.end(); ++i) {
                outputs[static_cast<size_t>(i)] = compute(inputs[static_cast<size_t>(i)]);
              }
            });
      });
    }
    tg.wait();

    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, numElements), [&ba](const tbb::blocked_range<int32_t>& r) {
          for (int32_t i = r.begin(); i != r.end(); ++i) {
            auto idx = static_cast<size_t>(i);
            std::array<int32_t, kNumLoops> vals;
            for (int32_t k = 0; k < kNumLoops; ++k) {
              vals[static_cast<size_t>(k)] = ba.outputs[static_cast<size_t>(k)][idx];
            }
            ba.result[idx] = fuse(vals);
          }
        });
  }
  checkResults(ba, numElements);
}
#endif // !BENCHMARK_WITHOUT_TBB

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int i : pow2HalfStepThreads()) {
      b->Args({i, j});
    }
  }
}

BENCHMARK(BM_serial)->Args({kSmallSize})->Args({kMediumSize})->Args({kLargeSize})->UseRealTime();

#if defined(_OPENMP)
BENCHMARK(BM_omp)->Apply(CustomArguments)->UseRealTime();
#endif // OPENMP
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_tbb_task_group)->Apply(CustomArguments)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

BENCHMARK(BM_dispenso_blocking)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_dispenso_cascaded)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
