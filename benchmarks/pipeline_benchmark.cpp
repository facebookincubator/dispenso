/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

#include <dispenso/pipeline.h>

#include "benchmark_common.h"

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/pipeline.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include <taskflow/taskflow.hpp>
#if TF_VERSION > 300000
#include <taskflow/algorithm/pipeline.hpp>
#endif // TF_VERSION

// (1) Generate images
// (2) Calculate geometric mean
// (3) Tonemap based on geometric mean

constexpr size_t kWidth = 256;
constexpr size_t kHeight = 256;
constexpr size_t kNumImages = 500;
constexpr size_t kSeed = 55;

struct Work {
  Work(size_t idx) : index(idx) {}

  Work(Work&& w)
      : index(w.index), geometricMean(w.geometricMean), inputImage(std::move(w.inputImage)) {}

  Work& operator=(Work&& w) {
    index = w.index;
    geometricMean = w.geometricMean;
    inputImage = std::move(w.inputImage);
    return *this;
  }

  size_t index;
  double geometricMean = 0;
  std::unique_ptr<uint16_t[]> inputImage;
};

std::vector<std::unique_ptr<uint8_t[]>> g_serialResults;

Work fillImage(Work work) {
  std::mt19937 rng(kSeed + work.index);

  work.inputImage = std::make_unique<uint16_t[]>(kWidth * kHeight);
  std::uniform_int_distribution<uint16_t> dist;

  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    work.inputImage[i] = dist(rng);
  }

  return work;
}

Work computeGeometricMean(Work work) {
  double sum = 0;
  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    sum += std::log(1.0e-10 + work.inputImage[i]);
  }
  work.geometricMean = std::exp(sum / (kWidth * kHeight));
  return work;
}

std::unique_ptr<uint8_t[]> tonemap(Work work) {
  auto out = std::make_unique<uint8_t[]>(kWidth * kHeight);
  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    double adjLum = work.inputImage[i] / work.geometricMean;
    out[i] = 255 * adjLum / (1.0 + adjLum);
  }
  return out;
}

void runSerial() {
  g_serialResults.resize(kNumImages);

  size_t index = 0;
  for (auto& out : g_serialResults) {
    out = tonemap(computeGeometricMean(fillImage(Work(index++))));
  }
}

void checkResults(const std::vector<std::unique_ptr<uint8_t[]>>& results) {
  if (g_serialResults.empty()) {
    runSerial();
  }

  if (g_serialResults.size() != results.size()) {
    std::cerr << "Number of results don't match" << std::endl;
    std::abort();
  }

  for (size_t i = 0; i < results.size(); ++i) {
    if (std::memcmp(
            g_serialResults[i].get(), results[i].get(), kWidth * kHeight * sizeof(uint8_t))) {
      std::cerr << "Mismatch in results" << std::endl;

      for (size_t j = 0; j < 10; ++j) {
        std::cerr << (int)g_serialResults[i][j] << " vs " << (int)results[i][j] << std::endl;
      }
      std::abort();
    }
  }
}

void BM_serial(benchmark::State& state) {
  for (auto UNUSED_VAR : state) {
    runSerial();
  }
}

void runDispenso(std::vector<std::unique_ptr<uint8_t[]>>& results) {
  results.resize(kNumImages);

  size_t counter = 0;

  dispenso::pipeline(
      [&counter]() -> dispenso::OpResult<Work> {
        if (counter < kNumImages) {
          return fillImage(Work(counter++));
        }
        return {};
      },
      computeGeometricMean,
      [&results](Work work) {
        size_t index = work.index;
        results[index] = tonemap(std::move(work));
      });
}

void BM_dispenso(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;

  (void)dispenso::globalThreadPool();

  for (auto UNUSED_VAR : state) {
    runDispenso(results);
  }

  checkResults(results);
}

void runDispensoPar(std::vector<std::unique_ptr<uint8_t[]>>& results) {
  results.resize(kNumImages);
  std::atomic<size_t> counter(0);

  dispenso::pipeline(
      dispenso::stage(
          [&counter]() -> dispenso::OpResult<Work> {
            size_t curIndex = counter.fetch_add(1, std::memory_order_acquire);
            if (curIndex < kNumImages) {
              return fillImage(Work(curIndex));
            }
            return {};
          },
          dispenso::kStageNoLimit),
      dispenso::stage(computeGeometricMean, dispenso::kStageNoLimit),
      dispenso::stage(
          [&results](Work work) {
            size_t index = work.index;
            results[index] = tonemap(std::move(work));
          },
          dispenso::kStageNoLimit));
}

void BM_dispenso_par(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;

  (void)dispenso::globalThreadPool();

  for (auto UNUSED_VAR : state) {
    runDispensoPar(results);
  }

  checkResults(results);
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void runTBB(std::vector<std::unique_ptr<uint8_t[]>>& results) {
  results.resize(kNumImages);

  size_t counter = 0;
  tbb::parallel_pipeline(
      /*max_number_of_live_token=*/std::thread::hardware_concurrency(),
      tbb::make_filter<void, Work*>(
          tbb::filter::serial,
          [&counter](tbb::flow_control& fc) -> Work* {
            if (counter < kNumImages) {
              return new Work(fillImage(Work(counter++)));
            }
            fc.stop();
            return nullptr;
          }) &
          tbb::make_filter<Work*, Work*>(
              tbb::filter::serial,
              [](Work* workIn) {
                Work& work = *workIn;
                work = computeGeometricMean(std::move(work));
                return workIn;
              }) &
          tbb::make_filter<Work*, void>(tbb::filter::serial, [&results](Work* workIn) {
            size_t index = workIn->index;
            results[index] = tonemap(std::move(*workIn));
            delete workIn;
          }));
}

void BM_tbb(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;

  for (auto UNUSED_VAR : state) {
    runTBB(results);
  }

  checkResults(results);
}

void runTBBPar(std::vector<std::unique_ptr<uint8_t[]>>& results) {
  results.resize(kNumImages);

  std::atomic<size_t> counter(0);
  tbb::parallel_pipeline(
      /*max_number_of_live_token=*/std::thread::hardware_concurrency(),
      tbb::make_filter<void, Work*>(
          tbb::filter::parallel,
          [&counter](tbb::flow_control& fc) -> Work* {
            size_t curIndex = counter.fetch_add(1, std::memory_order_acquire);
            if (curIndex < kNumImages) {
              return new Work(fillImage(Work(curIndex)));
            }
            fc.stop();
            return nullptr;
          }) &
          tbb::make_filter<Work*, Work*>(
              tbb::filter::parallel,
              [](Work* work) {
                *work = computeGeometricMean(std::move(*work));
                return work;
              }) &
          tbb::make_filter<Work*, void>(tbb::filter::parallel, [&results](Work* work) {
            size_t index = work->index;
            results[index] = tonemap(std::move(*work));
            delete work;
          }));
}

void BM_tbb_par(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;

  for (auto UNUSED_VAR : state) {
    runTBBPar(results);
  }

  checkResults(results);
}
#endif // !BENCHMARK_WITHOUT_TBB

void runTaskflow(std::vector<std::unique_ptr<uint8_t[]>>& results, tf::Executor& exec) {
  results.resize(kNumImages);
  std::vector<std::unique_ptr<Work>> work;
  // Ensure we don't resize underlying buffer causing data races
  work.reserve(kNumImages);

  tf::Taskflow taskflow;

  size_t counter2 = 0;
  size_t counter3 = 0;
  tf::Pipeline pl(
      std::thread::hardware_concurrency(),
      tf::Pipe{
          tf::PipeType::SERIAL,
          [&work](auto& pf) mutable {
            if (work.size() < kNumImages) {
              work.push_back(std::make_unique<Work>(fillImage(Work(work.size()))));
            } else {
              pf.stop();
            }
          }},
      tf::Pipe{
          tf::PipeType::SERIAL,
          [&counter2, &work](auto& pf) mutable {
            Work& w = *work[counter2++];
            w = computeGeometricMean(std::move(w));
          }},
      tf::Pipe{tf::PipeType::SERIAL, [&counter3, &work, &results](auto& pf) mutable {
                 Work& w = *work[counter3];
                 results[counter3++] = tonemap(std::move(w));
               }});
  taskflow.composed_of(pl);
  exec.run(taskflow).wait();
}

void BM_taskflow(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;
  tf::Executor executor(std::thread::hardware_concurrency());

  for (auto UNUSED_VAR : state) {
    runTaskflow(results, executor);
  }

  checkResults(results);
}

// TODO(bbudge): Debug this.  Unclear exactly why this crashes and/or hangs (TSAN)
void runTaskflowPar(std::vector<std::unique_ptr<uint8_t[]>>& results, tf::Executor& exec) {
  results.resize(kNumImages);
  std::vector<std::unique_ptr<Work>> work;
  // Ensure we don't resize underlying buffer causing data races
  work.reserve(kNumImages);

  tf::Taskflow taskflow;

  std::atomic<size_t> counter2 = 0;
  std::atomic<size_t> counter3 = 0;
  tf::Pipeline pl(
      std::thread::hardware_concurrency(),
      tf::Pipe{
          tf::PipeType::SERIAL,
          [&work](auto& pf) mutable {
            if (work.size() < kNumImages) {
              work.push_back(std::make_unique<Work>(fillImage(Work(work.size()))));
            } else {
              pf.stop();
            }
          }},
      tf::Pipe{
          tf::PipeType::PARALLEL,
          [&counter2, &work](auto& pf) mutable {
            Work& w = *work[counter2.fetch_add(1, std::memory_order_relaxed)];
            w = computeGeometricMean(std::move(w));
          }},
      tf::Pipe{tf::PipeType::PARALLEL, [&counter3, &work, &results](auto& pf) mutable {
                 size_t index = counter3.fetch_add(1, std::memory_order_relaxed);
                 Work& w = *work[index];
                 results[index] = tonemap(std::move(w));
               }});

  taskflow.composed_of(pl);
  exec.run(taskflow).wait();
}

void BM_taskflow_par(benchmark::State& state) {
  std::vector<std::unique_ptr<uint8_t[]>> results;
  tf::Executor executor(std::thread::hardware_concurrency());

  for (auto UNUSED_VAR : state) {
    runTaskflowPar(results, executor);
  }

  checkResults(results);
}

BENCHMARK(BM_serial)->UseRealTime();
BENCHMARK(BM_dispenso)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_taskflow)->UseRealTime();

BENCHMARK(BM_dispenso_par)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_par)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

// TODO(bbudge): Re-enable once this is fixed.
// BENCHMARK(BM_taskflow_par)->UseRealTime();

BENCHMARK_MAIN();
