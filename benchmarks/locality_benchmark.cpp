/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark for cache/NUMA-sensitive workloads.
 *
 * Tests a repeated stencil pattern over large arrays (exceeding L3 cache)
 * where deterministic thread-to-memory mapping (kStatic) may outperform
 * dynamic chunking (kAuto) due to cache locality and NUMA effects.
 *
 * The work per element is intentionally trivial (memory-bound) so that
 * memory access patterns dominate performance, not compute.
 */

// MSVC uses __restrict, GCC/Clang use __restrict__
#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#include <cstring>
#include <numeric>
#include <vector>

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include "thread_benchmark_common.h"

// ~32MB per array (4M doubles). Two arrays = 64MB, exceeds typical per-core L2
// but may fit in shared L3 on some machines. Scale up if needed.
static constexpr size_t kSmallSize = 100000; // ~800KB per array
static constexpr size_t kMediumSize = 4000000; // ~32MB per array
static constexpr size_t kLargeSize = 32000000; // ~256MB per array

static constexpr int kPasses = 10;

// Initialize arrays with deterministic data.
static void initArrays(std::vector<double>& input, std::vector<double>& output, size_t n) {
  input.resize(n);
  output.resize(n);
  for (size_t i = 0; i < n; ++i) {
    input[i] = static_cast<double>(i % 1000) * 0.001;
  }
  std::memset(output.data(), 0, n * sizeof(double));
}

// Simple 3-point stencil: output[i] = 0.25 * (input[i-1] + 2*input[i] + input[i+1])
// Cheap compute, memory-bound. Multiple passes amplify locality effects.
inline void
stencilPass(const double* RESTRICT src, double* RESTRICT dst, size_t begin, size_t end, size_t n) {
  for (size_t i = begin; i < end; ++i) {
    size_t im = (i > 0) ? i - 1 : 0;
    size_t ip = (i < n - 1) ? i + 1 : n - 1;
    dst[i] = 0.25 * (src[im] + 2.0 * src[i] + src[ip]);
  }
}

// Verify output is non-garbage (spot-check a few values).
static void checkOutput(const double* data, size_t n) {
  if (n == 0)
    return;
  // After stencil passes, values should be finite and in a reasonable range.
  for (size_t i = 0; i < n; i += n / 10 + 1) {
    if (!std::isfinite(data[i])) {
      std::cerr << "FAIL: non-finite value at index " << i << std::endl;
      abort();
    }
  }
}

template <size_t num_elements>
void BM_serial(benchmark::State& state) {
  std::vector<double> input, output;
  initArrays(input, output, num_elements);

  for (auto UNUSED_VAR : state) {
    for (int pass = 0; pass < kPasses; ++pass) {
      stencilPass(input.data(), output.data(), 0, num_elements, num_elements);
      std::swap(input, output);
    }
  }
  checkOutput(input.data(), num_elements);
}

void BM_dispenso_static(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const size_t num_elements = state.range(1);

  std::vector<double> input, output;
  initArrays(input, output, num_elements);

  dispenso::ThreadPool pool(num_threads);

  for (auto UNUSED_VAR : state) {
    for (int pass = 0; pass < kPasses; ++pass) {
      dispenso::TaskSet tasks(pool);
      const double* src = input.data();
      double* dst = output.data();
      auto range =
          dispenso::makeChunkedRange(size_t{0}, num_elements, dispenso::ParForChunking::kStatic);
      dispenso::parallel_for(tasks, range, [src, dst, num_elements](size_t begin, size_t end) {
        stencilPass(src, dst, begin, end, num_elements);
      });
      std::swap(input, output);
    }
  }
  checkOutput(input.data(), num_elements);
}

void BM_dispenso_auto(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const size_t num_elements = state.range(1);

  std::vector<double> input, output;
  initArrays(input, output, num_elements);

  dispenso::ThreadPool pool(num_threads);

  for (auto UNUSED_VAR : state) {
    for (int pass = 0; pass < kPasses; ++pass) {
      dispenso::TaskSet tasks(pool);
      const double* src = input.data();
      double* dst = output.data();
      auto range =
          dispenso::makeChunkedRange(size_t{0}, num_elements, dispenso::ParForChunking::kAuto);
      dispenso::parallel_for(tasks, range, [src, dst, num_elements](size_t begin, size_t end) {
        stencilPass(src, dst, begin, end, num_elements);
      });
      std::swap(input, output);
    }
  }
  checkOutput(input.data(), num_elements);
}

#if defined(_OPENMP)
void BM_omp(benchmark::State& state) {
  const int num_threads = state.range(0);
  const size_t num_elements = state.range(1);

  std::vector<double> input, output;
  initArrays(input, output, num_elements);

  omp_set_num_threads(num_threads);

  for (auto UNUSED_VAR : state) {
    for (int pass = 0; pass < kPasses; ++pass) {
      const double* src = input.data();
      double* dst = output.data();
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < static_cast<int64_t>(num_elements); ++i) {
        size_t im = (i > 0) ? i - 1 : 0;
        size_t ip = (i < num_elements - 1) ? i + 1 : num_elements - 1;
        dst[i] = 0.25 * (src[im] + 2.0 * src[i] + src[ip]);
      }
      std::swap(input, output);
    }
  }
  checkOutput(input.data(), num_elements);
}
#endif /* defined(_OPENMP) */

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const size_t num_elements = state.range(1);

  std::vector<double> input, output;
  initArrays(input, output, num_elements);

  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    for (int pass = 0; pass < kPasses; ++pass) {
      const double* src = input.data();
      double* dst = output.data();
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, num_elements),
          [src, dst, num_elements](const tbb::blocked_range<size_t>& r) {
            stencilPass(src, dst, r.begin(), r.end(), num_elements);
          });
      std::swap(input, output);
    }
  }
  checkOutput(input.data(), num_elements);
}
#endif // !BENCHMARK_WITHOUT_TBB

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (size_t j : {kSmallSize, kMediumSize, kLargeSize}) {
    for (int i : {1, 4, 16, 64, 128}) {
      b->Args({i, static_cast<int64_t>(j)});
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
BENCHMARK(BM_dispenso_static)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_dispenso_auto)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
