/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mandelbrot escape-time benchmark: deliberately uneven per-pixel workload to
// highlight dynamic load balancing. Per-pixel iteration counts vary from 1
// (far outside the set) to maxIters (inside or on the boundary), and the
// expensive pixels cluster geometrically. Static partitioning will give some
// threads many cheap pixels and others many expensive pixels — perfect
// asymmetry. Work-stealing schedulers (dispenso kAuto, TBB auto_partitioner,
// OpenMP guided/dynamic) should rebalance and win.

#include <cmath>
#include <cstdint>

#include <dispenso/parallel_for.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb_compat.h"
#endif

#include "thread_benchmark_common.h"

struct alignas(dispenso::kCacheLineSize) AlignedSum {
  uint64_t value{0};
};

// Image dimensions.
static constexpr int kMediumDim = 1024;
static constexpr int kLargeDim = 4096;

// Iteration cap. High enough that per-pixel work dominates scheduling cost
// for interior/boundary pixels.
static constexpr int kMaxIters = 4000;

// Viewport variants. Each viewport picks a region of the complex plane that
// maps to {x in [0,width), y in [0,height)} -> {re in [reMin,reMax],
// im in [imMin,imMax]}. The asymmetry is deliberate: Seahorse Valley
// crops above-center so kStatic's top/bottom bands have very different cost.
struct Viewport {
  double reMin, reMax, imMin, imMax;
};

enum class Region {
  kFullSet, // Whole Mandelbrot set, moderate unevenness
  kBoundary, // Seahorse Valley zoom, extreme unevenness
  kInterior, // Deep interior, near-uniform high cost (control)
};

static Viewport viewportFor(Region r) {
  switch (r) {
    case Region::kFullSet:
      return {-2.5, 1.0, -1.15, 1.15};
    case Region::kBoundary:
      // Seahorse Valley, asymmetric crop above-center
      return {-0.78, -0.72, 0.08, 0.13};
    case Region::kInterior:
      // Deep in the main cardioid; almost every pixel hits maxIters
      return {-0.12, -0.08, 0.65, 0.69};
  }
  return {-2.5, 1.0, -1.15, 1.15};
}

// Escape-time iteration. Returns iter count where |z| > 2 escaped, or
// maxIters if it never escaped within the budget.
static inline uint32_t mandelbrotIters(double cr, double ci, int maxIters) {
  double zr = 0.0, zi = 0.0;
  double zr2 = 0.0, zi2 = 0.0;
  int n = 0;
  while (n < maxIters && (zr2 + zi2) <= 4.0) {
    zi = 2.0 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    ++n;
  }
  return static_cast<uint32_t>(n);
}

// Compute one pixel's iteration count given pixel coordinates.
static inline uint32_t pixelIters(int idx, int width, const Viewport& vp) {
  int x = idx % width;
  int y = idx / width;
  // Map pixel coordinates to complex plane.
  double dx = (vp.reMax - vp.reMin) / static_cast<double>(width);
  double dy = (vp.imMax - vp.imMin) / static_cast<double>(width); // square pixels
  double cr = vp.reMin + (x + 0.5) * dx;
  double ci = vp.imMin + (y + 0.5) * dy;
  return mandelbrotIters(cr, ci, kMaxIters);
}

template <Region region, int dim>
void BM_serial(benchmark::State& state) {
  const Viewport vp = viewportFor(region);
  const int numPixels = dim * dim;
  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    for (int i = 0; i < numPixels; ++i) {
      sum += pixelIters(i, dim, vp);
    }
  }
  benchmark::DoNotOptimize(sum);
}

template <Region region>
void BM_dispenso_auto(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  dispenso::resizeGlobalThreadPool(static_cast<size_t>(num_threads));
  auto& pool = dispenso::globalThreadPool();
  dispenso::ParForOptions options;
  options.defaultChunking = dispenso::ParForChunking::kAdaptive;

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    std::vector<AlignedSum> sums;
    sums.reserve(num_threads + 1);
    dispenso::parallel_for(
        tasks,
        sums,
        []() { return AlignedSum{}; },
        0,
        numPixels,
        [dim, vp](AlignedSum& lsumStore, int i, int end) {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += pixelIters(i, dim, vp);
          }
          lsumStore.value += lsum;
        },
        options);
    sum = 0;
    for (auto& s : sums) {
      sum += s.value;
    }
  }
  benchmark::DoNotOptimize(sum);
}

#if !defined(BENCHMARK_WITHOUT_TBB)
template <Region region>
void BM_tbb(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    sum = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, numPixels),
        uint64_t{0},
        [dim, vp](const tbb::blocked_range<int>& r, uint64_t init) -> uint64_t {
          for (int i = r.begin(); i != r.end(); ++i) {
            init += pixelIters(i, dim, vp);
          }
          return init;
        },
        [](uint64_t x, uint64_t y) { return x + y; });
  }
  benchmark::DoNotOptimize(sum);
}

#endif // !BENCHMARK_WITHOUT_TBB

#if defined(_OPENMP)
template <Region region>
void BM_omp_guided(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  omp_set_num_threads(num_threads);

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
#pragma omp parallel for schedule(guided) reduction(+ : sum)
    for (int i = 0; i < numPixels; ++i) {
      sum += pixelIters(i, dim, vp);
    }
  }
  benchmark::DoNotOptimize(sum);
}
#endif // _OPENMP

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int dim : {kMediumDim, kLargeDim}) {
    for (int i : benchmarkThreadCounts()) {
      b->Args({i, dim});
    }
  }
}

// Serial baselines: FullSet (realistic mix) and Boundary (work-stealing stress test).
// kInterior omitted — uniformly cheap pixels, similar to trivial_compute_benchmark.
// kSmallDim (256) omitted — too few pixels to meaningfully parallelize at high thread counts.
BENCHMARK_TEMPLATE(BM_serial, Region::kFullSet, kMediumDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kBoundary, kMediumDim);

#if defined(_OPENMP)
BENCHMARK_TEMPLATE(BM_omp_guided, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_guided, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK_TEMPLATE(BM_tbb, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
#endif

BENCHMARK_TEMPLATE(BM_dispenso_auto, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
