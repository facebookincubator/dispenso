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

#include <dispenso/cpu_set.h>
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
static constexpr int kSmallDim = 256;
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
  options.defaultChunking = dispenso::ParForChunking::kAuto;

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

// Explicit fine-grained chunking: aim for ~256 chunks per thread instead of
// kAuto's 16-per-thread cap. Useful as a comparison point — shows whether
// the kAuto default chunk granularity is the bottleneck for uneven loads.
template <Region region>
void BM_dispenso_fine(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  dispenso::resizeGlobalThreadPool(static_cast<size_t>(num_threads));
  auto& pool = dispenso::globalThreadPool();
  const int chunkSize = std::max(1, numPixels / (256 * std::max(1, num_threads)));

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    std::vector<AlignedSum> sums;
    sums.reserve(num_threads + 1);
    dispenso::parallel_for(
        tasks,
        sums,
        []() { return AlignedSum{}; },
        dispenso::makeChunkedRange(0, numPixels, chunkSize),
        [dim, vp](AlignedSum& lsumStore, int i, int end) {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += pixelIters(i, dim, vp);
          }
          lsumStore.value += lsum;
        });
    sum = 0;
    for (auto& s : sums) {
      sum += s.value;
    }
  }
  benchmark::DoNotOptimize(sum);
}

template <Region region>
void BM_dispenso_static(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  // Use globalThreadPool + resize for fairness with TBB, which keeps a single
  // long-lived worker pool across benchmark invocations. Constructing a fresh
  // ThreadPool per benchmark run incurs first-touch cache penalties (cold TLS,
  // ring init, wakeState init) that TBB amortizes once.
  dispenso::resizeGlobalThreadPool(static_cast<size_t>(num_threads));
  auto& pool = dispenso::globalThreadPool();

  // Pure-compute variant: no per-chunk reduction writes, no shared state in
  // the body. Each chunk computes an iteration sum and uses DoNotOptimize on
  // it directly. This isolates the framework/cache cost from any reduction-
  // storage cost (per-thread AlignedSum allocation + writebacks).
  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::parallel_for(
        tasks,
        dispenso::makeChunkedRange(0, numPixels, dispenso::ParForChunking::kStatic),
        [dim, vp](int i, int end) {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += pixelIters(i, dim, vp);
          }
          benchmark::DoNotOptimize(lsum);
        });
  }
}

// EXPERIMENT: kStatic with maxThreads limited to the number of L2-sharing
// groups (== physical cores on AMD Zen / Intel without SMT-shared L2).
// Tests whether reducing chunk count from numHWThreads to numL2Groups
// (letting SMT siblings cooperate on the same range via the kernel scheduler)
// matches TBB's parallel utilization advantage.
template <Region region>
void BM_dispenso_static_l2(benchmark::State& state) {
  const int num_threads = state.range(0) - 1;
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  dispenso::resizeGlobalThreadPool(static_cast<size_t>(num_threads));
  auto& pool = dispenso::globalThreadPool();

  // Limit maxThreads to numL2Groups so we get one chunk per L2-cache group.
  // On Zen 4 with SMT, this is one chunk per physical core; SMT siblings end
  // up not getting a chunk and (ideally) stay quiet so they don't compete
  // for execution units with the worker.
  const int32_t numL2Groups = static_cast<int32_t>(dispenso::CpuSet::l2CacheGroups().size());
  dispenso::ParForOptions options;
  options.maxThreads = static_cast<uint32_t>(std::max(1, numL2Groups));

  for (auto UNUSED_VAR : state) {
    dispenso::TaskSet tasks(pool);
    dispenso::parallel_for(
        tasks,
        dispenso::makeChunkedRange(0, numPixels, dispenso::ParForChunking::kStatic),
        [dim, vp](int i, int end) {
          uint64_t lsum = 0;
          for (; i != end; ++i) {
            lsum += pixelIters(i, dim, vp);
          }
          benchmark::DoNotOptimize(lsum);
        },
        options);
  }
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

template <Region region>
void BM_tbb_simple(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    // simple_partitioner: splits down to the grain size and stops. With grain
    // size = numPixels / (num_threads * 16) we mimic dispenso's kAuto chunk
    // count, so any TBB advantage from auto_partitioner's reactive splitting
    // is removed and the remaining gap reflects runtime/scheduling overhead.
    int grainSize = std::max(1, numPixels / (num_threads * 16));
    sum = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, numPixels, grainSize),
        uint64_t{0},
        [dim, vp](const tbb::blocked_range<int>& r, uint64_t init) -> uint64_t {
          for (int i = r.begin(); i != r.end(); ++i) {
            init += pixelIters(i, dim, vp);
          }
          return init;
        },
        [](uint64_t x, uint64_t y) { return x + y; },
        tbb::simple_partitioner{});
  }
  benchmark::DoNotOptimize(sum);
}

template <Region region>
void BM_tbb_static(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    tbb_compat::task_scheduler_init initsched(num_threads);
    // static_partitioner: one chunk per thread, no work stealing — direct
    // analog of dispenso's kStatic. Lets us isolate TBB's per-iteration
    // overhead from any partitioning differences.
    sum = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, numPixels),
        uint64_t{0},
        [dim, vp](const tbb::blocked_range<int>& r, uint64_t init) -> uint64_t {
          for (int i = r.begin(); i != r.end(); ++i) {
            init += pixelIters(i, dim, vp);
          }
          return init;
        },
        [](uint64_t x, uint64_t y) { return x + y; },
        tbb::static_partitioner{});
  }
  benchmark::DoNotOptimize(sum);
}
#endif // !BENCHMARK_WITHOUT_TBB

#if defined(_OPENMP)
template <Region region>
void BM_omp_static(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int dim = static_cast<int>(state.range(1));
  const int numPixels = dim * dim;
  const Viewport vp = viewportFor(region);

  omp_set_num_threads(num_threads);

  uint64_t sum = 0;
  for (auto UNUSED_VAR : state) {
    sum = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int i = 0; i < numPixels; ++i) {
      sum += pixelIters(i, dim, vp);
    }
  }
  benchmark::DoNotOptimize(sum);
}

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
  for (int dim : {kSmallDim, kMediumDim, kLargeDim}) {
    for (int i : benchmarkThreadCounts()) {
      b->Args({i, dim});
    }
  }
}

// Serial baselines: one per region+size combo.
BENCHMARK_TEMPLATE(BM_serial, Region::kFullSet, kSmallDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kFullSet, kMediumDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kBoundary, kSmallDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kBoundary, kMediumDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kInterior, kSmallDim);
BENCHMARK_TEMPLATE(BM_serial, Region::kInterior, kMediumDim);

#if defined(_OPENMP)
BENCHMARK_TEMPLATE(BM_omp_static, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_static, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_static, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_guided, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_guided, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_omp_guided, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
#endif

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK_TEMPLATE(BM_tbb, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_simple, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_simple, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_simple, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_static, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_static, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_static, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
#endif

BENCHMARK_TEMPLATE(BM_dispenso_static, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static_l2, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static_l2, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_static_l2, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_auto, Region::kInterior)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_fine, Region::kFullSet)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_fine, Region::kBoundary)->Apply(CustomArguments)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_fine, Region::kInterior)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
