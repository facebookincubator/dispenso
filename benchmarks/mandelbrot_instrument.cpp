/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Standalone instrumentation harness for the mandelbrot workload. Runs one
// parallel_for (dispenso auto, dispenso fine, TBB auto, or TBB simple),
// captures per-chunk (thread_id, start, end), and reports:
//   - total chunk count
//   - chunks-per-thread distribution (min/mean/max)
//   - "affinity hit rate": the fraction of chunks whose immediately-preceding
//     range was executed by the SAME thread. Higher = better L1 reuse.
//
// Run as: mandelbrot_instrument <impl> <num_threads> <dim>
//   impl: dispenso_adaptive | dispenso_fine | tbb_auto | tbb_simple
//   num_threads: e.g. 192
//   dim: e.g. 1024

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dispenso/parallel_for.h>

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb_compat.h"

struct Viewport {
  double reMin, reMax, imMin, imMax;
};

static constexpr int kMaxIters = 4000;

// Three Mandelbrot viewports matching the main benchmark.
enum class Region { kFullSet, kBoundary, kInterior };
static Viewport viewportFor(Region r) {
  switch (r) {
    case Region::kFullSet:
      return {-2.5, 1.0, -1.15, 1.15};
    case Region::kBoundary:
      return {-0.78, -0.72, 0.08, 0.13};
    case Region::kInterior:
      return {-0.12, -0.08, 0.65, 0.69};
  }
  return {-2.5, 1.0, -1.15, 1.15};
}
static Viewport kViewport = viewportFor(Region::kBoundary);

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

static inline uint32_t pixelIters(int idx, int width, const Viewport& vp) {
  int x = idx % width;
  int y = idx / width;
  double dx = (vp.reMax - vp.reMin) / static_cast<double>(width);
  double dy = (vp.imMax - vp.imMin) / static_cast<double>(width);
  double cr = vp.reMin + (x + 0.5) * dx;
  double ci = vp.imMin + (y + 0.5) * dy;
  return mandelbrotIters(cr, ci, kMaxIters);
}

struct ChunkRecord {
  std::thread::id tid;
  int start;
  int end;
};

struct ChunkCollector {
  std::mutex mu;
  std::vector<ChunkRecord> chunks;

  void record(int start, int end) {
    std::lock_guard<std::mutex> g(mu);
    chunks.push_back({std::this_thread::get_id(), start, end});
  }
};

static void runDispensoAdaptive(int numThreads, int numPixels, int dim, ChunkCollector& collector) {
  dispenso::ThreadPool pool(numThreads - 1);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.defaultChunking = dispenso::ParForChunking::kAuto;
  uint64_t sum = 0;
  std::mutex sumMu;
  dispenso::parallel_for(
      tasks,
      0,
      numPixels,
      [dim, &collector, &sum, &sumMu](int i, int end) {
        collector.record(i, end);
        uint64_t lsum = 0;
        for (; i != end; ++i) {
          lsum += pixelIters(i, dim, kViewport);
        }
        std::lock_guard<std::mutex> g(sumMu);
        sum += lsum;
      },
      options);
  (void)sum;
}

static void runDispensoFine(int numThreads, int numPixels, int dim, ChunkCollector& collector) {
  dispenso::ThreadPool pool(numThreads - 1);
  const int chunkSize = std::max(1, numPixels / (256 * std::max(1, numThreads - 1)));
  dispenso::TaskSet tasks(pool);
  uint64_t sum = 0;
  std::mutex sumMu;
  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(0, numPixels, chunkSize),
      [dim, &collector, &sum, &sumMu](int i, int end) {
        collector.record(i, end);
        uint64_t lsum = 0;
        for (; i != end; ++i) {
          lsum += pixelIters(i, dim, kViewport);
        }
        std::lock_guard<std::mutex> g(sumMu);
        sum += lsum;
      });
  (void)sum;
}

static void runTbbAuto(int numThreads, int numPixels, int dim, ChunkCollector& collector) {
  tbb_compat::task_scheduler_init initsched(numThreads);
  uint64_t sum = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, numPixels),
      uint64_t{0},
      [dim, &collector](const tbb::blocked_range<int>& r, uint64_t init) -> uint64_t {
        collector.record(r.begin(), r.end());
        for (int i = r.begin(); i != r.end(); ++i) {
          init += pixelIters(i, dim, kViewport);
        }
        return init;
      },
      [](uint64_t x, uint64_t y) { return x + y; });
  (void)sum;
}

static void runTbbSimple(int numThreads, int numPixels, int dim, ChunkCollector& collector) {
  tbb_compat::task_scheduler_init initsched(numThreads);
  int grainSize = std::max(1, numPixels / (numThreads * 16));
  uint64_t sum = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, numPixels, grainSize),
      uint64_t{0},
      [dim, &collector](const tbb::blocked_range<int>& r, uint64_t init) -> uint64_t {
        collector.record(r.begin(), r.end());
        for (int i = r.begin(); i != r.end(); ++i) {
          init += pixelIters(i, dim, kViewport);
        }
        return init;
      },
      [](uint64_t x, uint64_t y) { return x + y; },
      tbb::simple_partitioner{});
  (void)sum;
}

static void analyze(const ChunkCollector& collector, const std::string& label, bool dumpHead) {
  std::vector<ChunkRecord> chunks = collector.chunks;
  // Sort by start index so we can detect "same-thread executes adjacent
  // ranges" (a proxy for L1 cache reuse on the next chunk).
  std::sort(chunks.begin(), chunks.end(), [](const ChunkRecord& a, const ChunkRecord& b) {
    return a.start < b.start;
  });

  std::unordered_map<std::thread::id, int> perThread;
  // Map each thread::id to a short integer label so we can print compact runs.
  std::unordered_map<std::thread::id, int> tidLabel;
  int nextLabel = 0;
  int sizeMin = std::numeric_limits<int>::max();
  int sizeMax = 0;
  uint64_t sizeSum = 0;
  for (const auto& c : chunks) {
    perThread[c.tid]++;
    if (tidLabel.find(c.tid) == tidLabel.end()) {
      tidLabel[c.tid] = nextLabel++;
    }
    int sz = c.end - c.start;
    sizeMin = std::min(sizeMin, sz);
    sizeMax = std::max(sizeMax, sz);
    sizeSum += static_cast<uint64_t>(sz);
  }

  int affinityHits = 0;
  int affinityTotal = 0;
  for (size_t i = 1; i < chunks.size(); ++i) {
    // Only count if the previous chunk is truly adjacent to this one. If
    // there's a gap (some other thread took the in-between range), it isn't
    // a fair "same thread = L1 reuse" measurement.
    if (chunks[i - 1].end != chunks[i].start) {
      continue;
    }
    ++affinityTotal;
    if (chunks[i - 1].tid == chunks[i].tid) {
      ++affinityHits;
    }
  }

  int perMin = std::numeric_limits<int>::max();
  int perMax = 0;
  for (auto& kv : perThread) {
    perMin = std::min(perMin, kv.second);
    perMax = std::max(perMax, kv.second);
  }
  double perMean = perThread.empty() ? 0.0 : double(chunks.size()) / double(perThread.size());
  double sizeMean = chunks.empty() ? 0.0 : double(sizeSum) / double(chunks.size());

  // If no chunks/threads were recorded the *Min accumulators are still INT_MAX;
  // report 0 instead of a garbage sentinel.
  if (chunks.empty()) {
    sizeMin = 0;
  }
  if (perThread.empty()) {
    perMin = 0;
  }

  printf("=== %s ===\n", label.c_str());
  printf("  total chunks:           %zu\n", chunks.size());
  printf("  threads that ran:       %zu\n", perThread.size());
  printf("  chunks/thread:          min=%d  mean=%.1f  max=%d\n", perMin, perMean, perMax);
  printf("  chunk size (items):     min=%d  mean=%.1f  max=%d\n", sizeMin, sizeMean, sizeMax);
  double affinityPct =
      affinityTotal == 0 ? 0.0 : 100.0 * double(affinityHits) / double(affinityTotal);
  printf(
      "  adjacent-chunk affinity: %d/%d (%.1f%%)  [higher = better L1 reuse]\n",
      affinityHits,
      affinityTotal,
      affinityPct);

  if (dumpHead) {
    // Show the first 40 chunks in [start..end) order with the executing
    // thread's short label. Runs of the same label are visible as locality.
    printf("  first 40 chunks (sorted by start) [tid]: start..end\n");
    size_t n = std::min<size_t>(40, chunks.size());
    for (size_t i = 0; i < n; ++i) {
      printf("    [%3d] %8d..%-8d\n", tidLabel[chunks[i].tid], chunks[i].start, chunks[i].end);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(
        stderr,
        "usage: %s <impl> <region> <num_threads> <dim>\n"
        "  impl: dispenso_adaptive | dispenso_fine | tbb_auto | tbb_simple\n"
        "  region: fullset | boundary | interior\n",
        argv[0]);
    return 1;
  }
  std::string impl = argv[1];
  std::string region = argv[2];
  int numThreads = std::atoi(argv[3]);
  int dim = std::atoi(argv[4]);
  int numPixels = dim * dim;

  if (region == "fullset") {
    kViewport = viewportFor(Region::kFullSet);
  } else if (region == "boundary") {
    kViewport = viewportFor(Region::kBoundary);
  } else if (region == "interior") {
    kViewport = viewportFor(Region::kInterior);
  } else {
    fprintf(stderr, "unknown region: %s\n", region.c_str());
    return 1;
  }

  ChunkCollector collector;
  if (impl == "dispenso_adaptive") {
    runDispensoAdaptive(numThreads, numPixels, dim, collector);
  } else if (impl == "dispenso_fine") {
    runDispensoFine(numThreads, numPixels, dim, collector);
  } else if (impl == "tbb_auto") {
    runTbbAuto(numThreads, numPixels, dim, collector);
  } else if (impl == "tbb_simple") {
    runTbbSimple(numThreads, numPixels, dim, collector);
  } else {
    fprintf(stderr, "unknown impl: %s\n", impl.c_str());
    return 1;
  }

  char label[256];
  snprintf(
      label,
      sizeof(label),
      "%s  region=%s  threads=%d  dim=%d",
      impl.c_str(),
      region.c_str(),
      numThreads,
      dim);
  analyze(collector, label, true);
  return 0;
}
