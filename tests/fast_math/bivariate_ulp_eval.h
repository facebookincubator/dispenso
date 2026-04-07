/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reusable harness for exhaustive ULP evaluation of two-argument fast_math
// functions. Uses Halton low-discrepancy sequences for uniform 2D domain
// coverage and dispenso::parallel_for for parallel evaluation.
//
// Usage pattern (in a per-function *_ulp_eval.cpp):
//
//   #include "bivariate_ulp_eval.h"
//
//   // Define ground truth, domains, and batch runners (scalar + SIMD).
//   // Call evalFunc2D() for each variant.
//
// The batch runner interface:
//   void runner(const float* xs, const float* ys, float* out, int32_t n);
// n is always a multiple of kBatchSize (64). Scalar runners process one
// element at a time; SIMD runners process kLanes at a time (4/8/16).

#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <dispenso/parallel_for.h>

#include <dispenso/fast_math/util.h>

namespace dispenso {
namespace fast_math {

// --- Halton sequence ---

// Compute element i of the Halton sequence in base b. Returns [0, 1).
inline double halton(uint64_t i, uint32_t b) {
  double f = 1.0;
  double r = 0.0;
  uint64_t idx = i;
  while (idx > 0) {
    f /= static_cast<double>(b);
    r += f * static_cast<double>(idx % b);
    idx /= b;
  }
  return r;
}

// Map a Halton value in [0, 1) to [lo, hi]. Uses log-space mapping when the
// range spans >100x (better coverage of both small and large values).
inline float mapToRange(double h, float lo, float hi) {
  if (lo > 0.0f && hi / lo > 100.0f) {
    double log_lo = std::log(static_cast<double>(lo));
    double log_hi = std::log(static_cast<double>(hi));
    return static_cast<float>(std::exp(log_lo + h * (log_hi - log_lo)));
  }
  return static_cast<float>(static_cast<double>(lo) + h * (static_cast<double>(hi) - lo));
}

// --- Types ---

struct Domain2D {
  float x_lo, x_hi;
  float y_lo, y_hi;
  const char* label;
};

struct EvalResult2D {
  uint32_t maxUlp;
  float worstX;
  float worstY;
  uint64_t count3; // samples with >= 3 ULP error
  uint64_t validSamples;
};

using GroundTruth2D = float (*)(float, float);

// Batch size for Halton sample generation. Must be a multiple of the widest
// SIMD width (64 bytes / 4 bytes = 16 lanes for AVX-512). 64 gives 4 full
// AVX-512 vectors per batch.
static constexpr int32_t kBatchSize = 64;

// --- Core evaluation ---

// Evaluate a two-argument function over a 2D domain using Halton sampling.
// gt: ground truth function (double-precision internally).
// runner: batch processor — processes kBatchSize (x,y) pairs from aligned arrays.
// domain: 2D domain to sample.
// numSamples: total number of Halton samples.
template <typename BatchRunner>
EvalResult2D
evalDomain2D(GroundTruth2D gt, BatchRunner runner, const Domain2D& domain, uint64_t numSamples) {
  // Round up to multiple of kBatchSize for clean batch processing.
  uint64_t numBatches = (numSamples + kBatchSize - 1) / kBatchSize;

  std::atomic<uint32_t> globalMaxUlp{0};
  std::atomic<float> globalWorstX{0.0f};
  std::atomic<float> globalWorstY{0.0f};
  std::atomic<uint64_t> globalCount3{0};
  std::atomic<uint64_t> globalValid{0};

  dispenso::parallel_for(
      dispenso::makeChunkedRange(uint64_t{0}, numBatches, uint64_t{256}),
      [&](uint64_t batchBegin, uint64_t batchEnd) {
        alignas(64) float xs[kBatchSize];
        alignas(64) float ys[kBatchSize];
        alignas(64) float results[kBatchSize];

        uint32_t localMax = 0;
        float localWorstX = 0.0f;
        float localWorstY = 0.0f;
        uint64_t localCount3 = 0;
        uint64_t localValid = 0;

        for (uint64_t bi = batchBegin; bi < batchEnd; ++bi) {
          uint64_t sampleBase = bi * kBatchSize;

          // Generate Halton samples for this batch.
          int32_t batchN = static_cast<int32_t>(
              std::min(static_cast<uint64_t>(kBatchSize), numSamples - sampleBase));
          for (int32_t j = 0; j < batchN; ++j) {
            uint64_t idx = sampleBase + static_cast<uint64_t>(j) + 1; // +1: skip degenerate i=0
            xs[j] = mapToRange(halton(idx, 2), domain.x_lo, domain.x_hi);
            ys[j] = mapToRange(halton(idx, 3), domain.y_lo, domain.y_hi);
          }
          // Pad remainder with safe values (won't be compared).
          for (int32_t j = batchN; j < kBatchSize; ++j) {
            xs[j] = domain.x_lo;
            ys[j] = domain.y_lo;
          }

          // Run approximation on full batch (SIMD needs aligned, full-width loads).
          runner(xs, ys, results, kBatchSize);

          // Compare only the real samples against ground truth.
          for (int32_t j = 0; j < batchN; ++j) {
            float g = gt(xs[j], ys[j]);
            float a = results[j];

            if (!std::isfinite(g) || g == 0.0f || !std::isfinite(a))
              continue;

            ++localValid;
            uint32_t d = float_distance(g, a);
            if (d >= 3)
              ++localCount3;
            if (d > localMax) {
              localMax = d;
              localWorstX = xs[j];
              localWorstY = ys[j];
            }
          }
        }

        globalCount3.fetch_add(localCount3, std::memory_order_relaxed);
        globalValid.fetch_add(localValid, std::memory_order_relaxed);
        uint32_t prev = globalMaxUlp.load(std::memory_order_relaxed);
        while (localMax > prev) {
          if (globalMaxUlp.compare_exchange_weak(prev, localMax, std::memory_order_relaxed)) {
            globalWorstX.store(localWorstX, std::memory_order_relaxed);
            globalWorstY.store(localWorstY, std::memory_order_relaxed);
            break;
          }
        }
      });

  return {
      globalMaxUlp.load(),
      globalWorstX.load(),
      globalWorstY.load(),
      globalCount3.load(),
      globalValid.load()};
}

// --- Reporting ---

// Evaluate all domains for a function and print a table.
template <typename BatchRunner, int32_t N>
void evalFunc2D(
    const char* name,
    GroundTruth2D gt,
    BatchRunner runner,
    Domain2D (&domains)[N],
    uint64_t numSamples) {
  printf("%-28s  %5s  %10s  %12s  %-s\n", name, "ULP", ">=3ULP", "valid", "Worst (x, y)");
  for (int32_t i = 0; i < N; ++i) {
    auto r = evalDomain2D(gt, runner, domains[i], numSamples);
    printf(
        "  %-26s  %5u  %10llu  %12llu  (%.6g, %.6g)\n",
        domains[i].label,
        r.maxUlp,
        static_cast<unsigned long long>(r.count3),
        static_cast<unsigned long long>(r.validSamples),
        r.worstX,
        r.worstY);
  }
}

// --- CLI helpers ---

struct EvalOptions {
  uint64_t numSamples = 1000000000ULL; // 1B default
  const char* filter = nullptr;
};

inline EvalOptions parseEvalOptions(int argc, char** argv) {
  EvalOptions opts;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
      opts.numSamples = static_cast<uint64_t>(std::atof(argv[++i]));
    } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
      opts.filter = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [--samples N] [--filter name]\n", argv[0]);
      printf("  --samples N    Halton samples per domain (default: 1e9)\n");
      printf("  --filter name  Only run variants containing 'name'\n");
    }
  }
  return opts;
}

inline bool shouldRun(const EvalOptions& opts, const char* name) {
  return opts.filter == nullptr || strstr(name, opts.filter) != nullptr;
}

} // namespace fast_math
} // namespace dispenso
