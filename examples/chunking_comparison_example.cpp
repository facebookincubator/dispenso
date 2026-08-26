/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example chunking_comparison_example.cpp
 * Compares serial execution, the default kStatic chunking, and kAdaptive on a
 * workload whose per-element cost grows across the range.
 *
 * This is the program linked from the README as "try it live"; keep the two in
 * step. Timings are only meaningful on a machine with real cores to itself.
 */

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include <dispenso/parallel_for.h>
#include <dispenso/timing.h>

constexpr size_t kN = 1 << 20;

// Cost per element grows 64x across the range, so an even split of the index
// space is a very uneven split of the work.
inline double compute(size_t i) {
  double acc = 0.0;
  const size_t iters = 1 + (i * 64) / kN;
  for (size_t k = 0; k < iters; ++k) {
    acc += std::exp(std::atan(static_cast<double>(i) * 1e-6 + static_cast<double>(k)));
    acc = std::log(acc);
  }
  return acc;
}

double run(dispenso::ParForChunking chunking, std::vector<double>& out) {
  dispenso::ParForOptions options;
  options.defaultChunking = chunking;

  const double start = dispenso::getTime();
  dispenso::parallel_for(size_t{0}, kN, [&](size_t i) { out[i] = compute(i); }, options);
  return (dispenso::getTime() - start) * 1e3;
}

double run(std::vector<double>& out) {
  const double start = dispenso::getTime();
  for (size_t i = 0; i < kN; ++i) {
    out[i] = compute(i);
  }
  return (dispenso::getTime() - start) * 1e3;
}

int main() {
  std::vector<double> out(kN);

  printf("Running with %d threads\n", static_cast<int>(std::thread::hardware_concurrency()));

  // Warm the pool so the first timed run does not pay thread start-up.
  run(dispenso::ParForChunking::kStatic, out);

  // kStatic is the default: one contiguous chunk per worker, decided up front.
  // Cheap when iterations cost the same; here the worker holding the tail does
  // far more work than the one holding the head.
  const double staticMs = run(dispenso::ParForChunking::kStatic, out);

  // kAdaptive still starts from one stripe per worker, but a worker that runs
  // out steals from a peer -- preferring one sharing its L3.
  const double adaptiveMs = run(dispenso::ParForChunking::kAdaptive, out);

  const double serialMs = run(out);

  printf("serial:     %7.2f ms\n", serialMs);
  printf("kStatic:    %7.2f ms\n", staticMs);
  printf("kAdaptive:  %7.2f ms\n", adaptiveMs);
  printf(
      "  Disregard these timings if running on Compiler Explorer.\n"
      "  These run on a shared machine with very few cores available.\n"
      "  It is entirely possible for any variant to be fastest on any given run.\n"
      "  Try this locally for better performance comparison\n");
  return 0;
}
