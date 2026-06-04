/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef _POSIX_C_SOURCE
#include <sys/resource.h>
#endif // _POSIX_C_SOURCE

#include <cmath>
#include <iostream>
#include <thread>

#include "benchmark_common.h"

// Standard set of thread counts for thread-count-varying benchmarks:
// powers of 2 from 1 up to the number of hardware threads, plus the
// hardware-thread count itself if it isn't already a power of 2 (so a
// 166-thread SMT machine ends up measuring ..., 128, 166).
//
// The earlier half-step schedule (3, 6, 12, 24, 48, 96, ...) doubled
// sweep wall time without yielding meaningfully different signal —
// adjacent measurements were almost always within noise of each other.
inline std::vector<int> benchmarkThreadCounts() {
  const int kRunningThreads = std::thread::hardware_concurrency();
  std::vector<int> result;
  for (int n = 1; n <= kRunningThreads; n *= 2) {
    result.push_back(n);
  }
  if (!result.empty() && result.back() != kRunningThreads) {
    result.push_back(kRunningThreads);
  }
  return result;
}

#if defined(_POSIX_C_SOURCE) || defined(__MACH__)
struct rusage g_rusage;

inline void startRusage() {
  std::atomic_thread_fence(std::memory_order_acquire);
  getrusage(RUSAGE_SELF, &g_rusage);
  std::atomic_thread_fence(std::memory_order_release);
}

inline double duration(struct timeval start, struct timeval end) {
  return (end.tv_sec + 1e-6 * end.tv_usec) - (start.tv_sec + 1e-6 * start.tv_usec);
}

inline void endRusage(benchmark::State& state) {
  std::atomic_thread_fence(std::memory_order_acquire);
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  std::atomic_thread_fence(std::memory_order_release);

  double userTime = duration(g_rusage.ru_utime, res.ru_utime);
  double sysTime = duration(g_rusage.ru_stime, res.ru_stime);

  state.counters["\t0 User"] = userTime;
  state.counters["\t1 System"] = sysTime;
}
#else
inline void startRusage() {}
inline void endRusage(benchmark::State& state) {}
#endif //_POSIX_C_SOURCE

inline double getMean(const std::vector<double>& data) {
  double sum = 0.0;
  for (auto d : data) {
    sum += d;
  }
  return sum / data.size();
}

inline double getStddev(double mean, const std::vector<double>& data) {
  double sumsq = 0.0;
  for (auto d : data) {
    auto dev = mean - d;
    sumsq += dev * dev;
  }
  return std::sqrt(sumsq / data.size());
}

void doStats(const std::vector<double>& times, benchmark::State& state) {
  double mean = getMean(times);
  state.counters["mean"] = mean;
  state.counters["stddev"] = getStddev(mean, times);
}
