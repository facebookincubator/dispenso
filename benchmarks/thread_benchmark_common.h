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

inline std::vector<int> pow2HalfStepThreads() {
  const int kRunningThreads = std::thread::hardware_concurrency();
  std::vector<int> result;
  result.push_back(1);
  for (int block = 2; block <= kRunningThreads; block *= 2) {
    int step = block / 2;

    for (int i = block; i < 2 * block && i <= kRunningThreads; i += step) {
      result.push_back(i);
    }
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
