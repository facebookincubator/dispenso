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

#ifdef _POSIX_C_SOURCE
struct rusage g_rusage;

inline void startRusage() {
  getrusage(RUSAGE_SELF, &g_rusage);
}

inline double duration(struct timeval start, struct timeval end) {
  return (1e6 * end.tv_sec + end.tv_usec - 1e6 * start.tv_sec + start.tv_usec) * 1e-6;
}

inline void endRusage(benchmark::State& state) {
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);

  double userTime = duration(g_rusage.ru_utime, res.ru_utime);
  double sysTime = duration(g_rusage.ru_stime, res.ru_stime);

  state.counters["\t0 User"] = userTime;
  state.counters["\t1 System"] = sysTime;
}
#else
inline void startRusage() {}
inline void endRusage(benchmark::State& state) {}
#endif //_POSIX_C_SOURCE
