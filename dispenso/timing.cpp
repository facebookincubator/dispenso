/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/timing.h>

#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>
#endif // _MSC_VER

namespace dispenso {
namespace {
#if defined(__x86_64__) || defined(_M_AMD64)
#define DISPENSO_HAS_TIMESTAMP
#if defined(_MSC_VER)
inline uint64_t rdtscp() {
  uint32_t ui;
  return __rdtscp(&ui);
}
#else
inline uint64_t rdtscp() {
  uint32_t lo, hi;
  __asm__ volatile("rdtscp"
                   : /* outputs */ "=a"(lo), "=d"(hi)
                   : /* no inputs */
                   : /* clobbers */ "%rcx");
  return (uint64_t)lo | (((uint64_t)hi) << 32);
}
#endif // OS
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define DISPENSO_HAS_TIMESTAMP
uint64_t rdtscp(void) {
  uint64_t val;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}
#endif // ARCH
} // namespace

#if defined(DISPENSO_HAS_TIMESTAMP)

std::pair<double, double> initTime() {
  using namespace std::chrono_literals;

  auto baseStart = std::chrono::high_resolution_clock::now();
  auto start = rdtscp();
  std::this_thread::sleep_for(10ms);
  auto end = rdtscp();
  auto baseEnd = std::chrono::high_resolution_clock::now();

  auto base = std::chrono::duration<double>(baseEnd - baseStart).count();
  double secondsPerCycle = base / (end - start);

  double startTime = start * secondsPerCycle;
  return {secondsPerCycle, startTime};
}

double getTime() {
  static std::pair<double, double> tmInfo = initTime();
  double& secondsPerCycle = tmInfo.first;
  double& startTime = tmInfo.second;

  double t = rdtscp() * secondsPerCycle;
  return t - startTime;
}
#else
double getTime() {
  static auto startTime = std::chrono::high_resolution_clock::now();
  auto cur = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(cur - startTime).count();
}
#endif // DISPENSO_HAS_TIMESTAMP

} // namespace dispenso
