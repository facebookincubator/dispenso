/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/timing.h>

#include <chrono>
#include <cmath>

#include <dispenso/detail/timestamp.h>

#if defined(_WIN32)
#include <Windows.h>
#endif // _WIN32

#if defined(__MACH__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif // __MACH__

namespace dispenso {

#if defined(DISPENSO_HAS_TIMESTAMP)

#if !defined(__aarch64__)

static bool snapFreq(double& firstApprox) {
  switch (static_cast<int>(firstApprox)) {
    case 0:
      if (std::abs(int(firstApprox * 10.0)) <= 1) {
        firstApprox = 0.0;
        return true;
      }
      break;
    case 9:
      if (std::abs(int(firstApprox * 10.0) - 99) <= 1) {
        firstApprox = 10.0;

        return true;
      }
      break;
    case 3:
      if (std::abs(int(firstApprox * 10.0) - 33) <= 1) {
        firstApprox = 3.0 + 1.0 / 3.0;
        return true;
      }
      break;
    case 6:
      if (std::abs(int(firstApprox * 10.0) - 66) <= 1) {
        firstApprox = 6.0 + 2.0 / 3.0;
        return true;
      }
      break;
  }
  return false;
}

static double fallbackTicksPerSecond() {
  using namespace std::chrono_literals;
  constexpr double kChronoOverheadBias = 250e-9;

  auto baseStart = std::chrono::high_resolution_clock::now();
  auto start = detail::timestamp();
  std::this_thread::sleep_for(50ms);
  auto end = detail::timestamp();
  auto baseEnd = std::chrono::high_resolution_clock::now();

  auto base = std::chrono::duration<double>(baseEnd - baseStart).count() - kChronoOverheadBias;
  double firstApprox = (static_cast<double>(end - start)) / base;

  // Try to refine the approximation.  In some circumstances we can "snap" the frequency to a very
  // good guess that is off by less than one part in thousands.  Accuracy should already be quite
  // good in any case, but this allows us to improve in some cases.

  // Get first 3 digits
  firstApprox *= 1e-7;

  int firstInt = static_cast<int>(firstApprox);
  firstApprox -= firstInt;

  firstApprox *= 10.0;

  if (!snapFreq(firstApprox)) {
    int secondInt = static_cast<int>(firstApprox);
    firstApprox -= secondInt;
    firstApprox *= 10.0;
    snapFreq(firstApprox);
    firstApprox *= 0.1;
    firstApprox += secondInt;
  }

  firstApprox *= 0.1;

  firstApprox += firstInt;
  firstApprox *= 1e7;
  return firstApprox;
}
#endif // !__aarch64__

#if defined(__aarch64__)
static double ticksPerSecond() {
  uint64_t val;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
  return static_cast<double>(val);
}
#elif defined(__MACH__)
static double ticksPerSecond() {
  mach_timebase_info_data_t info;
  if (mach_timebase_info(&info) != KERN_SUCCESS) {
    return fallbackTicksPerSecond();
  }
  return 1e9 * static_cast<double>(info.denom) / static_cast<double>(info.numer);
}
#else
double ticksPerSecond() {
  return fallbackTicksPerSecond();
}
#endif

double getTime() {
  static double secondsPerTick = 1.0 / ticksPerSecond();
  static double startTime = static_cast<double>(detail::timestamp()) * secondsPerTick;

  double t = static_cast<double>(detail::timestamp()) * secondsPerTick;
  return t - startTime;
}
#else
double getTime() {
  static auto startTime = std::chrono::high_resolution_clock::now();
  auto cur = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double>(cur - startTime).count();
}
#endif // DISPENSO_HAS_TIMESTAMP

namespace {
// This should ensure that we initialize the time before main.
double g_dummyTime = getTime();
} // namespace

} // namespace dispenso
