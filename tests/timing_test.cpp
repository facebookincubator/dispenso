/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/timing.h>

#include <chrono>
#include <cmath>
#include <thread>

#include <gtest/gtest.h>

// These tests are disabled under TSAN because timing-based tests are inherently
// sensitive to instrumentation overhead and thread scheduling, making them flaky
// in sanitizer builds.

#if !DISPENSO_HAS_TSAN

TEST(Timing, GetTimeReturnsNonNegative) {
  double t = dispenso::getTime();
  EXPECT_GE(t, 0.0);
}

TEST(Timing, GetTimeIsMonotonic) {
  double prev = dispenso::getTime();
  for (int i = 0; i < 100; ++i) {
    double cur = dispenso::getTime();
    EXPECT_GE(cur, prev);
    prev = cur;
  }
}

TEST(Timing, GetTimeProgresses) {
  double start = dispenso::getTime();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  double end = dispenso::getTime();

  // After sleeping 10ms, time should have advanced by at least 5ms
  // (allowing for timing imprecision)
  EXPECT_GT(end - start, 0.005);
}

TEST(Timing, StatisticalAccuracy) {
  // Run many iterations and verify that most measurements are within tolerance
  // of std::chrono. This is a statistical test that allows for some outliers
  // due to OS scheduling, context switches, etc.

  constexpr int kIterations = 1000;
  constexpr int kMinSuccesses = 900; // 90% must pass
  constexpr double kToleranceSeconds = 2e-6; // 2 microseconds tolerance

  int successes = 0;

  for (int i = 0; i < kIterations; ++i) {
    auto chronoStart = std::chrono::high_resolution_clock::now();
    double dispensoStart = dispenso::getTime();

    // Small busy-wait to accumulate some time
    volatile int dummy = 0;
    for (int j = 0; j < 1000; ++j) {
      dummy += j;
    }
    (void)dummy;

    auto chronoEnd = std::chrono::high_resolution_clock::now();
    double dispensoEnd = dispenso::getTime();

    double chronoElapsed = std::chrono::duration<double>(chronoEnd - chronoStart).count();
    double dispensoElapsed = dispensoEnd - dispensoStart;

    // Check if the measurements are within tolerance
    double diff = std::abs(chronoElapsed - dispensoElapsed);
    if (diff < kToleranceSeconds) {
      ++successes;
    }
  }

  EXPECT_GE(successes, kMinSuccesses)
      << "Expected at least " << kMinSuccesses << " out of " << kIterations
      << " timing measurements to be within " << (kToleranceSeconds * 1e6)
      << "us tolerance, but only " << successes << " were.";
}

TEST(Timing, LongerDurationAccuracy) {
  // Test accuracy over a longer duration by comparing dispenso timing to chrono
  // We use sleep to introduce a delay, but measure with both timing systems
  constexpr int kIterations = 10;
  constexpr double kSleepSeconds = 0.02; // 20ms minimum sleep
  constexpr double kToleranceRatio = 0.10; // 10% tolerance between timing systems

  for (int i = 0; i < kIterations; ++i) {
    auto chronoStart = std::chrono::steady_clock::now();
    double dispensoStart = dispenso::getTime();

    std::this_thread::sleep_for(
        std::chrono::microseconds(static_cast<int64_t>(kSleepSeconds * 1e6)));

    double dispensoEnd = dispenso::getTime();
    auto chronoEnd = std::chrono::steady_clock::now();

    double dispensoElapsed = dispensoEnd - dispensoStart;
    double chronoElapsed = std::chrono::duration<double>(chronoEnd - chronoStart).count();

    // Both should report at least the sleep duration (sleep is a minimum wait)
    EXPECT_GE(dispensoElapsed, kSleepSeconds * 0.9)
        << "Iteration " << i << ": dispenso elapsed time " << dispensoElapsed
        << " is too short compared to sleep duration " << kSleepSeconds;

    // dispenso and chrono should agree within tolerance
    double diff = std::abs(chronoElapsed - dispensoElapsed);
    double maxDiff = chronoElapsed * kToleranceRatio;
    EXPECT_LE(diff, maxDiff) << "Iteration " << i << ": dispenso (" << dispensoElapsed
                             << "s) and chrono (" << chronoElapsed << "s) differ by " << diff
                             << "s, which exceeds " << (kToleranceRatio * 100) << "% tolerance";
  }
}

TEST(Timing, RapidCalls) {
  // Verify that rapid successive calls don't produce anomalies
  constexpr int kIterations = 10000;
  double times[kIterations];

  for (int i = 0; i < kIterations; ++i) {
    times[i] = dispenso::getTime();
  }

  // All times should be monotonically non-decreasing
  for (int i = 1; i < kIterations; ++i) {
    EXPECT_GE(times[i], times[i - 1]) << "Time went backwards at index " << i;
  }

  // Total elapsed time should be small (< 1 second for 10k calls)
  double totalElapsed = times[kIterations - 1] - times[0];
  EXPECT_LT(totalElapsed, 1.0) << "10000 getTime() calls took " << totalElapsed << " seconds";
}

#endif // !DISPENSO_HAS_TSAN
