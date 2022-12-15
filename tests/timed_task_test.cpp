/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/completion_event.h>
#include <dispenso/schedulable.h>
#include <dispenso/task_set.h>
#include <dispenso/timed_task.h>

#include <gtest/gtest.h>

using namespace std::chrono_literals;

// Some baseline error epsilon values.
constexpr double kp50Epsilon = 0.5e-3;
constexpr double kp90Epsilon = 1.0e-3;
constexpr double kp95Epsilon = 2.0e-3;

bool g_isRealtime = false;

// Adjust basic epsilons via multipliers based on real-time status and platform.
double errAdjust(double eps) {
  if (g_isRealtime) {
#if defined(__MACH__)
    return eps * 0.5;
#elif defined(_WIN32)
    return 2.0 * eps;
#else
    return eps * 0.33;
#endif // platform
  } else {
    return eps;
  }
}

dispenso::TimedTaskScheduler& testScheduler() {
  static const bool rtEnv = getenv("TEST_WITH_REALTIME") != nullptr;
  if (rtEnv) {
    static dispenso::TimedTaskScheduler sched(dispenso::ThreadPriority::kRealtime);
    if (!g_isRealtime) {
      std::cerr << "Running with real-time priority" << std::endl;
    }
    g_isRealtime = true;
    return sched;
  } else {
    return dispenso::globalTimedTaskScheduler();
  }
}

TEST(TimedTaskTest, RunLikelyZeroTimes) {
  // 40 ms
  constexpr double kWaitLen = 0.04;

  double start;
  double calledTime;
  bool calledAtAll = false;

  {
    dispenso::ThreadPool pool(1);
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &calledAtAll]() {
          calledTime = dispenso::getTime();
          calledAtAll = true;
          return true;
        },
        start + kWaitLen);
  }

  if (calledAtAll) {
    EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp95Epsilon));
    EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp95Epsilon));
  }
}

TEST(TimedTaskTest, MoveAndRunLikelyZeroTimes) {
  // 40 ms
  constexpr double kWaitLen = 0.04;

  double start;
  double calledTime;
  bool calledAtAll = false;

  {
    dispenso::ThreadPool pool(1);
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &calledAtAll]() {
          calledTime = dispenso::getTime();
          calledAtAll = true;
          return true;
        },
        start + kWaitLen);

    dispenso::TimedTask movedTo = std::move(task);
  }

  if (calledAtAll) {
    EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp95Epsilon));
    EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp95Epsilon));
  }
}

TEST(TimedTaskTest, RunOnce) {
  // 40 ms
  constexpr double kWaitLen = 0.04;

  double start;
  double calledTime;
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &fin]() {
          calledTime = dispenso::getTime();
          fin.notify();
          return true;
        },
        start + kWaitLen);

    fin.wait();
  }

  EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp95Epsilon));
}

constexpr size_t k100Times = 100;

TEST(TimedTaskTest, RunPeriodic) {
  // 2 ms
  constexpr double kWaitLen = 0.002;

  double start;
  std::vector<double> calledTimes(k100Times);
  std::atomic<size_t> count(0);
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTimes, &count, &fin]() {
          auto cur = count.fetch_add(1, std::memory_order_release);
          calledTimes[cur] = dispenso::getTime();
          if (cur + 1 == k100Times) {
            fin.notify();
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kWaitLen,
        k100Times);
    fin.wait();
  }

  double prev = start;
  for (double& t : calledTimes) {
    double absT = std::abs(kWaitLen - (t - prev));
    prev = t;
    t = absT;
  }

  std::sort(calledTimes.begin(), calledTimes.end());

  EXPECT_LE(calledTimes[50], errAdjust(kp50Epsilon));
  EXPECT_LE(calledTimes[90], errAdjust(kp90Epsilon));
  EXPECT_LE(calledTimes[95], errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunPeriodicSteady) {
  // 2 ms
  constexpr double kWaitLen = 0.002;

  double start;
  std::vector<double> calledTimes(k100Times);
  std::atomic<size_t> count(0);
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTimes, &count, &fin]() {
          auto cur = count.fetch_add(1, std::memory_order_release);
          calledTimes[cur] = dispenso::getTime();
          if (cur + 1 == k100Times) {
            fin.notify();
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kWaitLen,
        k100Times,
        dispenso::TimedTaskType::kSteady);

    fin.wait();
  }

  double prev = start;
  for (double& t : calledTimes) {
    double absT = std::abs(kWaitLen - (t - prev));
    prev += kWaitLen;
    t = absT;
  }

  std::sort(calledTimes.begin(), calledTimes.end());

  EXPECT_LE(calledTimes[50], errAdjust(kp50Epsilon));
  EXPECT_LE(calledTimes[90], errAdjust(kp90Epsilon));
  EXPECT_LE(calledTimes[95], errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunPeriodicDontWait) {
  // 2 ms
  constexpr double kWaitLen = 0.002;

  double start;
  std::vector<double> calledTimes(2 * k100Times);
  std::atomic<size_t> count(0);
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTimes, &count, &fin]() {
          auto cur = count.fetch_add(1, std::memory_order_release);
          calledTimes[cur] = dispenso::getTime();
          if (cur + 1 == k100Times) {
            fin.notify();
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kWaitLen,
        calledTimes.size());
    fin.wait();
  }

  EXPECT_GE(count.load(std::memory_order_acquire), k100Times);

  calledTimes.resize(k100Times);

  double prev = start;
  for (double& t : calledTimes) {
    double absT = std::abs(kWaitLen - (t - prev));
    prev = t;
    t = absT;
  }

  std::sort(calledTimes.begin(), calledTimes.end());

  EXPECT_LE(calledTimes[50], errAdjust(kp50Epsilon));
  EXPECT_LE(calledTimes[90], errAdjust(kp90Epsilon));
  EXPECT_LE(calledTimes[95], errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunPeriodicSteadyUnderLoad) {
  // 2 ms
  constexpr double kWaitLen = 0.002;

  double start;
  std::vector<double> calledTimes(k100Times);
  std::atomic<size_t> count(0);
  std::atomic<size_t> dummy(0);

  dispenso::ThreadPool& pool = dispenso::globalThreadPool();
  dispenso::ConcurrentTaskSet tasks(pool);
  dispenso::CompletionEvent fin;

  tasks.schedule([&tasks, &count, times = k100Times, &dummy]() {
    while (count.load(std::memory_order_acquire) < times) {
      tasks.schedule([&dummy]() { dummy.fetch_add(1, std::memory_order_relaxed); });
    }
  });

  {
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTimes, &count, &fin]() {
          auto cur = count.fetch_add(1, std::memory_order_acq_rel);
          calledTimes[cur] = dispenso::getTime();
          if (cur + 1 == k100Times) {
            fin.notify();
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kWaitLen,
        k100Times,
        dispenso::TimedTaskType::kSteady);
    fin.wait();
  }

  tasks.wait();

  double prev = start;
  for (double& t : calledTimes) {
    double absT = std::abs(kWaitLen - (t - prev));
    prev += kWaitLen;
    t = absT;
  }

  std::sort(calledTimes.begin(), calledTimes.end());

#if !DISPENSO_HAS_TSAN
  double kMultiplier = g_isRealtime ? 1.0 : 5.0;
#else
  double kMultiplier = 15.0;
#endif // TSAN

  EXPECT_LE(calledTimes[50], kMultiplier * errAdjust(kp50Epsilon));
  EXPECT_LE(calledTimes[90], kMultiplier * errAdjust(kp90Epsilon));
  EXPECT_LE(calledTimes[95], kMultiplier * errAdjust(kp95Epsilon));
}

constexpr size_t k10Times = 10;

TEST(TimedTaskTest, RunDetach) {
  // 2 ms
  constexpr double kWaitLen = 0.007;

  double start;
  std::vector<double> calledTimes(k10Times);
  std::atomic<size_t> count(0);
  dispenso::CompletionEvent fin;

  {
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        dispenso::globalThreadPool(),
        [&calledTimes, &count, &fin]() {
          auto cur = count.fetch_add(1, std::memory_order_release);
          calledTimes[cur] = dispenso::getTime();
          if (cur + 1 == k10Times) {
            fin.notify();
          }
          return true;
        },
        start + kWaitLen,
        kWaitLen,
        k10Times);
    task.detach();
  }

  fin.wait();

  double prev = start;
  for (double& t : calledTimes) {
    double absT = std::abs(kWaitLen - (t - prev));
    prev = t;
    t = absT;
  }

  std::sort(calledTimes.begin(), calledTimes.end());

  EXPECT_LE(calledTimes[5], errAdjust(kp50Epsilon));
  EXPECT_LE(calledTimes[9], errAdjust(kp90Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayByDuration) {
  // 40 ms
  constexpr auto kWaitLenChrono = 40ms;
  constexpr double kWaitLen = 0.04;

  double start;
  double calledTime;
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &fin]() {
          calledTime = dispenso::getTime();
          fin.notify();
          return true;
        },
        kWaitLenChrono);

    fin.wait();
  }

  EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayToTimePoint) {
  // 40 ms
  constexpr auto kWaitLenChrono = 40ms;
  constexpr double kWaitLen = 0.04;

  double start;
  double calledTime;
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &fin]() {
          calledTime = dispenso::getTime();
          fin.notify();
          return true;
        },
        std::chrono::high_resolution_clock::now() + kWaitLenChrono);

    fin.wait();
  }

  EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayByDurationWithPeriod) {
  constexpr auto kWaitLenChrono = 6ms;
  constexpr double kWaitLen = 0.006;
  constexpr auto kPeriodChrono = 2ms;
  constexpr auto kPeriod = 0.002;

  double start;
  double calledTime[2];
  std::atomic<int> slot(0);

  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &slot, &fin]() {
          auto cur = slot.fetch_add(1, std::memory_order_acq_rel);
          calledTime[cur] = dispenso::getTime();
          if (cur + 1 == 2) {
            fin.notify();
            return false;
          }
          return true;
        },
        kWaitLenChrono,
        kPeriodChrono,
        2);

    fin.wait();
  }

  EXPECT_GT(calledTime[0] - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[0] - start, kWaitLen + errAdjust(kp95Epsilon));

  EXPECT_GT(calledTime[1] - calledTime[0], kPeriod - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[1] - calledTime[0], kPeriod + errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayByDurationWithPeriodSteady) {
  constexpr auto kWaitLenChrono = 6ms;
  constexpr double kWaitLen = 0.006;
  constexpr auto kPeriodChrono = 2ms;
  constexpr auto kPeriod = 0.002;

  double start;
  double calledTime[2];
  std::atomic<int> slot(0);

  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &slot, &fin]() {
          auto cur = slot.fetch_add(1, std::memory_order_acq_rel);
          calledTime[cur] = dispenso::getTime();
          if (cur + 1 == 2) {
            fin.notify();
            return false;
          }
          return true;
        },
        kWaitLenChrono,
        kPeriodChrono,
        2);

    fin.wait();
  }

  EXPECT_GT(calledTime[0] - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[0] - start, kWaitLen + errAdjust(kp95Epsilon));

  EXPECT_GT(calledTime[1] - start, (kWaitLen + kPeriod) - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[1] - start, (kWaitLen + kPeriod) + errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayToTimepointWithPeriod) {
  constexpr auto kWaitLenChrono = 6ms;
  constexpr double kWaitLen = 0.006;
  constexpr auto kPeriodChrono = 2ms;
  constexpr auto kPeriod = 0.002;

  double start;
  double calledTime[2];
  std::atomic<int> slot(0);

  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &slot, &fin]() {
          auto cur = slot.fetch_add(1, std::memory_order_acq_rel);
          calledTime[cur] = dispenso::getTime();
          if (cur + 1 == 2) {
            fin.notify();
            return false;
          }
          return true;
        },
        std::chrono::high_resolution_clock::now() + kWaitLenChrono,
        kPeriodChrono,
        2);

    fin.wait();
  }

  EXPECT_GT(calledTime[0] - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[0] - start, kWaitLen + errAdjust(kp95Epsilon));

  EXPECT_GT(calledTime[1] - calledTime[0], kPeriod - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[1] - calledTime[0], kPeriod + errAdjust(kp95Epsilon));
}

TEST(TimedTaskTest, RunChronoDelayToTimepointWithPeriodSteady) {
  constexpr auto kWaitLenChrono = 6ms;
  constexpr double kWaitLen = 0.006;
  constexpr auto kPeriodChrono = 2ms;
  constexpr auto kPeriod = 0.002;

  double start;
  double calledTime[2];
  std::atomic<int> slot(0);

  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &slot, &fin]() {
          auto cur = slot.fetch_add(1, std::memory_order_acq_rel);
          calledTime[cur] = dispenso::getTime();
          if (cur + 1 == 2) {
            fin.notify();
            return false;
          }
          return true;
        },
        std::chrono::high_resolution_clock::now() + kWaitLenChrono,
        kPeriodChrono,
        2);

    fin.wait();
  }

  EXPECT_GT(calledTime[0] - start, kWaitLen - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[0] - start, kWaitLen + errAdjust(kp95Epsilon));

  EXPECT_GT(calledTime[1] - start, (kWaitLen + kPeriod) - errAdjust(kp95Epsilon));
  EXPECT_LT(calledTime[1] - start, (kWaitLen + kPeriod) + errAdjust(kp95Epsilon));
}

constexpr size_t kRunForever = std::numeric_limits<size_t>::max();

TEST(TimedTaskTest, RunOnceImmediatelyLongPeriod) {
  constexpr double kWaitLen = 0.0;
  constexpr double kPeriod = 100.0;

  double start;
  double calledTime;
  dispenso::CompletionEvent fin;

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&calledTime, &fin]() {
          calledTime = dispenso::getTime();
          fin.notify();
          return true;
        },
        start + kWaitLen,
        kPeriod,
        kRunForever);

    fin.wait();
  }

  EXPECT_GT(calledTime - start, kWaitLen - errAdjust(kp50Epsilon));
  EXPECT_LT(calledTime - start, kWaitLen + errAdjust(kp50Epsilon));
}

constexpr int kTerminateCount = 7;

TEST(TimedTaskTest, CancelViaProvidedFunction) {
  constexpr double kWaitLen = 0.004;
  // Make the period very tiny
  constexpr double kPeriod = 50e-6;

  double start;
  std::atomic<int> counter(0);

  {
    dispenso::ImmediateInvoker pool;
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        pool,
        [&counter]() {
          if (counter.fetch_add(1, std::memory_order_acq_rel) + 1 == kTerminateCount) {
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kPeriod,
        kRunForever);

    while (counter.load(std::memory_order_acquire) != kTerminateCount) {
    }

    // Task should be terminating
    std::this_thread::sleep_for(10ms);
  }

  EXPECT_EQ(counter.load(std::memory_order_acquire), kTerminateCount);
}

TEST(TimedTaskTest, CancelViaProvidedFunctionInThreadPool) {
  constexpr double kWaitLen = 0.004;
  // Make the period very tiny
  constexpr double kPeriod = 50e-6;

  double start;
  std::atomic<int> counter(0);

  {
    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        dispenso::globalThreadPool(),
        [&counter]() {
          if (counter.fetch_add(1, std::memory_order_acq_rel) + 1 == kTerminateCount) {
            return false;
          }
          return true;
        },
        start + kWaitLen,
        kPeriod,
        kRunForever);

    while (counter.load(std::memory_order_acquire) != kTerminateCount) {
    }

    // Task should be terminating
    std::this_thread::sleep_for(10ms);
  }

  EXPECT_GE(counter.load(std::memory_order_acquire), kTerminateCount);
  EXPECT_LE(counter.load(std::memory_order_acquire), kTerminateCount + 1);
}

TEST(TimedTaskTest, FunctionDestructionPostTaskDestruct) {
  constexpr double kWaitLen = 0.004;
  // Make the period very tiny
  constexpr double kPeriod = 50e-6;

  double start;

  {
    std::vector<size_t> dummyObj1;

    struct Dummy {
      std::vector<size_t>& vec;

      ~Dummy() {
        if (vec.size() != 0) {
          std::cerr << "INVALID CALL TO DESTRUCTOR" << std::endl;
          std::abort();
        }
      }

    } dummy{dummyObj1};

    start = dispenso::getTime();
    dispenso::TimedTask task = testScheduler().schedule(
        dispenso::globalThreadPool(),
        [dummy]() { return true; },
        start + kWaitLen,
        kPeriod,
        kRunForever);
    // Here task goes out of scope
  }

  // Goal: passed function destructor should never execute out here... if it does it will
  // incorrectly reference value that went out of scope
}
