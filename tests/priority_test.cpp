/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cmath>

#include <dispenso/completion_event.h>
#include <dispenso/latch.h>
#include <dispenso/parallel_for.h>
#include <dispenso/priority.h>
#include <dispenso/timing.h>

#include <gtest/gtest.h>

// NOTE: This isn't suitable for an automated unit test for multiple reasons.  With OS
// scheduling we have some amount of nondeterminism.  Additionally, many (most?) machines will not
// have permissions for kHigh and kRealtime priorities depending on OS and policies.
//
//  On Linux, with permissions appropriate, this test passes about 14/15 times.  On the single
//  failure, I see an average sleep error like this:
// Expected: (info[2].error()) >= (info[3].error()), actual: 5.25784e-05 vs 5.41972e-05
// or about 50ish microseconds average error for both kHigh and kRealtime priorities.

using namespace std::chrono_literals;

struct ThreadInfo {
  uint64_t count = 0;
  double sleepErrorSum = 0.0;
  bool prioOk = false;

  double error() const {
    return sleepErrorSum / static_cast<double>(count);
  }
};

void run(
    size_t index,
    ThreadInfo& info,
    dispenso::CompletionEvent& notifier,
    dispenso::Latch& started) {
  switch (index) {
    case 0:
      info.prioOk = dispenso::setCurrentThreadPriority(dispenso::ThreadPriority::kLow);
      break;
    case 1:
      info.prioOk = dispenso::setCurrentThreadPriority(dispenso::ThreadPriority::kNormal);
      break;
    case 2:
      info.prioOk = dispenso::setCurrentThreadPriority(dispenso::ThreadPriority::kHigh);
      break;
    case 3:
      info.prioOk = dispenso::setCurrentThreadPriority(dispenso::ThreadPriority::kRealtime);
      break;
    default:
      info.prioOk = true;
      break;
  }

  // Ensure all threads reach this point before we begin, so that we don't let the first threads
  // make progress before the system is bogged down.
  started.arrive_and_wait();

  // Keep other threads busy.  If cores are idle, the result will be a crapshoot.
  if (index > 3) {
    while (!notifier.completed()) {
      ++info.count;
#if defined(DISPENSO_HAS_TSAN)
      // In TSAN atomics are implemented via reader/writer locks, and I believe these are not
      // guaranteeing progress.  We need to take some time out from the tight loop calling
      // notifier.completed() in order to allow the atomic write to succeed.
      std::this_thread::yield();
#endif // TSAN
    }
    return;
  }

  while (true) {
    double start = dispenso::getTime();
    if (!notifier.waitFor(1ms)) {
      double end = dispenso::getTime();
      ++info.count;
      info.sleepErrorSum += std::abs((end - start) - 1e-3);
    } else {
      break;
    }
  }
}

TEST(Priorty, PriorityGetsCycles) {
  dispenso::ParForOptions options;
  options.wait = false;

  int overloadConcurrency = 2 * std::thread::hardware_concurrency();

  if (sizeof(void*) == 4) {
    overloadConcurrency = std::min(overloadConcurrency, 62);
  }

  dispenso::ThreadPool pool(std::max<dispenso::ssize_t>(10, overloadConcurrency));

  std::vector<ThreadInfo> info(pool.numThreads());

  dispenso::CompletionEvent stop;
  dispenso::Latch started(static_cast<uint32_t>(pool.numThreads()));

  dispenso::TaskSet tasks(pool);
  dispenso::parallel_for(
      tasks,
      0,
      pool.numThreads(),
      [&info, &stop, &started](size_t index) { run(index, info[index], stop, started); },
      options);

  // Let threads wake about 5000 times.
  std::this_thread::sleep_for(5s);

  stop.notify();

  tasks.wait();

  for (auto& i : info) {
    EXPECT_TRUE(i.prioOk) << "Failed for " << &i - info.data();
  }

#if !defined(DISPENSO_HAS_TSAN)
  // TSAN messes with scheduling enough that all bets are off.
  EXPECT_GE(info[0].error(), info[1].error());
  EXPECT_GE(info[1].error(), info[2].error());
  EXPECT_GE(info[2].error(), info[3].error());
#endif // TSAN
}
