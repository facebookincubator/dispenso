/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <deque>
#include <thread>

#include <dispenso/completion_event.h>

#include <gtest/gtest.h>

TEST(CompletionEvent, NotifyBeforeWait) {
  dispenso::CompletionEvent event;

  event.notify();
  // Should immediately return;
  event.wait();
}

TEST(CompletionEvent, NotifyBeforeWaitFor) {
  dispenso::CompletionEvent event;

  event.notify();
  // Should immediately return;
  EXPECT_TRUE(event.waitFor(std::chrono::microseconds(1)));
}

// In an ideal world, we could expect the following test to loop 10 times or so.  In reality, we
// can't make such guarantees when it comes to sleep() and wait() functions.  For instance, on Linux
// with 64 mostly-idle cores, 100 out of 100 runs of this test resulted in looping between 8 and 12
// times, even under TSAN.  On Mac with 4 less-idle cores, the test would pass about 90 out of 100.
// Inflating the interval to 7 to 13 passed 98 out of 100.  In the end, we cannot really count on
// any concrete number of times through the loop (think TSAN, think loaded machine, etc...), and so
// we simply let this test fall back to "will this time out?".
TEST(CompletionEvent, WaitForSomeTime) {
  dispenso::CompletionEvent event;

  std::thread t([&event]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.notify();
  });

  while (true) {
    if (event.waitFor(std::chrono::milliseconds(2))) {
      break;
    }
  }

  t.join();
}

TEST(CompletionEvent, WaitForSomeTimeWithReset) {
  dispenso::CompletionEvent event;
  std::atomic<bool> barrier(0);

  std::thread t([&event, &barrier]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.notify();

    while (!barrier.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.notify();
  });

  while (!(event.waitFor(std::chrono::milliseconds(2)))) {
  }

  EXPECT_TRUE(event.waitFor(std::chrono::microseconds(1))) << "This should immediately return true";

  // No threads waiting, nor notifying, so we can reset.
  event.reset();

  // Trigger the barrier so that the event can be notified.
  barrier.store(1, std::memory_order_release);

  while (true) {
    if (event.waitFor(std::chrono::milliseconds(2))) {
      break;
    }
  }

  t.join();
}

TEST(CompletionEvent, EffectiveBarrier) {
  dispenso::CompletionEvent event;

  std::deque<std::thread> threads;

  std::atomic<int> count(0);

  constexpr int kThreads = 4;

  for (size_t i = 0; i < kThreads; ++i) {
    threads.emplace_back([&event, &count]() {
      count.fetch_sub(1, std::memory_order_relaxed);
      event.wait();
      count.fetch_add(2, std::memory_order_relaxed);
    });
  }

  while (count.load(std::memory_order_acquire) > -kThreads) {
  }

  // Take a long rest in this thread.  This gives us a chance to ensure that the event cannot
  // spurious wake, and begin modifying "count".
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  EXPECT_EQ(-kThreads, count.load(std::memory_order_acquire));

  event.notify();

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(kThreads, count.load(std::memory_order_acquire));
}
