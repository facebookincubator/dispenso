/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <deque>
#include <thread>

#include <dispenso/latch.h>

#include <gtest/gtest.h>

using namespace std::chrono_literals;

TEST(Latch, ArriveAndWait) {
  size_t publishData = 0;
  dispenso::Latch latch(3);

  std::deque<std::thread> threads;

  for (size_t i = 0; i < 2; ++i) {
    threads.emplace_back([&latch, &publishData]() {
      latch.arrive_and_wait();
      EXPECT_EQ(publishData, 3);
    });
  }

  // Give plenty of time for hijinx if there were any bug.
  std::this_thread::sleep_for(10ms);

  publishData = 3;

  // Wait cannot succeed until we also throw our hat in the ring, since we have 3 threads in the
  // group, but only two threads waiting to check for a new value of publishData.  We do this
  // after setting the value of publishData, from only one thread (main thread).  After
  // arrive_and_wait, wait succeeds, and waiting threads are woken, and they should see the correct
  // value of publishData.
  latch.arrive_and_wait();

  for (auto& t : threads) {
    t.join();
  }
}

TEST(Latch, CountDown) {
  size_t publishData = 0;
  dispenso::Latch latch(3);

  std::deque<std::thread> threads;

  for (size_t i = 0; i < 2; ++i) {
    threads.emplace_back([&latch, &publishData]() {
      latch.count_down();

      if (latch.try_wait()) {
        EXPECT_EQ(publishData, 3);
      } else {
        latch.wait();
        EXPECT_EQ(publishData, 3);
      }
    });
  }

  publishData = 3;

  // Wait cannot succeed until we also throw our hat in the ring, since we have 3 threads in the
  // group, but only two threads waiting to check for a new value of publishData.  We do this
  // after setting the value of publishData, from only one thread (main thread).  After count_down,
  // wait succeeds, and waiting threads are woken, and they should see the correct value of
  // publishData.
  latch.count_down();

  // Wait isn't required here.

  for (auto& t : threads) {
    t.join();
  }
}

TEST(Latch, ArriveAndWaitWithCountDown) {
  size_t publishData = 0;
  dispenso::Latch latch(3);

  std::deque<std::thread> threads;

  for (size_t i = 0; i < 2; ++i) {
    threads.emplace_back([&latch, &publishData]() {
      latch.arrive_and_wait();
      EXPECT_EQ(publishData, 3);
    });
  }

  publishData = 3;

  // Wait cannot succeed until we also throw our hat in the ring, since we have 3 threads in the
  // group, but only two threads waiting to check for a new value of publishData.  We do this
  // after setting the value of publishData, from only one thread (main thread).  After count_down,
  // wait succeeds, and waiting threads are woken, and they should see the correct value of
  // publishData.
  latch.count_down();

  // Wait isn't required here.

  for (auto& t : threads) {
    t.join();
  }
}
