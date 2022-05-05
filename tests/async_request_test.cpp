/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <thread>

#include <dispenso/async_request.h>

#include <gtest/gtest.h>

TEST(AsyncRequest, SequentialAsExpected) {
  dispenso::AsyncRequest<int> req;

  EXPECT_FALSE(req.updateRequested());
  EXPECT_FALSE(req.tryEmplaceUpdate(5));
  EXPECT_FALSE(req.getUpdate());

  req.requestUpdate();

  EXPECT_TRUE(req.updateRequested());

  EXPECT_FALSE(req.getUpdate());

  EXPECT_TRUE(req.tryEmplaceUpdate(0));

  auto result = req.getUpdate();
  EXPECT_TRUE(result);
  EXPECT_EQ(0, result.value());
}

TEST(AsyncRequest, AsyncAsExpected) {
  dispenso::AsyncRequest<int> req;
  std::atomic<bool> running(true);
  std::thread t([&req, &running]() {
    int next = 0;
    while (running.load(std::memory_order_relaxed)) {
      if (req.updateRequested()) {
        req.tryEmplaceUpdate(next++);
      }
    }
  });

  int sum = 0;
  int sumExpected = 0;
  for (int i = 0; i < 5000; ++i) {
    sumExpected += i;

    req.requestUpdate();
    while (true) {
      auto result = req.getUpdate();
      if (result.has_value()) {
        sum += result.value();
        break;
      }
    }
  }

  running.store(false, std::memory_order_release);
  t.join();

  EXPECT_EQ(sum, sumExpected);
}
