/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_id.h>

#include <unordered_set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

TEST(ThreadId, Repeatable) {
  constexpr int kRounds = 100;
  constexpr int kThreadsPerRound = 8;
  for (int round = 0; round < kRounds; ++round) {
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreadsPerRound; ++i) {
      threads.emplace_back([]() {
        constexpr int kTrials = 1000;
        auto id = dispenso::threadId();

        for (int i = 0; i < kTrials; ++i) {
          EXPECT_EQ(id, dispenso::threadId());
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }
  }
}

TEST(ThreadId, Unique) {
  constexpr int kRounds = 1000;
  constexpr int kThreadsPerRound = 8;

  std::vector<uint64_t> ids(kRounds * kThreadsPerRound);
  std::atomic<size_t> slot(0);

  for (int round = 0; round < kRounds; ++round) {
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreadsPerRound; ++i) {
      threads.emplace_back([&ids, &slot]() {
        ids[slot.fetch_add(1, std::memory_order_relaxed)] = dispenso::threadId();
      });
    }

    for (auto& t : threads) {
      t.join();
    }
  }

  std::unordered_set<uint64_t> uniquenessSet;
  for (uint64_t id : ids) {
    EXPECT_TRUE(uniquenessSet.insert(id).second);
  }
}
