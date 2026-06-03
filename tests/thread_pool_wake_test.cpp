/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_pool_wake.h>

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::PoolWakeState;

// =============================================================================
// Construction Tests
// =============================================================================

TEST(PoolWakeState, ConstructionZeroThreads) {
  PoolWakeState state(0);
  EXPECT_EQ(state.numThreads(), 0);
  EXPECT_EQ(state.numGroups(), 0);
}

TEST(PoolWakeState, ConstructionBasic) {
  PoolWakeState state(16, 16);
  EXPECT_EQ(state.numThreads(), 16);
  EXPECT_EQ(state.numGroups(), 1);
  EXPECT_EQ(state.groupSize(), 16);
}

TEST(PoolWakeState, ConstructionMultipleGroups) {
  PoolWakeState state(64, 16);
  EXPECT_EQ(state.numThreads(), 64);
  EXPECT_EQ(state.numGroups(), 4);
  EXPECT_EQ(state.groupSize(), 16);
}

TEST(PoolWakeState, ConstructionPartialLastGroup) {
  PoolWakeState state(20, 16);
  EXPECT_EQ(state.numThreads(), 20);
  EXPECT_EQ(state.numGroups(), 2);
}

TEST(PoolWakeState, ConstructionCustomGroupSize) {
  PoolWakeState state(32, 8);
  EXPECT_EQ(state.numThreads(), 32);
  EXPECT_EQ(state.numGroups(), 4);
  EXPECT_EQ(state.groupSize(), 8);
}

// =============================================================================
// Sleep Mask Tests
// =============================================================================

TEST(PoolWakeState, EnterExitSleep) {
  PoolWakeState state(16);

  // Initially no one is sleeping
  EXPECT_EQ(state.totalSleeping(), 0);
  EXPECT_FALSE(state.tryClaimSleeper(0));

  // Enter sleep
  state.enterSleep(0);
  EXPECT_EQ(state.totalSleeping(), 1);
  EXPECT_TRUE(state.tryClaimSleeper(0));

  // After claim, the bit is cleared — second claim fails
  EXPECT_FALSE(state.tryClaimSleeper(0));
}

TEST(PoolWakeState, ExitSleepClearsBit) {
  PoolWakeState state(16);

  state.enterSleep(5);
  EXPECT_EQ(state.totalSleeping(), 1);
  state.exitSleep(5);
  EXPECT_EQ(state.totalSleeping(), 0);

  // Bit was cleared by exitSleep — claim should fail
  EXPECT_FALSE(state.tryClaimSleeper(5));
}

TEST(PoolWakeState, MultipleSleepers) {
  PoolWakeState state(16);

  state.enterSleep(0);
  state.enterSleep(5);
  state.enterSleep(15);

  EXPECT_TRUE(state.tryClaimSleeper(0));
  EXPECT_TRUE(state.tryClaimSleeper(5));
  EXPECT_TRUE(state.tryClaimSleeper(15));

  // All claimed
  EXPECT_FALSE(state.tryClaimSleeper(0));
  EXPECT_FALSE(state.tryClaimSleeper(5));
  EXPECT_FALSE(state.tryClaimSleeper(15));
}

TEST(PoolWakeState, SleepersAcrossGroups) {
  PoolWakeState state(32, 16);

  state.enterSleep(0); // Group 0
  state.enterSleep(16); // Group 1

  EXPECT_TRUE(state.tryClaimSleeper(0));
  EXPECT_TRUE(state.tryClaimSleeper(16));
}

// =============================================================================
// Wake Budget Tests
// =============================================================================

TEST(PoolWakeState, WakeOneWithBudgetNoSleepers) {
  PoolWakeState state(16);
  // No one sleeping — should return false
  EXPECT_FALSE(state.wakeOneWithBudget(5));
}

TEST(PoolWakeState, WakeOneWithBudgetSetsAndWakes) {
  PoolWakeState state(4);

  state.enterSleep(2);
  uint32_t epochBefore = state.waiterFor(2).current();

  EXPECT_TRUE(state.wakeOneWithBudget(3));

  // Epoch should have been bumped
  EXPECT_NE(state.waiterFor(2).current(), epochBefore);

  // Sleep bit should be cleared (claimed by waker)
  EXPECT_FALSE(state.tryClaimSleeper(2));
}

TEST(PoolWakeState, WakeNZero) {
  PoolWakeState state(16);
  state.enterSleep(0);
  state.wakeN(0);
  // Thread 0 should still be "sleeping" (bit not cleared)
  EXPECT_TRUE(state.tryClaimSleeper(0));
}

TEST(PoolWakeState, WakeOneBumpsEpoch) {
  PoolWakeState state(4);

  state.enterSleep(1);
  uint32_t epochBefore = state.waiterFor(1).current();

  EXPECT_TRUE(state.wakeOne());
  EXPECT_NE(state.waiterFor(1).current(), epochBefore);
}

TEST(PoolWakeState, WakeOneNoSleepers) {
  PoolWakeState state(4);
  EXPECT_FALSE(state.wakeOne());
}

TEST(PoolWakeState, ClaimAndWakeOneReturnsIndex) {
  PoolWakeState state(8);

  state.enterSleep(3);
  uint32_t epochBefore = state.waiterFor(3).current();

  int32_t idx = state.claimAndWakeOne();
  EXPECT_EQ(idx, 3);

  // Epoch should be bumped
  EXPECT_NE(state.waiterFor(3).current(), epochBefore);

  // Bit should be cleared (claimed)
  EXPECT_FALSE(state.tryClaimSleeper(3));
}

TEST(PoolWakeState, ClaimAndWakeOneNoSleepers) {
  PoolWakeState state(8);
  EXPECT_EQ(state.claimAndWakeOne(), -1);
}

TEST(PoolWakeState, ClaimAndWakeOneNoDuplicate) {
  PoolWakeState state(8);

  state.enterSleep(2);

  int32_t first = state.claimAndWakeOne();
  EXPECT_EQ(first, 2);

  // Second call should find no sleepers
  int32_t second = state.claimAndWakeOne();
  EXPECT_EQ(second, -1);
}

TEST(PoolWakeState, WakeNOne) {
  PoolWakeState state(4);

  state.enterSleep(0);
  uint32_t epochBefore = state.waiterFor(0).current();

  state.wakeN(1);

  // n==1 delegates to wakeOne(), which bumps the epoch
  EXPECT_NE(state.waiterFor(0).current(), epochBefore);
}

TEST(PoolWakeState, WakeNMultiple) {
  PoolWakeState state(8, 8, 4);

  for (int32_t i = 0; i < 4; ++i) {
    state.enterSleep(i);
  }

  // n>1 delegates to wakeOneWithBudget(n-1), which claims one sleeper
  // and sets a cascade budget
  state.wakeN(4);

  // At least one thread should have been claimed
  int32_t stillSleeping = 0;
  for (int32_t i = 0; i < 4; ++i) {
    if (state.tryClaimSleeper(i)) {
      ++stillSleeping;
    }
  }
  EXPECT_LT(stillSleeping, 4);
}

TEST(PoolWakeState, WakeRangeWakesInRange) {
  PoolWakeState state(32, 16);

  // Sleep threads in both groups
  state.enterSleep(0); // Group 0
  state.enterSleep(5); // Group 0
  state.enterSleep(16); // Group 1

  uint32_t e0 = state.waiterFor(0).current();
  uint32_t e5 = state.waiterFor(5).current();

  // Wake only the first 8 threads (group 0 only)
  state.wakeRange(8);

  // Threads 0 and 5 should be claimed (in range)
  EXPECT_FALSE(state.tryClaimSleeper(0));
  EXPECT_FALSE(state.tryClaimSleeper(5));

  // Thread 16 should still be sleeping (out of range)
  EXPECT_TRUE(state.tryClaimSleeper(16));

  // Epochs for woken threads should be bumped
  EXPECT_NE(state.waiterFor(0).current(), e0);
  EXPECT_NE(state.waiterFor(5).current(), e5);
}

TEST(PoolWakeState, WakeRangePartialGroup) {
  PoolWakeState state(32, 16);

  state.enterSleep(0); // Group 0, in range
  state.enterSleep(10); // Group 0, out of range (count=8)
  state.enterSleep(16); // Group 1, out of range

  state.wakeRange(8);

  // Thread 0 should be claimed (in range)
  EXPECT_FALSE(state.tryClaimSleeper(0));

  // Thread 10 and 16 should still be sleeping (out of range)
  EXPECT_TRUE(state.tryClaimSleeper(10));
  EXPECT_TRUE(state.tryClaimSleeper(16));
}

// =============================================================================
// Process Budget Tests
// =============================================================================

TEST(PoolWakeState, ProcessBudgetZero) {
  PoolWakeState state(16);
  EXPECT_EQ(state.processBudget(0), 0);
}

TEST(PoolWakeState, ProcessBudgetWakesPeers) {
  PoolWakeState state(16, 16, 4);

  // Put thread 0 to sleep so wakeOneWithBudget can target it, plus several
  // peers for the cascade to find.
  state.enterSleep(0);
  for (int32_t i = 1; i < 8; ++i) {
    state.enterSleep(i);
  }

  // Wake thread 0 with budget=5 (via the real API)
  EXPECT_TRUE(state.wakeOneWithBudget(5));

  // Simulate thread 0 waking and processing its budget
  int32_t consumed = state.processBudget(0);
  EXPECT_EQ(consumed, 5);

  // Some peers should have been claimed by the cascade
  int32_t stillSleeping = 0;
  for (int32_t i = 1; i < 8; ++i) {
    if (state.tryClaimSleeper(i)) {
      ++stillSleeping;
    }
  }
  // 7 peers were sleeping; cascade should have woken at least 1
  EXPECT_LT(stillSleeping, 7);
}

TEST(PoolWakeState, ProcessBudgetPrefersOtherGroup) {
  PoolWakeState state(32, 16, 4);

  // Put thread 0 to sleep so wakeOneWithBudget can target it
  state.enterSleep(0);
  // Put threads to sleep in both groups for the cascade to find
  state.enterSleep(1); // Group 0
  state.enterSleep(17); // Group 1

  // Wake thread 0 with budget=1
  EXPECT_TRUE(state.wakeOneWithBudget(1));

  // Thread 0 processes its budget
  state.processBudget(0);

  // processBudget scans other groups first (breadth-first for NUMA
  // distribution), so it should wake thread 17 (group 1) before thread 1
  // (group 0). Thread 1 should still be sleeping.
  bool thread1StillSleeping = state.tryClaimSleeper(1);
  EXPECT_TRUE(thread1StillSleeping) << "Should have preferred other group (thread 17) first";
}

// =============================================================================
// Concurrent Tests
// =============================================================================

TEST(PoolWakeState, ConcurrentClaimNoDuplicateWake) {
  PoolWakeState state(16);

  state.enterSleep(5);

  // Multiple threads try to claim the same sleeper
  std::atomic<int32_t> claimCount{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&state, &claimCount]() {
      if (state.tryClaimSleeper(5)) {
        claimCount.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // Exactly one thread should have claimed it
  EXPECT_EQ(claimCount.load(), 1);
}

TEST(PoolWakeState, WakeAllWakesAllSleepers) {
  PoolWakeState state(4);

  // Put all 4 threads to sleep
  for (int32_t i = 0; i < 4; ++i) {
    state.enterSleep(i);
  }

  uint32_t epochs[4];
  for (int32_t i = 0; i < 4; ++i) {
    epochs[i] = state.waiterFor(i).current();
  }

  state.wakeAll();

  // All sleeping threads should have their epochs bumped
  for (int32_t i = 0; i < 4; ++i) {
    EXPECT_NE(state.waiterFor(i).current(), epochs[i]) << "Thread " << i << " waiter not bumped";
  }
}

TEST(PoolWakeState, WakeAllClearsSleepBits) {
  constexpr int32_t kNumThreads = 8;
  PoolWakeState state(kNumThreads);

  // Put only even threads to sleep
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    state.enterSleep(i);
  }

  state.wakeAll();

  // wakeAll clears sleep mask bits (via tryClaimSleeper) but does NOT
  // decrement totalSleeping — only exitSleep does that. Verify bits are
  // cleared, then simulate real thread behavior by calling exitSleep.
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    EXPECT_FALSE(state.tryClaimSleeper(i)) << "Thread " << i << " sleep bit not cleared by wakeAll";
  }
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    state.exitSleep(i);
  }
  EXPECT_EQ(state.totalSleeping(), 0);

  // Re-enter sleep and verify we can wake again (no stuck state)
  for (int32_t i = 0; i < kNumThreads; ++i) {
    state.enterSleep(i);
  }
  uint32_t epochs[kNumThreads];
  for (int32_t i = 0; i < kNumThreads; ++i) {
    epochs[i] = state.waiterFor(i).current();
  }
  state.wakeAll();
  for (int32_t i = 0; i < kNumThreads; ++i) {
    EXPECT_NE(state.waiterFor(i).current(), epochs[i]) << "Thread " << i << " not woken";
    state.exitSleep(i);
  }
  EXPECT_EQ(state.totalSleeping(), 0);
}
