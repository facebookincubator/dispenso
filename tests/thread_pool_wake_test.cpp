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
  PoolWakeState state(16, 8);
  EXPECT_EQ(state.numThreads(), 16);
  EXPECT_EQ(state.numGroups(), 2);
  EXPECT_EQ(state.groupSize(), 8);
}

TEST(PoolWakeState, ConstructionMultipleGroups) {
  PoolWakeState state(64, 8);
  EXPECT_EQ(state.numThreads(), 64);
  EXPECT_EQ(state.numGroups(), 8);
  EXPECT_EQ(state.groupSize(), 8);
}

TEST(PoolWakeState, ConstructionPartialLastGroup) {
  PoolWakeState state(20, 8);
  EXPECT_EQ(state.numThreads(), 20);
  EXPECT_EQ(state.numGroups(), 3);
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
// Wake N + Cascade Tests
// =============================================================================

TEST(PoolWakeState, WakeNNoSleepers) {
  PoolWakeState state(16);
  EXPECT_EQ(state.totalSleeping(), 0);
  state.wakeN(5);
  EXPECT_EQ(state.totalSleeping(), 0);
}

TEST(PoolWakeState, WakeNWakesSeedGroup) {
  PoolWakeState state(4);

  state.enterSleep(2);
  uint32_t epochBefore = state.waiterFor(2).current();

  state.wakeN(3);

  // The waiter for thread 2's group should have been bumped.
  EXPECT_NE(state.waiterFor(2).current(), epochBefore);
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

  uint32_t epochBefore = state.waiterFor(0).current();

  state.wakeN(4);

  // wakeN broadcasts to the seed group — epoch should be bumped.
  EXPECT_NE(state.waiterFor(0).current(), epochBefore);
}

TEST(PoolWakeState, WakeRangeWakesInRange) {
  PoolWakeState state(32, 16);

  // Sleep threads in both groups
  state.enterSleep(0); // Group 0
  state.enterSleep(5); // Group 0
  state.enterSleep(16); // Group 1

  uint32_t e0 = state.waiterFor(0).current();
  uint32_t e16 = state.waiterFor(16).current();

  // Wake only the first 8 threads (group 0 only)
  state.wakeRange(8);

  // Group 0's epoch should be bumped (has sleepers in range)
  EXPECT_NE(state.waiterFor(0).current(), e0);

  // Group 1's epoch should NOT be bumped (out of range)
  EXPECT_EQ(state.waiterFor(16).current(), e16);
}

TEST(PoolWakeState, WakeRangePartialGroup) {
  PoolWakeState state(32, 16);

  state.enterSleep(0); // Group 0, in range
  state.enterSleep(10); // Group 0, out of range (count=8) but same group
  state.enterSleep(16); // Group 1, out of range

  uint32_t e0 = state.waiterFor(0).current();
  uint32_t e16 = state.waiterFor(16).current();

  state.wakeRange(8);

  // Group 0's epoch should be bumped (has sleepers in the [0,8) range).
  // Note: thread 10 is in group 0 but outside the count=8 range; however
  // wakeRange masks only the bits within range before broadcasting, so the
  // wake count reflects only in-range sleepers.
  EXPECT_NE(state.waiterFor(0).current(), e0);

  // Group 1's epoch should NOT be bumped (entirely out of range)
  EXPECT_EQ(state.waiterFor(16).current(), e16);
}

// =============================================================================
// Process Budget Tests
// =============================================================================

TEST(PoolWakeState, ProcessBudgetZero) {
  PoolWakeState state(16);
  EXPECT_EQ(state.processBudget(0), -1);
}

TEST(PoolWakeState, ProcessBudgetWakesPeerGroup) {
  // Small groupSize so wakeN with N > groupSize forces a cascade.
  PoolWakeState state(32, 4, 4);

  // Put threads to sleep in two groups so wakeN sets a cascade bit on g0
  // pointing at g1 (g1's bit set on g0.wakePending).
  for (int32_t i = 0; i < 4; ++i) {
    state.enterSleep(i); // Group 0
  }
  for (int32_t i = 4; i < 8; ++i) {
    state.enterSleep(i); // Group 1
  }

  // wakeN with N > groupSize forces a cascade: g0 is the seed, g1's bit
  // gets set on g0.wakePending. The producer wakes g0; a g0 thread would
  // normally drain wakePending via processBudget. Simulate that here by
  // calling processBudget directly from a g0 thread (thread 0).
  state.wakeN(8);

  // Thread 0's processBudget should claim the g1 bit and broadcast to g1.
  int32_t target = state.processBudget(0);
  EXPECT_GE(target, 0) << "processBudget should have claimed a cascade target";
}

TEST(PoolWakeState, ProcessBudgetReturnsMinusOneWhenNoPending) {
  PoolWakeState state(32, 4, 4);

  // Only one group sleeping — wakeN sets no cascade bits.
  state.enterSleep(0);
  state.wakeN(1);

  // No wakePending bits to claim, so processBudget on g0 returns -1.
  EXPECT_EQ(state.processBudget(0), -1);
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

TEST(PoolWakeState, WakeAllBumpsAllEpochs) {
  constexpr int32_t kNumThreads = 8;
  PoolWakeState state(kNumThreads);

  // Put only even threads to sleep
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    state.enterSleep(i);
  }

  uint32_t epochs[kNumThreads];
  for (int32_t i = 0; i < kNumThreads; ++i) {
    epochs[i] = state.waiterFor(i).current();
  }

  state.wakeAll();

  // wakeAll bumps epochs for groups with sleepers; threads self-exit via
  // exitSleep when they wake. Verify epochs were bumped for sleeping groups.
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    EXPECT_NE(state.waiterFor(i).current(), epochs[i])
        << "Thread " << i << " group epoch not bumped by wakeAll";
  }

  // Simulate thread wakeup by calling exitSleep
  for (int32_t i = 0; i < kNumThreads; i += 2) {
    state.exitSleep(i);
  }
  EXPECT_EQ(state.totalSleeping(), 0);

  // Re-enter sleep and verify we can wake again (no stuck state)
  for (int32_t i = 0; i < kNumThreads; ++i) {
    state.enterSleep(i);
  }
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
