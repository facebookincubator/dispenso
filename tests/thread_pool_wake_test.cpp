/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/thread_pool_wake.h>

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::detail::PoolWakeState;

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
// Cascade Wake Tests
// =============================================================================

TEST(PoolWakeState, CascadeWakeSeedNoSleepers) {
  PoolWakeState state(16);
  // No sleepers — cascadeWakeSeed should still bump epochs but issue no
  // wake syscalls; report cold path NOT taken.
  uint32_t epochBefore = state.waiterFor(0).current();
  EXPECT_FALSE(state.cascadeWakeSeed(5));
  // Epoch should advance for affected groups even with no sleepers, so
  // a spinning worker that hasn't loop-checked yet won't park.
  EXPECT_NE(state.waiterFor(0).current(), epochBefore);
}

TEST(PoolWakeState, CascadeWakeSeedWakesSeedGroup) {
  // 32 threads, gs=8, 4 groups. Sleeper in g0 ensures producer wakes g0.
  PoolWakeState state(32, 8);
  state.enterSleep(2);
  uint32_t epochBefore = state.waiterFor(2).current();

  EXPECT_TRUE(state.cascadeWakeSeed(32));
  // Seed group's epoch should advance.
  EXPECT_NE(state.waiterFor(2).current(), epochBefore);
}

TEST(PoolWakeState, CascadeWakeSeedZero) {
  PoolWakeState state(16);
  state.enterSleep(0);
  EXPECT_FALSE(state.cascadeWakeSeed(0));
  // Thread 0 should still be marked sleeping (no bit cleared).
  EXPECT_TRUE(state.tryClaimSleeper(0));
}

// =============================================================================
// Wake API Tests
// =============================================================================

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

TEST(PoolWakeState, WakeRangeWakesInRange) {
  PoolWakeState state(32, 16); // groupSize 16 → group 0: 0..15, group 1: 16..31

  state.enterSleep(0); // group 0
  state.enterSleep(16); // group 1

  uint32_t eGroup0 = state.waiterFor(0).current();
  uint32_t eGroup1 = state.waiterFor(16).current();

  // wakeRange(8) covers group 0 only. Wakes are group-granular: the in-range
  // group's epoch is bumped (parked workers return from waitFor and re-check);
  // out-of-range groups are untouched. Sleep bits are cleared by each woken
  // worker in exitSleep(), not by the waker, so we assert on epochs here.
  state.wakeRange(8);

  EXPECT_NE(state.waiterFor(0).current(), eGroup0); // group 0 woken
  EXPECT_EQ(state.waiterFor(16).current(), eGroup1); // group 1 untouched
}

// =============================================================================
// Cascade Wake Tests
// =============================================================================

TEST(PoolWakeState, CascadeWakeSeedPartialCount) {
  // gs=8, 32 threads, 4 groups. With count=16, only g0 and g1 are in range.
  PoolWakeState state(32, 8);
  state.enterSleep(0); // g0
  state.enterSleep(8); // g1
  state.enterSleep(16); // g2 — OUT of range for count=16

  uint32_t e0 = state.waiterFor(0).current();
  uint32_t e1 = state.waiterFor(8).current();
  uint32_t e2 = state.waiterFor(16).current();

  EXPECT_TRUE(state.cascadeWakeSeed(16));

  // g0 and g1 bump; g2 should be unaffected.
  EXPECT_NE(state.waiterFor(0).current(), e0);
  EXPECT_NE(state.waiterFor(8).current(), e1);
  EXPECT_EQ(state.waiterFor(16).current(), e2);
}

TEST(PoolWakeState, CascadeTargetForLevel1) {
  // 32 threads, gs=8, 4 groups.
  // Level 1: g0's threads 0..7 cascade to g1..g4 (but only 3 cascade slots
  // are filled since there are only 3 other groups).
  PoolWakeState state(32, 8);

  // g0's threads cascade to the level-1 targets g1, g2, g3.
  EXPECT_EQ(state.cascadeTargetFor(0, 32), 1);
  EXPECT_EQ(state.cascadeTargetFor(1, 32), 2);
  EXPECT_EQ(state.cascadeTargetFor(2, 32), 3);
  // Slots 3..7 in g0 should have no cascade target (we ran out of groups).
  EXPECT_EQ(state.cascadeTargetFor(3, 32), -1);
  EXPECT_EQ(state.cascadeTargetFor(7, 32), -1);
}

TEST(PoolWakeState, CascadeTargetForCountGate) {
  // gs=8, 32 threads. With count=16 only g0/g1 are in range; cascade
  // targets that point at g2/g3 should be filtered out.
  PoolWakeState state(32, 8);

  EXPECT_EQ(state.cascadeTargetFor(0, 32), 1);
  EXPECT_EQ(state.cascadeTargetFor(0, 16), 1); // g1 still in range
  EXPECT_EQ(state.cascadeTargetFor(1, 32), 2);
  EXPECT_EQ(state.cascadeTargetFor(1, 16), -1); // g2 out of range
}

TEST(PoolWakeState, CascadeTargetForLevel2) {
  // 192 threads, gs=8, 24 groups. Forces level-2 cascade.
  PoolWakeState state(192, 8);

  // Level 1: g0's 8 threads cascade to g1..g8.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(state.cascadeTargetFor(i, 192), 1 + i);
  }
  // Level 2: g1's threads (idx 8..15) cascade to g9..g16.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(state.cascadeTargetFor(8 + i, 192), 9 + i);
  }
  // g2's threads (idx 16..22) cascade to g17..g23 (7 targets, 1 slot empty).
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(state.cascadeTargetFor(16 + i, 192), 17 + i);
  }
  EXPECT_EQ(state.cascadeTargetFor(23, 192), -1);
  // Anything in g3+ should have no cascade target (all groups assigned).
  EXPECT_EQ(state.cascadeTargetFor(24, 192), -1);
  EXPECT_EQ(state.cascadeTargetFor(100, 192), -1);
}

TEST(PoolWakeState, CascadeWakeIndividualGroup) {
  PoolWakeState state(32, 8);
  // Sleeper in g2 should be woken via cascadeWake(g2).
  state.enterSleep(16);
  uint32_t before = state.waiterFor(16).current();
  state.cascadeWake(2);
  EXPECT_NE(state.waiterFor(16).current(), before);
}

TEST(PoolWakeState, CascadeWakeEmptyGroupOnlyBumps) {
  PoolWakeState state(32, 8);
  // No sleepers in g2 — cascadeWake should still bump (no syscall).
  uint32_t before = state.waiterFor(16).current();
  state.cascadeWake(2);
  EXPECT_NE(state.waiterFor(16).current(), before);
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

TEST(PoolWakeState, WakeAllBumpsEpochsAndDrains) {
  constexpr int32_t kNumThreads = 8;
  PoolWakeState state(kNumThreads);

  for (int32_t i = 0; i < kNumThreads; ++i) {
    state.enterSleep(i);
  }
  uint32_t epochs[kNumThreads];
  for (int32_t i = 0; i < kNumThreads; ++i) {
    epochs[i] = state.waiterFor(i).current();
  }

  // wakeAll bumps every group's epoch so parked workers return from waitFor.
  // It does NOT clear sleep bits or totalSleeping — each woken worker clears
  // its own bit via exitSleep(). We mirror that worker behavior here.
  state.wakeAll();
  for (int32_t i = 0; i < kNumThreads; ++i) {
    EXPECT_NE(state.waiterFor(i).current(), epochs[i]) << "Thread " << i << " not woken";
    state.exitSleep(i);
  }
  EXPECT_EQ(state.totalSleeping(), 0);

  // Re-enter sleep and verify we can wake again (no stuck state).
  for (int32_t i = 0; i < kNumThreads; ++i) {
    state.enterSleep(i);
  }
  for (int32_t i = 0; i < kNumThreads; ++i) {
    epochs[i] = state.waiterFor(i).current();
  }
  state.wakeAll();
  for (int32_t i = 0; i < kNumThreads; ++i) {
    EXPECT_NE(state.waiterFor(i).current(), epochs[i]) << "Re-entry: thread " << i << " not woken";
    state.exitSleep(i);
  }
  EXPECT_EQ(state.totalSleeping(), 0);
}
