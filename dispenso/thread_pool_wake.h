/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file thread_pool_wake.h
 * @ingroup group_core
 * Wake infrastructure for ThreadPool's fork-join and central queue scheduling.
 *
 * Implements a budget-limited cascade wake strategy with per-thread EpochWaiters
 * and per-group atomic sleep masks. See docs/design/wake_cascade.md for the
 * design rationale.
 **/

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <dispenso/detail/epoch_waiter.h>
#include <dispenso/platform.h>

namespace dispenso {

// Platform-tunable wake parameters.
// BranchFactor: max children per node in the wake cascade tree.
// Higher values = fewer cascade levels but more serial futex calls per level.
// Override via -DDISPENSO_TUNE_WAKE_BRANCH_FACTOR=N for cross-platform tuning.
#if defined(DISPENSO_TUNE_WAKE_BRANCH_FACTOR)
constexpr int32_t kDefaultWakeBranchFactor = DISPENSO_TUNE_WAKE_BRANCH_FACTOR;
#elif defined(_WIN32)
// Windows: WaitOnAddress is more expensive per-call, so use wider fan-out
// to reduce cascade depth.
constexpr int32_t kDefaultWakeBranchFactor = 8;
#else
// Linux/macOS: futex is cheap per-call, so 4-wide gives the best
// balance of cascade depth vs per-level cost.
constexpr int32_t kDefaultWakeBranchFactor = 4;
#endif

/**
 * @brief Per-group state for tracking sleeping threads.
 *
 * Each group (up to 64 threads) has a sleep mask where bit i is set when
 * thread i in the group is sleeping on its EpochWaiter. Wakers atomically
 * test-and-clear bits to claim exclusive wake responsibility, preventing
 * redundant futex calls.
 */
struct DISPENSO_CACHELINE_ALIGNED GroupWakeState {
  std::atomic<uint64_t> sleepMask{0};
};

/**
 * @brief Cache-line-aligned block of EpochWaiters for one group.
 *
 * Each group shares one EpochWaiter (one futex address). The block is
 * cache-line aligned to prevent inter-group false sharing.
 */
static constexpr int32_t kMaxThreadsPerWakeGroup = 64;

// Number of threads sharing one EpochWaiter (and thus one futex address).
// A single futex_wake(INT_MAX) wakes all threads in a subgroup, reducing
// the number of syscalls for batch wakes by this factor. The tradeoff is
// that targeted wakes (cascade wakeOne) cause up to kWaiterSubgroupSize-1
// spurious wakes — those threads check for work, find none, and sleep again.
// Override via -DDISPENSO_TUNE_WAITER_SUBGROUP_SIZE=N for cross-platform tuning.
#if defined(DISPENSO_TUNE_WAITER_SUBGROUP_SIZE)
static constexpr int32_t kWaiterSubgroupSize = DISPENSO_TUNE_WAITER_SUBGROUP_SIZE;
#else
static constexpr int32_t kWaiterSubgroupSize = 4;
#endif

// distributeBudget declares EpochWaiter peerSlots[kWaiterSubgroupSize - 1];
// a value of 1 would yield a zero-length array (UB).
static_assert(kWaiterSubgroupSize >= 2, "kWaiterSubgroupSize must be at least 2");

struct DISPENSO_CACHELINE_ALIGNED WaiterBlock {
  detail::EpochWaiter waiters[kMaxThreadsPerWakeGroup / kWaiterSubgroupSize];
};

/**
 * @class PoolWakeState
 * @brief Wake infrastructure for a ThreadPool.
 *
 * Manages per-thread EpochWaiters, per-group sleep masks, and the
 * budget-limited cascade wake mechanism. Used by ThreadPool for both
 * fork-join (tree cascade via hasWork) and central queue (budget cascade
 * via wakeN) work dispatch.
 *
 * ## Thread Safety
 *
 * - `enterSleep` / `exitSleep`: called only by the owning thread
 * - `tryClaimSleeper`: safe to call from any thread (atomic test-and-clear)
 * - `wakeOne` / `wakeN`: safe to call from any thread
 * - `processBudget`: called only by the owning thread after waking
 */
class PoolWakeState {
 public:
  /**
   * @brief Construct wake state for a pool with the given number of threads.
   *
   * @param numThreads Total number of pool worker threads.
   * @param groupSize Number of threads per wake group (typically 16, matching
   *                  CpuSet topology groups).
   * @param branchFactor Maximum children per node in budget cascade.
   */
  DISPENSO_DLL_ACCESS PoolWakeState(
      int32_t numThreads,
#if defined(DISPENSO_TUNE_WAKE_GROUP_SIZE)
      int32_t groupSize = DISPENSO_TUNE_WAKE_GROUP_SIZE,
#elif defined(_WIN32)
      int32_t groupSize = 8,
#else
      int32_t groupSize = 16,
#endif
      int32_t branchFactor = kDefaultWakeBranchFactor);

  DISPENSO_DLL_ACCESS ~PoolWakeState();

  // Non-copyable, non-movable
  PoolWakeState(const PoolWakeState&) = delete;
  PoolWakeState& operator=(const PoolWakeState&) = delete;

  /**
   * @brief Get the EpochWaiter for a specific thread.
   *
   * Multiple threads (kWaiterSubgroupSize) share one EpochWaiter, so
   * bumpAndWakeAll() on this waiter wakes the entire subgroup with a
   * single futex syscall.
   */
  detail::EpochWaiter& waiterFor(int32_t threadIdx) {
    int32_t inGroup = threadIdx % groupSize_;
    return waiterBlocks_[static_cast<size_t>(threadIdx / groupSize_)]
        .waiters[static_cast<size_t>(inGroup / kWaiterSubgroupSize)];
  }

  /**
   * @brief Mark a thread as sleeping. Called before entering EpochWaiter sleep.
   *
   * Sets the thread's bit in its group's sleep mask.
   */
  void enterSleep(int32_t threadIdx) {
    int32_t group = threadIdx / groupSize_;
    int32_t bit = threadIdx % groupSize_;
    groupStates_[static_cast<size_t>(group)].sleepMask.fetch_or(
        uint64_t{1} << bit, std::memory_order_release);
    totalSleeping_.fetch_add(1, std::memory_order_relaxed);
  }

  /**
   * @brief Mark a thread as awake. Called after returning from EpochWaiter sleep.
   *
   * Clears the thread's bit in its group's sleep mask. This handles the
   * timeout-wake case where no waker cleared the bit.
   */
  void exitSleep(int32_t threadIdx) {
    int32_t group = threadIdx / groupSize_;
    int32_t bit = threadIdx % groupSize_;
    groupStates_[static_cast<size_t>(group)].sleepMask.fetch_and(
        ~(uint64_t{1} << bit), std::memory_order_relaxed);
    totalSleeping_.fetch_sub(1, std::memory_order_relaxed);
  }

  /**
   * @brief Atomically try to claim a sleeping thread for waking.
   *
   * @return true if we cleared the bit (we own the wake and should call
   *         bumpAndWake on the thread's waiter). false if the thread was
   *         already awake or claimed by another waker.
   */
  bool tryClaimSleeper(int32_t threadIdx) {
    int32_t group = threadIdx / groupSize_;
    int32_t bit = threadIdx % groupSize_;
    uint64_t bitMask = uint64_t{1} << bit;
    uint64_t prev = groupStates_[static_cast<size_t>(group)].sleepMask.fetch_and(
        ~bitMask, std::memory_order_acq_rel);
    return (prev & bitMask) != 0;
  }

  /**
   * @brief Wake exactly one sleeping thread without cascade.
   *
   * Fast path for schedule() / conditionallyWake(): finds a sleeping thread
   * via the round-robin hint, bumps its waiter. No sleep mask claim (the
   * thread clears its own bit in exitSleep), no budget store. Avoids the
   * fetch_and overhead of tryClaimSleeper for the single-wake case.
   *
   * @return true if a sleeping thread was found and woken.
   */
  DISPENSO_DLL_ACCESS bool wakeOne();

  /**
   * @brief Claim a sleeping thread, wake it, and return its index.
   *
   * Proactive wake: atomically claims the sleeper (clears its bit so
   * subsequent callers see fewer sleepers), bumps its waiter, and returns
   * the thread index. The caller can then push work directly to that
   * thread's ring so it's waiting when the thread wakes.
   *
   * @return Thread index of the woken thread, or -1 if no sleeping threads.
   */
  DISPENSO_DLL_ACCESS int32_t claimAndWakeOne();

  /**
   * @brief Wake threads in range [0, count) that are sleeping.
   *
   * Targeted wake for fork-join: only wakes threads whose rings have work.
   * Skips groups beyond the range. For each sleeping thread in range,
   * clears its bit and bumps its waiter.
   */
  DISPENSO_DLL_ACCESS void wakeRange(int32_t count);

  /**
   * @brief Wake one sleeping thread with the given cascade budget.
   *
   * Scans group sleep masks starting from startGroup (for locality),
   * claims one sleeper, sets its cascade budget, and bumps its waiter.
   * Budget counts cascade actions (futex calls), not threads — each
   * action wakes one subgroup, intentionally over-waking for coverage.
   *
   * @param budget Cascade actions remaining (0 = no cascade, >0 = continue).
   * @param startGroup Group index to start scanning from. If -1, uses
   *                   the round-robin hint for O(1) single-wake performance.
   * @return true if a thread was woken, false if no sleeping threads found.
   */
  DISPENSO_DLL_ACCESS bool wakeOneWithBudget(int32_t budget, int32_t startGroup = -1);

  /**
   * @brief Wake threads using budget-limited cascade.
   *
   * Budget counts cascade actions, not threads woken. Each cascade action
   * issues one futex call that wakes a subgroup (up to kWaiterSubgroupSize
   * threads). The budget is deducted by kWaiterSubgroupSize/2 per action,
   * which mildly over-wakes (each action wakes up to kWaiterSubgroupSize
   * threads but consumes only half that from the budget). This is
   * intentional: over-waking is a no-op (already-awake threads are
   * unaffected), while under-waking causes latency (threads wait for
   * spin timeout).
   *
   * @param n Budget for cascade actions.
   */
  void wakeN(int32_t n) {
    if (n <= 0 || totalSleeping_.load(std::memory_order_relaxed) <= 0) {
      return;
    }
    if (n == 1) {
      wakeOne();
    } else {
      wakeOneWithBudget(n - 1);
    }
  }

  /**
   * @brief Wake all sleeping threads. Used during shutdown.
   */
  DISPENSO_DLL_ACCESS void wakeAll();

  /**
   * @brief Process wake budget. Called at the top of the thread loop and
   * periodically during the spin phase.
   *
   * Fast path is inlined: a single relaxed load that returns 0 when no budget
   * is pending (the common case). The slow path (finding a sleeping thread,
   * distributing budget, issuing futex wake) is out-of-line in the .cpp.
   *
   * @param threadIdx The index of the thread processing the budget.
   * @return The budget that was consumed (0 if no budget was pending).
   */
  int32_t processBudget(int32_t threadIdx) {
    // Precomputed lookup: non-cascade threads have kNoCascade, cascade-team
    // threads have their budget slot index. Avoids division/modulo per call.
    uint16_t slot = budgetSlotTable_[static_cast<size_t>(threadIdx)];
    if (slot == kNoCascade) {
      return 0;
    }
    if (wakeBudgets_[slot].value.load(std::memory_order_relaxed) <= 0) {
      return 0;
    }
    return processBudgetSlow(threadIdx);
  }

  DISPENSO_DLL_ACCESS int32_t processBudgetSlow(int32_t threadIdx);

  /** @brief Total number of threads managed by this wake state. */
  int32_t numThreads() const {
    return numThreads_;
  }
  /** @brief Number of wake groups. */
  int32_t numGroups() const {
    return numGroups_;
  }
  /** @brief Threads per wake group. */
  int32_t groupSize() const {
    return groupSize_;
  }
  /** @brief Maximum children per node in the cascade tree. */
  int32_t branchFactor() const {
    return branchFactor_;
  }
  int32_t totalSleeping() const {
    return totalSleeping_.load(std::memory_order_relaxed);
  }

 private:
  // Bitmask for cascade-team threads within a group's sleep mask.
  static constexpr uint64_t kCascadeTeamMask = (uint64_t{1} << kWaiterSubgroupSize) - 1;
  // Sentinel value: thread is not in the cascade team.
  static constexpr uint16_t kNoCascade = UINT16_MAX;

  // Map a thread index to its cascade budget slot.
  // Thread must be in the cascade team (posInGroup < kWaiterSubgroupSize).
  size_t cascadeBudgetIdx(int32_t threadIdx) const {
    assert(
        budgetSlotTable_[static_cast<size_t>(threadIdx)] != kNoCascade &&
        "cascadeBudgetIdx called for non-cascade thread");
    return static_cast<size_t>(budgetSlotTable_[static_cast<size_t>(threadIdx)]);
  }

  // Distribute remaining cascade budget to the target thread and its peers.
  // Called after successfully claiming a sleeper in processBudgetSlow.
  void distributeBudget(int32_t remaining, int32_t groupIdx, int32_t targetIdx, int32_t callerIdx);

  decltype(detail::makeAlignedArray<WaiterBlock>(0)) waiterBlocks_;
  decltype(detail::makeAlignedArray<GroupWakeState>(0)) groupStates_;

  // Per-cascade-thread wake budgets, cache-line padded to prevent false
  // sharing. Only the first subgroup per group (kWaiterSubgroupSize threads)
  // participates in parallel cascade; the other threads in the group just
  // wake and find work. Total slots = numGroups * kWaiterSubgroupSize.
  struct DISPENSO_CACHELINE_ALIGNED WakeBudget {
    std::atomic<int32_t> value{0};
  };
  decltype(detail::makeAlignedArray<WakeBudget>(0)) wakeBudgets_;

  // Precomputed thread-to-budget-slot mapping. Cascade-team threads
  // (first subgroup of each group) map to their budget slot index.
  // Non-cascade threads map to kNoCascade. Avoids division/modulo
  // in the processBudget fast path.
  std::unique_ptr<uint16_t[]> budgetSlotTable_;

  // Round-robin hint for single-wake: avoids scanning all groups when only
  // one thread needs waking (schedule, Future). Relaxed — no correctness
  // concern if stale, just a perf hint.
  alignas(kCacheLineSize) std::atomic<int32_t> nextWakeGroup_{0};

  // Total sleeping threads. Incremented in enterSleep, decremented in
  // exitSleep (always by the sleeping thread itself — never by claimers).
  // Provides O(1) "no sleepers" fast path for claimAndWakeOne/wakeOne,
  // avoiding the O(numGroups) sleep mask scan during sustained bursts.
  alignas(kCacheLineSize) std::atomic<int32_t> totalSleeping_{0};

  // Precomputed wrap-around table: nextGroupTable_[i] = (i + 1) % numGroups_.
  // Avoids modulo in the hot path.
  std::unique_ptr<int32_t[]> nextGroupTable_;

  int32_t numThreads_;
  int32_t numGroups_;
  int32_t groupSize_;
  int32_t branchFactor_;
};

} // namespace dispenso
