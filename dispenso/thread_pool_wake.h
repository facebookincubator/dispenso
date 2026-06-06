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
#include <vector>

#include <dispenso/detail/epoch_waiter.h>
#include <dispenso/detail/math.h>
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
 * @brief Per-group state.
 *
 * `sleepMask`: bit i set when thread i in the group is sleeping on the
 * group's EpochWaiter. Wakers may atomically test-and-clear bits to claim
 * exclusive wake responsibility (e.g. for targeted wakes via
 * `claimAndWakeOne`).
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

// One EpochWaiter (futex address) per group: all threads in a group share
// it. A single futex_wake(addr, K) wakes K arbitrary threads from the
// group; futex_wake(addr, 1) wakes one. Bulk wakes for K > groupSize fan
// out across groups via the Pattern C cascade — workers in the seed group
// each carry a pre-staged lambda that wakes one target group; level-2
// cascade hosts in those groups wake the remaining groups in parallel.
struct DISPENSO_CACHELINE_ALIGNED WaiterBlock {
  detail::EpochWaiter waiter;
};

/**
 * @class PoolWakeState
 * @brief Wake infrastructure for a ThreadPool.
 *
 * Manages per-thread EpochWaiters, per-group sleep masks, and a static
 * 1-seed 2-level cascade table for fan-out wakes. Used by ThreadPool for
 * both fork-join (per-thread ring dispatch via `cascadeWakeSeed` + cascade
 * lambdas) and central queue (also via `cascadeWakeSeed`) work dispatch.
 *
 * ## Thread Safety
 *
 * - `enterSleep` / `exitSleep`: called only by the owning thread
 * - `tryClaimSleeper`: safe to call from any thread (atomic test-and-clear)
 * - `claimAndWakeOne` / `wakeRange` / `cascadeWakeSeed` / `wakeAll`:
 *   safe to call from any thread
 * - `cascadeWake`: called only from cascade-host lambdas (one per cascade
 *   slot per dispatch)
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
#else
      // Default G=8 across platforms based on the Linux tuning sweep
      // (smaller groups reduce steal-ring + group-futex contention, with
      // promote-seed cascade absorbing the extra cascade hop). Expected to
      // be reasonable on macOS/Windows but not yet validated there — see
      // docs/design/wake_tuning.md for the per-platform tuning process.
      int32_t groupSize = 8,
#endif
      int32_t branchFactor = kDefaultWakeBranchFactor);

  DISPENSO_DLL_ACCESS ~PoolWakeState();

  // Non-copyable, non-movable
  PoolWakeState(const PoolWakeState&) = delete;
  PoolWakeState& operator=(const PoolWakeState&) = delete;

  /**
   * @brief Get the EpochWaiter for a specific thread's group. All threads
   * in a group share one EpochWaiter (one futex address); bumpAndWakeN on
   * this waiter wakes K arbitrary threads from the group with one syscall.
   */
  detail::EpochWaiter& waiterFor(int32_t threadIdx) {
    return waiterBlocks_[static_cast<size_t>(threadIdx / groupSize_)].waiter;
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
   * Serial fork-join wake: for each affected group, bumps the epoch and
   * issues a futex syscall iff sleepers exist in that group. O(numGroups)
   * syscalls in the worst case; superseded by `cascadeWakeSeed` for the
   * default scheduleBulkToRings path. Kept as a fallback when cascade is
   * disabled at compile time (`-DDISPENSO_DISABLE_CASCADE_WAKERANGE`).
   */
  DISPENSO_DLL_ACCESS void wakeRange(int32_t count);

  /**
   * @brief Wake all sleeping threads. Used during shutdown.
   */
  DISPENSO_DLL_ACCESS void wakeAll();

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

  // ---- Pattern C cascade for scheduleBulkToRings ----
  //
  // 1-seed 2-level static cascade tree:
  //   Producer wakes seed group (g0) with one bumpAndWakeN.
  //   g0's threads each have a cascade-host slot; on wake they execute a
  //   lambda that wakes one level-2 target group, then runs their own user
  //   work.
  //   level-2 groups' threads each have cascade-host slots that wake one
  //   level-3 target group.
  //
  // Tree shape for N threads (groupSize=G):
  //   level 1: 1 group  (g0)                covers G threads
  //   level 2: G groups                     covers G*G threads
  //   level 3: G*G groups (when needed)     covers G*G*G threads
  //
  // 192 threads / G=8 = 24 groups: levels 1+2 cover 1+8 = 9 groups (72
  // threads). Need level 3 for the remaining 15 groups.

  // Returns the level-2/3 target group for a cascade-host thread, or -1 if
  // this thread is not a cascade host OR if the target group lies outside
  // the wake range [0, count). The count gate matches scheduleBulkToRings's
  // "wake only threads with work" contract: for partial-pool dispatch we
  // don't want cascade hosts firing wakes into empty groups.
  int32_t cascadeTargetFor(int32_t threadIdx, int32_t count) const {
    if (threadIdx < 0 || static_cast<size_t>(threadIdx) >= cascadeTargets_.size()) {
      return -1;
    }
    int32_t target = cascadeTargets_[static_cast<size_t>(threadIdx)];
    int32_t lastGroup = (count - 1) / groupSize_;
    return (target <= lastGroup) ? target : -1;
  }

  // Wakes one target group: bumps the group's epoch and issues bumpAndWakeN
  // only if the sleepMask is non-zero. Called by cascade-host lambdas.
  void cascadeWake(int32_t targetGroup) {
    uint64_t mask =
        groupStates_[static_cast<size_t>(targetGroup)].sleepMask.load(std::memory_order_relaxed);
    auto& waiter = waiterFor(targetGroup * groupSize_);
    if (mask == 0) {
      waiter.bump();
    } else {
      int32_t numSleepers = detail::countSetBits(mask);
      waiter.bumpAndWakeN(numSleepers, groupSize_);
    }
  }

  // Single-pass wake for scheduleBulkToRings. Bumps every affected group's
  // epoch and wakes seed group g0 if any sleepers exist in [0, count).
  // Returns true if the cold path was taken (sleepers found).
  DISPENSO_DLL_ACCESS bool cascadeWakeSeed(int32_t count);

 private:
  decltype(detail::makeAlignedArray<WaiterBlock>(0)) waiterBlocks_;
  decltype(detail::makeAlignedArray<GroupWakeState>(0)) groupStates_;

  // Round-robin hint for single-wake: avoids scanning all groups when only
  // one thread needs waking (schedule, Future). Relaxed — no correctness
  // concern if stale, just a perf hint.
  alignas(kCacheLineSize) std::atomic<int32_t> nextWakeGroup_{0};

  // Total sleeping threads. Incremented in enterSleep, decremented in
  // exitSleep (always by the sleeping thread itself — never by claimers).
  // Provides O(1) "no sleepers" fast path for claimAndWakeOne /
  // cascadeWakeSeed, avoiding the O(numGroups) sleep mask scan during
  // sustained bursts.
  alignas(kCacheLineSize) std::atomic<int32_t> totalSleeping_{0};

  // Precomputed wrap-around table: nextGroupTable_[i] = (i + 1) % numGroups_.
  // Avoids modulo in the hot path.
  std::unique_ptr<int32_t[]> nextGroupTable_;

  // Pattern C cascade: cascadeTargets_[threadIdx] = group this thread cascades
  // to on wake, or -1 if not a cascade host. Built at construction.
  std::vector<int32_t> cascadeTargets_;

  int32_t numThreads_;
  int32_t numGroups_;
  int32_t groupSize_;
  int32_t branchFactor_;
};

} // namespace dispenso
