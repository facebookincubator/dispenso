/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_pool_wake.h>

#include <dispenso/detail/math.h>

#include <algorithm>
#include <cassert>

namespace dispenso {

PoolWakeState::PoolWakeState(int32_t numThreads, int32_t groupSize, int32_t branchFactor)
    : numThreads_(numThreads),
      numGroups_((numThreads + groupSize - 1) / groupSize),
      groupSize_(groupSize),
      branchFactor_(branchFactor) {
  assert(numThreads >= 0);
  assert(groupSize > 0 && groupSize <= kMaxThreadsPerWakeGroup);
  assert(branchFactor > 0);

  if (numThreads > 0) {
    waiterBlocks_ = detail::makeAlignedArray<WaiterBlock>(static_cast<size_t>(numGroups_));
    groupStates_ = detail::makeAlignedArray<GroupWakeState>(static_cast<size_t>(numGroups_));
    // Precompute wrap-around table to avoid modulo in hot path.
    nextGroupTable_ = std::make_unique<int32_t[]>(static_cast<size_t>(numGroups_));
    for (int32_t i = 0; i < numGroups_ - 1; ++i) {
      nextGroupTable_[static_cast<size_t>(i)] = i + 1;
    }
    nextGroupTable_[static_cast<size_t>(numGroups_ - 1)] = 0;

    // Build the Pattern C cascade table. Level 1: g0's threads cascade to
    // g1..gG. Level 2: g1..gG's threads cascade to g(G+1)..g(G+G*G).
    // Each thread is at most one cascade host.
    //
    // For 192 threads / G=8 / 24 groups:
    //   threads 0..7  (g0) -> g1..g8        (8 lambdas)
    //   threads 8..15 (g1) -> g9..g16       (8 lambdas)
    //   threads 16..23 (g2) -> g17..g24 (cap at 23)
    //   ... continue until all groups assigned.
    cascadeTargets_.assign(static_cast<size_t>(numThreads_), -1);
    int32_t nextTarget = 1; // g0 is the seed; cascade fills g1..gN-1
    for (int32_t hostGroup = 0; hostGroup < numGroups_ && nextTarget < numGroups_; ++hostGroup) {
      for (int32_t slot = 0; slot < groupSize_ && nextTarget < numGroups_; ++slot) {
        int32_t hostThread = hostGroup * groupSize_ + slot;
        if (hostThread >= numThreads_) {
          break;
        }
        cascadeTargets_[static_cast<size_t>(hostThread)] = nextTarget;
        ++nextTarget;
      }
    }
  }
}

PoolWakeState::~PoolWakeState() = default;

void PoolWakeState::wakeRange(int32_t count) {
  if (count <= 0) {
    return;
  }
  // For each group in [0, count), bump the group's epoch so spinning
  // workers won't park, and additionally issue a futex syscall only if
  // some thread in the group is actually sleeping. The epoch bump alone
  // costs one relaxed atomic; the futex syscall costs ~1us. Skipping
  // the syscall when sleepMask == 0 (entire group spinning) cuts the
  // warm-pool wake cost by ~100x.
  //
  // Correctness: a spinning worker that hasn't yet looped back to the
  // top of its scheduler loop will pick up the scheduled task on its
  // next iteration of tryFindAndExecuteWork. If a worker later transitions
  // into waitOnThread(epoch), the bumped epoch causes wait to return
  // immediately without sleeping.
  int32_t lastGroup = (count - 1) / groupSize_;
  for (int32_t g = 0; g <= lastGroup && g < numGroups_; ++g) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (g == lastGroup) {
      int32_t bitsInLastGroup = count - g * groupSize_;
      if (bitsInLastGroup < 64) {
        mask &= (uint64_t{1} << bitsInLastGroup) - 1;
      }
    }
    auto& waiter = waiterFor(g * groupSize_);
    if (mask == 0) {
      // All workers in this group are spinning — epoch bump prevents
      // them from parking on their next waitOnThread without paying
      // for a syscall.
      waiter.bump();
    } else {
      // At least one worker is parked — need a real wake. Wake just
      // the parked ones (bumpAndWakeN counts).
      int32_t numSleepers = detail::countSetBits(mask);
      waiter.bumpAndWakeN(numSleepers, groupSize_);
    }
  }
}

int32_t PoolWakeState::claimAndWakeOne() {
  if (totalSleeping_.load(std::memory_order_relaxed) <= 0) {
    return -1;
  }
  int32_t g = nextWakeGroup_.load(std::memory_order_relaxed);
  if (g >= numGroups_) {
    g = 0;
  }
  for (int32_t gi = 0; gi < numGroups_; ++gi) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    while (mask) {
      int bit = detail::countTrailingZeros(mask);
      int32_t threadIdx = g * groupSize_ + bit;
      if (threadIdx < numThreads_ && tryClaimSleeper(threadIdx)) {
        // Wake one thread from this group's shared waiter (the claimed
        // thread's bit is cleared so other callers see one fewer sleeper).
        waiterFor(threadIdx).bumpAndWake();
        nextWakeGroup_.store(nextGroupTable_[static_cast<size_t>(g)], std::memory_order_relaxed);
        return threadIdx;
      }
      mask &= mask - 1;
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }
  return -1;
}

bool PoolWakeState::cascadeWakeSeed(int32_t count) {
  if (count <= 0) {
    return false;
  }
  int32_t lastGroup = (count - 1) / groupSize_;
  if (lastGroup >= numGroups_) {
    lastGroup = numGroups_ - 1;
  }

  // Pool-wide fast path: no sleepers anywhere. Just bump every affected
  // group's epoch (so any racing parker sees the bump and skips sleep) and
  // return. Zero syscalls, ~N atomic stores. This matches the warm-pool
  // path of the original wakeRange.
  if (totalSleeping_.load(std::memory_order_relaxed) == 0) {
    for (int32_t g = 0; g <= lastGroup; ++g) {
      waiterFor(g * groupSize_).bump();
    }
    return false;
  }

  // Single pass: bump every affected group's epoch and wake any sleepers.
  // The ring-dispatch path pre-stages cascade-host lambdas into g0 rings
  // that call cascadeWake() for g1+, but the central-queue path has no
  // such lambdas. Wake all groups with sleepers directly so both paths
  // get prompt futex wakes. The extra syscalls are negligible — they only
  // fire when threads are actually sleeping.
  for (int32_t g = 0; g <= lastGroup; ++g) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (g == lastGroup) {
      int32_t bitsInLastGroup = count - g * groupSize_;
      if (bitsInLastGroup < 64) {
        mask &= (uint64_t{1} << bitsInLastGroup) - 1;
      }
    }
    auto& waiter = waiterFor(g * groupSize_);
    if (mask == 0) {
      waiter.bump();
    } else {
      int32_t numSleepers = detail::countSetBits(mask);
      waiter.bumpAndWakeN(numSleepers, groupSize_);
    }
  }
  return true;
}

void PoolWakeState::wakeAll() {
  // Bump every group's epoch and wake groups with sleepers. The epoch bump
  // is needed even for groups with sleepMask == 0: a thread could be past
  // its data.running() check but before enterSleep() (which sets the bit).
  // Without the bump, such a thread enters waitFor with a stale epoch and
  // blocks until timeout — causing slow shutdown.
  for (int32_t g = 0; g < numGroups_; ++g) {
    if (groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed)) {
      waiterFor(g * groupSize_).bumpAndWakeAll();
    } else {
      waiterFor(g * groupSize_).bump();
    }
  }
}

} // namespace dispenso
