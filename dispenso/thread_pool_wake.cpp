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
    // Cascade budgets: one per cascade-team thread (first subgroup per group).
    size_t numBudgets = static_cast<size_t>(numGroups_) * kWaiterSubgroupSize;
    wakeBudgets_ = detail::makeAlignedArray<WakeBudget>(numBudgets);
    // Precompute thread-to-budget-slot table.
    budgetSlotTable_ = std::make_unique<uint16_t[]>(static_cast<size_t>(numThreads));
    for (int32_t i = 0; i < numThreads; ++i) {
      int32_t posInGroup = i % groupSize;
      if (posInGroup < kWaiterSubgroupSize) {
        int32_t group = i / groupSize;
        budgetSlotTable_[static_cast<size_t>(i)] =
            static_cast<uint16_t>(group * kWaiterSubgroupSize + posInGroup);
      } else {
        budgetSlotTable_[static_cast<size_t>(i)] = kNoCascade;
      }
    }
    // Precompute wrap-around table to avoid modulo in hot path
    nextGroupTable_ = std::make_unique<int32_t[]>(static_cast<size_t>(numGroups_));
    for (int32_t i = 0; i < numGroups_ - 1; ++i) {
      nextGroupTable_[static_cast<size_t>(i)] = i + 1;
    }
    nextGroupTable_[static_cast<size_t>(numGroups_ - 1)] = 0;
  }
}

PoolWakeState::~PoolWakeState() = default;

bool PoolWakeState::wakeOne() {
  if (totalSleeping_.load(std::memory_order_relaxed) <= 0) {
    return false;
  }
  int32_t g = nextWakeGroup_.load(std::memory_order_relaxed);
  if (g >= numGroups_) {
    g = 0;
  }
  for (int32_t gi = 0; gi < numGroups_; ++gi) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (mask) {
      int bit = detail::countTrailingZeros(mask);
      int32_t threadIdx = g * groupSize_ + bit;
      if (threadIdx < numThreads_) {
        // No claim needed — just bump the waiter. The thread clears its own
        // bit in exitSleep. Wake the full subgroup — conditionallyWake may be
        // called when multiple tasks are pending.
        waiterFor(threadIdx).bumpAndWakeAll();
        nextWakeGroup_.store(nextGroupTable_[static_cast<size_t>(g)], std::memory_order_relaxed);
        return true;
      }
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }
  return false;
}

void PoolWakeState::wakeRange(int32_t count) {
  if (count <= 0) {
    return;
  }
  // Only scan groups that contain threads in [0, count).
  int32_t lastGroup = (count - 1) / groupSize_;
  for (int32_t g = 0; g <= lastGroup && g < numGroups_; ++g) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (!mask) {
      continue;
    }
    // Mask off bits beyond our range in the last group
    if (g == lastGroup) {
      int32_t bitsInLastGroup = count - g * groupSize_;
      if (bitsInLastGroup < 64) {
        mask &= (uint64_t{1} << bitsInLastGroup) - 1;
      }
    }
    // Wake all sleeping threads in range within this group
    while (mask) {
      int bit = detail::countTrailingZeros(mask);
      int32_t threadIdx = g * groupSize_ + bit;
      if (tryClaimSleeper(threadIdx)) {
        waiterFor(threadIdx).bumpAndWakeAll();
      }
      mask &= mask - 1;
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
        // Claimed: bit is cleared, so other callers won't try to wake this thread.
        waiterFor(threadIdx).bumpAndWakeAll();
        nextWakeGroup_.store(nextGroupTable_[static_cast<size_t>(g)], std::memory_order_relaxed);
        return threadIdx;
      }
      mask &= mask - 1;
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }
  return -1;
}

bool PoolWakeState::wakeOneWithBudget(int32_t budget, int32_t startGroup) {
  if (totalSleeping_.load(std::memory_order_relaxed) <= 0) {
    return false;
  }
  // Use round-robin hint when no specific group is requested.
  if (startGroup < 0) {
    startGroup = nextWakeGroup_.load(std::memory_order_relaxed);
    if (startGroup >= numGroups_) {
      startGroup = 0;
    }
  }
  int32_t g = startGroup;
  for (int32_t gi = 0; gi < numGroups_; ++gi) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (mask) {
      // Prefer cascade-team threads (first subgroup, bits 0..kWaiterSubgroupSize-1).
      // These threads process budgets and can cascade further.
      uint64_t cascadeMask = mask & kCascadeTeamMask;
      if (!cascadeMask) {
        cascadeMask = mask; // No cascade-team threads sleeping; use any thread.
      }
      while (cascadeMask) {
        int bit = detail::countTrailingZeros(cascadeMask);
        int32_t threadIdx = g * groupSize_ + bit;
        if (threadIdx < numThreads_ && tryClaimSleeper(threadIdx)) {
          if (budget > 0) {
            // Assign budget to the target's cascade-team position.
            // If the target IS in the cascade team, this is its own slot.
            // If not, assign to position 0 of the group (it will be woken
            // separately or find the budget on its next spin-phase check).
            size_t budgetIdx = cascadeBudgetIdx(
                bit < kWaiterSubgroupSize ? threadIdx : g * groupSize_); // group leader
            wakeBudgets_[budgetIdx].value.store(budget, std::memory_order_release);
          }
          waiterFor(threadIdx).bumpAndWakeAll();
          nextWakeGroup_.store(nextGroupTable_[static_cast<size_t>(g)], std::memory_order_relaxed);
          return true;
        }
        cascadeMask &= cascadeMask - 1;
      }
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }
  return false;
}

void PoolWakeState::wakeAll() {
  // Bump every group's epoch unconditionally. A thread could be past its
  // data.running() check but before enterSleep() (sleepMask bit not yet set).
  // Without the epoch bump, such a thread enters waitFor with a stale epoch
  // and blocks until timeout — causing slow shutdown.
  for (int32_t g = 0; g < numGroups_; ++g) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    while (mask) {
      int bit = detail::countTrailingZeros(mask);
      int32_t threadIdx = g * groupSize_ + bit;
      if (threadIdx < numThreads_) {
        tryClaimSleeper(threadIdx);
      }
      mask &= mask - 1;
    }
    int32_t threadsInGroup = std::min(groupSize_, numThreads_ - g * groupSize_);
    int32_t numSubWaiters = (threadsInGroup + kWaiterSubgroupSize - 1) / kWaiterSubgroupSize;
    for (int32_t s = 0; s < numSubWaiters; ++s) {
      waiterFor(g * groupSize_ + s * kWaiterSubgroupSize).bumpAndWakeAll();
    }
  }
}

int32_t PoolWakeState::processBudgetSlow(int32_t threadIdx) {
  // Called only when the inline fast path detected a non-zero budget.
  // Exchange atomically claims the budget (another thread may race us).
  size_t myBudgetIdx = cascadeBudgetIdx(threadIdx);
  int32_t budget = wakeBudgets_[myBudgetIdx].value.exchange(0, std::memory_order_acquire);
  if (budget <= 0) {
    return 0;
  }

  // Parallel cascade with leader team: find a sleeping thread in another
  // group's cascade team (first subgroup). Distribute budget across the
  // target's cascade-team peers so they fan out in parallel.
  //
  // Budget counts cascade actions, not threads. Each action issues one
  // futex call that wakes a subgroup (up to kWaiterSubgroupSize threads).
  // Budget is deducted by kWaiterSubgroupSize/2 per action, mildly
  // over-waking for spatial coverage. Over-waking is cheap (no-op for
  // already-awake threads), while under-waking causes latency.
  // O(log_kWaiterSubgroupSize(N)) wake latency.

  int32_t myGroup = threadIdx / groupSize_;

  // Scan other groups first (breadth-first for NUMA distribution), then own.
  int32_t g = nextGroupTable_[static_cast<size_t>(myGroup)];
  for (int32_t gi = 0; gi < numGroups_; ++gi) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    // Prefer cascade-team threads (first subgroup bits).
    uint64_t cascadeMask = mask & kCascadeTeamMask;
    if (!cascadeMask) {
      cascadeMask = mask; // Fallback to any sleeping thread.
    }
    while (cascadeMask) {
      int bit = detail::countTrailingZeros(cascadeMask);
      int32_t targetIdx = g * groupSize_ + bit;
      if (targetIdx < numThreads_ && tryClaimSleeper(targetIdx)) {
        // Each futex call wakes up to kWaiterSubgroupSize threads. Deduct
        // half the subgroup size as a balanced estimate of actual wakes.
        static constexpr int32_t kBudgetDeduction = kWaiterSubgroupSize / 2;
        int32_t remaining = budget - kBudgetDeduction;

        if (remaining > 0) {
          distributeBudget(remaining, g, targetIdx, threadIdx);
        }

        waiterFor(targetIdx).bumpAndWakeAll();
        return budget;
      }
      cascadeMask &= cascadeMask - 1;
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }

  return 0;
}

void PoolWakeState::distributeBudget(
    int32_t remaining,
    int32_t groupIdx,
    int32_t targetIdx,
    int32_t callerIdx) {
  int32_t targetBit = targetIdx % groupSize_;
  if (targetBit < kWaiterSubgroupSize) {
    // Target is in the cascade team. Distribute budget across its
    // cascade-team peers (they all wake from one futex call).
    int32_t cascadeBase = groupIdx * groupSize_;
    int32_t numRecipients = 1;
    int32_t peerSlots[kWaiterSubgroupSize - 1];
    int32_t numPeers = 0;
    for (int32_t p = 0; p < kWaiterSubgroupSize; ++p) {
      int32_t peerIdx = cascadeBase + p;
      if (peerIdx < numThreads_ && peerIdx != targetIdx && peerIdx != callerIdx) {
        peerSlots[numPeers++] = static_cast<int32_t>(cascadeBudgetIdx(peerIdx));
        ++numRecipients;
      }
    }
    int32_t perPeer = numRecipients > 1 ? remaining / numRecipients : 0;
    int32_t assigned = 0;
    for (int32_t p = 0; p < numPeers; ++p) {
      wakeBudgets_[static_cast<size_t>(peerSlots[p])].value.store(
          perPeer, std::memory_order_release);
      assigned += perPeer;
    }
    wakeBudgets_[cascadeBudgetIdx(targetIdx)].value.store(
        remaining - assigned, std::memory_order_release);
  } else {
    // Target is NOT in the cascade team. Give full budget to group leader.
    wakeBudgets_[cascadeBudgetIdx(groupIdx * groupSize_)].value.store(
        remaining, std::memory_order_release);
  }
}

} // namespace dispenso
