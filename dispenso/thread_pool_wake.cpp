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
        // Wake one thread from this group's shared waiter. The kernel picks
        // any one of the parked waiters; that's fine since group threads
        // share a steal ring and any of them can grab the work.
        waiterFor(threadIdx).bumpAndWake();
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
  // Only scan groups that contain threads in [0, count). With one waiter
  // per group, issue ONE bump-and-wake per group sized to the number of
  // sleepers in range — avoids the redundant per-bit wake the old
  // sg=4 layout needed.
  int32_t lastGroup = (count - 1) / groupSize_;
  for (int32_t g = 0; g <= lastGroup && g < numGroups_; ++g) {
    uint64_t mask = groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed);
    if (!mask) {
      continue;
    }
    if (g == lastGroup) {
      int32_t bitsInLastGroup = count - g * groupSize_;
      if (bitsInLastGroup < 64) {
        mask &= (uint64_t{1} << bitsInLastGroup) - 1;
      }
    }
    if (!mask) {
      continue;
    }
    int32_t numSleepers = detail::countSetBits(mask);
    waiterFor(g * groupSize_).bumpAndWakeN(numSleepers, groupSize_);
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

// Find a group with sleepers, starting from `from`, walking via
// nextGroupTable_. Returns -1 if none.
int32_t PoolWakeState::findGroupWithSleepers(int32_t from) const {
  int32_t g = from;
  for (int32_t gi = 0; gi < numGroups_; ++gi) {
    if (groupStates_[static_cast<size_t>(g)].sleepMask.load(std::memory_order_relaxed)) {
      return g;
    }
    g = nextGroupTable_[static_cast<size_t>(g)];
  }
  return -1;
}

void PoolWakeState::setupCascade(int32_t g0, int32_t remaining) {
  int32_t numToCascade = (remaining + groupSize_ - 1) / groupSize_;

#ifndef DISPENSO_TUNE_PROMOTE_SEED
// On Windows, promote-seed is a small net loss (~1.5% mean across the
// per-bench tuning sweep, with the clearest losses on
// BM_dispenso_blocking/*). Each WakeByAddressSingle is expensive enough
// that the extra in-thread wake to bootstrap g1 doesn't pay for itself
// against a single seed cascading from g0. On Linux/macOS the second
// seed runs in parallel with g0's cascade for free, so keep it on.
#if defined(_WIN32)
#define DISPENSO_TUNE_PROMOTE_SEED 0
#else
#define DISPENSO_TUNE_PROMOTE_SEED 1
#endif
#endif
  int32_t g1 = -1;
#if DISPENSO_TUNE_PROMOTE_SEED
  if (numToCascade > groupSize_) {
    g1 = findGroupWithSleepers(nextGroupTable_[static_cast<size_t>(g0)]);
    if (g1 == g0) {
      g1 = -1;
    }
  }
#endif

  uint64_t bits0 = 0, bits1 = 0;
  int32_t scan = nextGroupTable_[static_cast<size_t>(g0)];
  int32_t bitIdx = 0;
  for (int32_t gi = 0; gi < numGroups_ - 1 && numToCascade > 0; ++gi) {
    if (scan == g1) {
      scan = nextGroupTable_[static_cast<size_t>(scan)];
      continue;
    }
    if (scan < 64 &&
        groupStates_[static_cast<size_t>(scan)].sleepMask.load(std::memory_order_relaxed)) {
      if (g1 >= 0 && (bitIdx & 1)) {
        bits1 |= (uint64_t{1} << scan);
      } else {
        bits0 |= (uint64_t{1} << scan);
      }
      --numToCascade;
      ++bitIdx;
    }
    scan = nextGroupTable_[static_cast<size_t>(scan)];
  }
  if (bits0 != 0) {
    groupStates_[static_cast<size_t>(g0)].wakePending.fetch_or(bits0, std::memory_order_release);
  }
  if (g1 >= 0) {
    if (bits1 != 0) {
      groupStates_[static_cast<size_t>(g1)].wakePending.fetch_or(bits1, std::memory_order_release);
    }
    groupStates_[static_cast<size_t>(g0)].promoteSeed.store(g1, std::memory_order_release);
  }
}

void PoolWakeState::wakeN(int32_t n) {
  // Relaxed loads on totalSleeping_/sleepMask are intentional: a missed
  // sleeper here is a bounded latency hit (the sleeper wakes via the
  // sleepLengthUs_ timeout and re-checks for work), not a deadlock.
  if (n <= 0 || totalSleeping_.load(std::memory_order_relaxed) <= 0) {
    return;
  }
  if (n == 1) {
    wakeOne();
    return;
  }

  int32_t startGroup = nextWakeGroup_.load(std::memory_order_relaxed);
  if (startGroup >= numGroups_) {
    startGroup = 0;
  }
  int32_t g0 = findGroupWithSleepers(startGroup);
  if (g0 < 0) {
    return;
  }

  // K0 = up to groupSize threads from g0; remainder wakes other groups via
  // the bitmask cascade.
  int32_t k0 = std::min(n, groupSize_);
  int32_t remaining = n - k0;

  if (remaining > 0) {
    setupCascade(g0, remaining);
  }

  // Broadcast to g0. Workers in g0 will see promoteSeed (if set) and
  // wakePending bits (if any) and CAS-claim them in cascadeStepSlow.
  waiterFor(g0 * groupSize_).bumpAndWakeN(k0, groupSize_);
  nextWakeGroup_.store(nextGroupTable_[static_cast<size_t>(g0)], std::memory_order_relaxed);
}

int32_t PoolWakeState::cascadeStepSlow(int32_t group) {
  auto& gs = groupStates_[static_cast<size_t>(group)];

  // First, try to claim the promoteSeed (one thread wins, others fall through).
  // If we win, bumpAndWake the promote target to bootstrap parallel cascade.
  // We then continue to claim a wakePending bit so this thread also fans out.
  //
  // The relaxed load is safe: this thread just returned from EpochWaiter::wait,
  // which provides a happens-after edge with the producer's bumpAndWakeN, which
  // in turn happens-after the producer's release-store of promoteSeed. The
  // exchange's acquire ordering guarantees we observe g1.wakePending bits if
  // we successfully claim — relaxed-then-acquire avoids a redundant fence on
  // the common no-promote path.
  //
  // We use bumpAndWakeAll here rather than bumpAndWakeN(popcount(bits1)+1, ...)
  // for simplicity. The trade-off: when g1 has more parked threads than bits1
  // cascade work, the extra threads wake, find no work, and re-sleep (spurious
  // wake CPU bounded by spurious_round_trip * (groupSize - popcount(bits1)),
  // running in parallel on otherwise-idle cores). Acceptable for the rare case
  // promote-seed engages; revisit if measurements show it dominates.
  int32_t promote = gs.promoteSeed.load(std::memory_order_relaxed);
  if (promote >= 0) {
    int32_t claimed = gs.promoteSeed.exchange(-1, std::memory_order_acquire);
    if (claimed >= 0) {
      waiterFor(claimed * groupSize_).bumpAndWakeAll();
    }
  }

  uint64_t pending = gs.wakePending.load(std::memory_order_acquire);
  while (pending != 0) {
    int target = detail::countTrailingZeros(pending);
    uint64_t bit = uint64_t{1} << target;
    uint64_t prev = gs.wakePending.fetch_and(~bit, std::memory_order_acq_rel);
    if (prev & bit) {
      // We claimed this bit — wake the target group fully.
      waiterFor(target * groupSize_).bumpAndWakeAll();
      return target;
    }
    // Lost the race; another worker took this bit. Re-load to pick up
    // any new bits that were set between our previous load and now.
    pending = gs.wakePending.load(std::memory_order_acquire);
  }
  return -1;
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
