/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file distributed_rw_lock_impl.h
 * A BRAVO-inspired distributed reader/writer lock implementation.
 *
 * N cache-line-aligned sub-locks — readers pick one by index (no cross-core
 * contention), writers iterate all N. Read path is ~1 atomic on a cache-local
 * line. Write path (rare) acquires all N slots.
 **/

#pragma once

#include <dispenso/detail/rw_lock_impl.h>

namespace dispenso {
namespace detail {

/**
 * A distributed reader/writer lock composed of N cache-line-aligned sub-locks.
 *
 * The read path is intrusive: callers provide an index to select their sub-lock,
 * achieving near-zero contention when different threads use different indices.
 *
 * The write path iterates all N sub-locks in two phases:
 * 1. Set the writer bit on every slot (blocks new readers)
 * 2. Wait for all existing readers to drain on every slot
 *
 * @tparam N Number of sub-locks. Must be a power of 2. Default 128.
 **/
template <size_t N = 128>
class DistributedRWLockImpl {
  static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

 public:
  static constexpr size_t kNumSlots = N;
  static constexpr size_t kMask = N - 1;

  /**
   * Acquire shared (read) access on the sub-lock at `index & kMask`.
   **/
  void lock_shared(size_t index) {
    slots_[index & kMask].lock_shared();
  }

  /**
   * Release shared (read) access on the sub-lock at `index & kMask`.
   * Must use the same index that was passed to lock_shared().
   **/
  void unlock_shared(size_t index) {
    slots_[index & kMask].unlock_shared();
  }

  /**
   * Try to acquire shared (read) access on the sub-lock at `index & kMask`.
   * @return true if the lock was acquired, false if a writer holds the slot.
   **/
  bool try_lock_shared(size_t index) {
    return slots_[index & kMask].try_lock_shared();
  }

  /**
   * Acquire exclusive (write) access on ALL sub-locks.
   * Two-phase protocol: set writer bit on all slots, then drain readers on all.
   **/
  void lock() {
    for (size_t i = 0; i < N; ++i) {
      slots_[i].setWriteBit();
    }
    for (size_t i = 0; i < N; ++i) {
      slots_[i].waitForReaderDrain();
    }
  }

  /**
   * Try to acquire exclusive (write) access on ALL sub-locks.
   *
   * @note On success, this method blocks until all readers across all slots
   * have drained — writers cannot proceed while readers are active by
   * definition. The "try" semantics apply only to the writer-bit acquisition
   * phase: if any slot is currently held by another writer, this returns
   * false without blocking. Once all writer bits are claimed, the success
   * path waits for reader drain just like lock().
   *
   * @return true if all writer bits were acquired (and readers have drained),
   *         false if any slot was held by another writer (rolls back).
   **/
  bool try_lock() {
    for (size_t i = 0; i < N; ++i) {
      // Phase 1 is writer-bit-only: tryWriteBit() claims the bit ignoring active
      // readers (we drain them below), failing only against another writer. The
      // single-slot RWLockImpl::try_lock() would instead roll back on readers,
      // which is wrong for this two-phase protocol.
      if (!slots_[i].tryWriteBit()) {
        // Roll back: unlock all slots we've already acquired
        for (size_t j = 0; j < i; ++j) {
          slots_[j].unlock();
        }
        return false;
      }
    }
    // All writer bits set and no other writers. Now drain readers.
    for (size_t i = 0; i < N; ++i) {
      slots_[i].waitForReaderDrain();
    }
    return true;
  }

  /**
   * Release exclusive (write) access on ALL sub-locks.
   **/
  void unlock() {
    for (size_t i = 0; i < N; ++i) {
      slots_[i].unlock();
    }
  }

 private:
  // Expose the protected two-phase methods from RWLockImpl via a derived Slot.
  struct alignas(kCacheLineSize) Slot : public RWLockImpl {
    using RWLockImpl::setWriteBit;
    using RWLockImpl::tryWriteBit;
    using RWLockImpl::waitForReaderDrain;
  };

  Slot slots_[N];
};

} // namespace detail
} // namespace dispenso
