/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>

#include <dispenso/detail/completion_event_impl.h>
#include <dispenso/platform.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {
namespace detail {
class RWLockImpl {
 public:
  /**
   * Locks for write access
   *
   * @note It is undefined behavior to recursively lock
   **/
  void lock();

  /**
   * Tries to lock for write access, returns if unable to lock
   *
   * @return true if lock was acquired, false otherwise
   **/
  bool try_lock();

  /**
   * Unlocks write access
   *
   * @note Must already be locked by the current thread of execution, otherwise, the behavior is
   * undefined.
   **/
  void unlock();

  /**
   * Locks for read access
   *
   * @note It is undefined behavior to recursively lock
   **/
  void lock_shared();

  /**
   * Tries to lock for read access, returns if unable to lock
   *
   * @return true if lock was acquired, false otherwise
   *
   * @note It is undefined behavior to recursively lock
   **/
  bool try_lock_shared();

  /**
   * Unlocks read access
   *
   * @note Must already be locked by the current thread of execution, otherwise, the behavior is
   * undefined.
   **/
  void unlock_shared();

  /**
   * Upgrade from a reader lock to a writer lock.  lock_upgrade is a power-user interface.  There is
   * a very good reason why it is not exposed as upgrade_mutex in the standard.  To use it safely,
   * you *MUST* ensure only one thread can try to lock for write concurrently.  If that cannot be
   * guaranteed, you should unlock for read, and lock for write instead of using lock_upgrade to
   * avoid potential deadlock.
   *
   * @note Calling this if the writer lock is already held, or if no reader lock is already held is
   * undefined behavior.
   **/
  void lock_upgrade();

  /**
   * Downgrade the lock from a writer lock to a reader lock.
   *
   * @note Calling this if the writer lock is not held results in undefined behavior
   **/
  void lock_downgrade();

 protected:
  static constexpr int kWriteBit = std::numeric_limits<int>::min();
  static constexpr int kReaderBits = std::numeric_limits<int>::max();

  // Number of pause-spins try_lock() gives active readers to drain before it
  // gives up, rolls back its writer bit, and reports failure. Kept small so
  // try_lock() stays effectively non-blocking.
  static constexpr int kTryLockDrainSpins = 16;

  /**
   * Try to claim the writer bit without blocking, failing only if another writer
   * already holds it. Readers are intentionally ignored: on success the caller
   * owns the bit but readers may still be active, so the caller is responsible
   * for draining them (e.g. DistributedRWLockImpl's two-phase try_lock). This is
   * NOT a standalone exclusive acquire -- use try_lock() for that.
   *
   * @return true if this call set the writer bit, false if another writer held it.
   **/
  bool tryWriteBit();

  /**
   * Phase 1 of write lock: claim the writer bit, spinning if another writer holds it.
   * After return, this writer owns the bit but readers may still be active.
   **/
  void setWriteBit();

  /**
   * Phase 2 of write lock: wait for all readers to drain.
   * Uses OS-level waiting (futex/WaitOnAddress/condvar) to avoid burning CPU
   * while readers complete.  Must be called after setWriteBit() has returned.
   **/
  void waitForReaderDrain();

 private:
  CompletionEventImpl event_{0};

  std::atomic<int>& lockWord() {
    return event_.intrusiveStatus();
  }

  // Decrement reader count and wake a waiting writer if we were the last reader.
  void readerRelease() {
    // Publish a happens-before edge for TSAN. A writer draining readers observes
    // this release through CompletionEventImpl::wait(), which on macOS loads the
    // lock word as an 8-byte atomic while this release is a 4-byte atomic on the
    // same address (the combined_/parts_ union). The handoff is correct on
    // hardware (coherent same-address RMWs + acquire/release), but TSAN tracks
    // release shadows per (address, size) and cannot bridge the 4-byte->8-byte
    // edge, so it would falsely flag data guarded by this reader. Pair with the
    // HAPPENS_AFTER in waitForReaderDrain(). (No-op unless built under TSAN; on
    // Linux the union is same-size and this is simply belt-and-suspenders.)
    DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(&event_);
    int prev = lockWord().fetch_sub(1, std::memory_order_acq_rel);
    if (prev == (kWriteBit | 1)) {
      event_.tryNotify();
    }
  }
};

inline void RWLockImpl::setWriteBit() {
  int val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  }
}

inline void RWLockImpl::waitForReaderDrain() {
  event_.wait(kWriteBit);
  // Receive the happens-before edges published by each drained reader's
  // readerRelease(). See readerRelease() for why the explicit annotation is
  // needed (mixed-size atomic access defeats TSAN's per-(address, size) HB
  // tracking on the macOS CompletionEventImpl).
  DISPENSO_TSAN_ANNOTATE_HAPPENS_AFTER(&event_);
}

inline void RWLockImpl::lock() {
  setWriteBit();
  waitForReaderDrain();
}

inline bool RWLockImpl::tryWriteBit() {
  int val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  return !(val & kWriteBit);
}

inline bool RWLockImpl::try_lock() {
  int val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  if (val & kWriteBit) {
    // Another writer already owns the bit. We did not set it, so we must not
    // clear it on the way out.
    return false;
  }
  if (val == 0) {
    // No writer and no readers: we have exclusive ownership.
    return true;
  }
  // We own the writer bit (it blocks new readers), but readers that were already
  // present must still drain before we hold exclusive access. Give them a short,
  // bounded chance to finish -- only the pre-existing readers can run, since the
  // writer bit keeps new ones out.
  for (int spin = 0; spin < kTryLockDrainSpins; ++spin) {
    cpuRelax();
    if (lockWord().load(std::memory_order_acquire) == kWriteBit) {
      return true;
    }
  }
  // Readers did not drain in time. Release our writer bit (preserving the
  // concurrent reader counts in the low bits) and report failure. try_lock is
  // permitted to fail spuriously, so this is a valid outcome.
  lockWord().fetch_and(kReaderBits, std::memory_order_acq_rel);
  return false;
}

inline void RWLockImpl::unlock() {
  lockWord().fetch_and(kReaderBits, std::memory_order_acq_rel);
}

inline void RWLockImpl::lock_shared() {
  int val = lockWord().fetch_add(1, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    readerRelease();
    while (val & kWriteBit) {
      val = lockWord().load(std::memory_order_acquire);
    }
    val = lockWord().fetch_add(1, std::memory_order_acq_rel);
  }
}

inline bool RWLockImpl::try_lock_shared() {
  int val = lockWord().fetch_add(1, std::memory_order_acq_rel);
  if (val & kWriteBit) {
    readerRelease();
    return false;
  }
  return true;
}

inline void RWLockImpl::unlock_shared() {
  readerRelease();
}

inline void RWLockImpl::lock_upgrade() {
  int val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    val = lockWord().fetch_or(kWriteBit, std::memory_order_acq_rel);
  }
  // We've claimed single write ownership now.  We need to drain off readers, including ourself
  lockWord().fetch_sub(1, std::memory_order_acq_rel);
  waitForReaderDrain();
}

inline void RWLockImpl::lock_downgrade() {
  // Get reader ownership first
  lockWord().fetch_add(1, std::memory_order_acq_rel);
  unlock();
}
} // namespace detail
} // namespace dispenso
