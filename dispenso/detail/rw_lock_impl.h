// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/platform.h>

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

 private:
  static constexpr uint32_t kWriteBit = 0x80000000;
  static constexpr uint32_t kReaderBits = 0x7fffffff;
  std::atomic<uint32_t> lock_{0};
};

inline void RWLockImpl::lock() {
  uint32_t val = lock_.fetch_or(kWriteBit, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    val = lock_.fetch_or(kWriteBit, std::memory_order_acq_rel);
  }
  // We've claimed single write ownership now.  We need to drain off readers
  while (val != kWriteBit) {
    val = lock_.load(std::memory_order_acquire);
  }
}

inline bool RWLockImpl::try_lock() {
  uint32_t val = lock_.fetch_or(kWriteBit, std::memory_order_acq_rel);
  return !(val & kWriteBit);
}

inline void RWLockImpl::unlock() {
  lock_.fetch_and(kReaderBits, std::memory_order_acq_rel);
}

inline void RWLockImpl::lock_shared() {
  uint32_t val = lock_.fetch_add(1, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    val = lock_.fetch_sub(1, std::memory_order_acq_rel);
    while (val & kWriteBit) {
      val = lock_.load(std::memory_order_acquire);
    }

    val = lock_.fetch_add(1, std::memory_order_acq_rel);
  }
}

inline bool RWLockImpl::try_lock_shared() {
  uint32_t val = lock_.fetch_add(1, std::memory_order_acq_rel);
  if (val & kWriteBit) {
    lock_.fetch_sub(1, std::memory_order_acq_rel);
    return false;
  }
  return true;
}

inline void RWLockImpl::unlock_shared() {
  lock_.fetch_sub(1, std::memory_order_acq_rel);
}

inline void RWLockImpl::lock_upgrade() {
  uint32_t val = lock_.fetch_or(kWriteBit, std::memory_order_acq_rel);
  while (val & kWriteBit) {
    val = lock_.fetch_or(kWriteBit, std::memory_order_acq_rel);
  }
  // We've claimed single write ownership now.  We need to drain off readers, including ourself
  lock_.fetch_sub(1, std::memory_order_acq_rel);
  while (val != kWriteBit) {
    val = lock_.load(std::memory_order_acquire);
  }
}

inline void RWLockImpl::lock_downgrade() {
  // Get reader ownership first
  lock_.fetch_add(1, std::memory_order_acq_rel);
  unlock();
}
} // namespace detail
} // namespace dispenso
