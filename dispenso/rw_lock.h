// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/detail/rw_lock_impl.h>

namespace dispenso {

/**
 * A reader/writer lock interface compatible with std::shared_mutex (for use with std::unique_lock
 * and std::shared_lock).  The interface is designed to be very fast in the face of high levels of
 * contention for high read traffic and low write traffic.
 *
 * @note RWLock is not as fully-featured as std::shared_mutex: It does not go to the OS to wait.
 * This behavior is good for guarding very fast operations, but less good for guarding very slow
 * operations.  Additionally, RWLock is not compatible with std::condition_variable, though
 * std::condition_variable_any may work (untested).  It could be possible to extend RWLock with it's
 * own ConditionVariable, make waiting operations sleep in the OS, and also to add timed functions;
 * however those may slow things down in the fast case.  If some/all of that functionality is
 * needed, use std::shared_mutex, or develop a new type.
 **/
class alignas(kCacheLineSize) RWLock : public detail::RWLockImpl {
 public:
  /**
   * Locks for write access
   *
   * @note It is undefined behavior to recursively lock
   **/
  using detail::RWLockImpl::lock;

  /**
   * Tries to lock for write access, returns if unable to lock
   *
   * @return true if lock was acquired, false otherwise
   **/
  using detail::RWLockImpl::try_lock;

  /**
   * Unlocks write access
   *
   * @note Must already be locked by the current thread of execution, otherwise, the behavior is
   * undefined.
   **/
  using detail::RWLockImpl::unlock;

  /**
   * Locks for read access
   *
   * @note It is undefined behavior to recursively lock
   **/
  using detail::RWLockImpl::lock_shared;

  /**
   * Tries to lock for read access, returns if unable to lock
   *
   * @return true if lock was acquired, false otherwise
   *
   * @note It is undefined behavior to recursively lock
   **/
  using detail::RWLockImpl::try_lock_shared;

  /**
   * Unlocks read access
   *
   * @note Must already be locked by the current thread of execution, otherwise, the behavior is
   * undefined.
   **/
  using detail::RWLockImpl::unlock_shared;

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
  using detail::RWLockImpl::lock_upgrade;

  /**
   * Downgrade the lock from a writer lock to a reader lock.
   *
   * @note Calling this if the writer lock is not held results in undefined behavior
   **/
  using detail::RWLockImpl::lock_downgrade;
};

/**
 * An unaligned version of the RWLock.  This could be useful if you e.g. want to create an array of
 * these to guard a large number of slots, and the likelihood of multiple threads touching any
 * region concurrently is low.  All other behavior remains the same, so refer to the documentation
 * for RWLock.
 **/
class UnalignedRWLock : public detail::RWLockImpl {};

} // namespace dispenso
