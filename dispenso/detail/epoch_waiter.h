/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <atomic>
#include <chrono>

#include "notifier_common.h"

namespace dispenso {
namespace detail {

#if defined(__linux__)

class EpochWaiter {
 public:
  EpochWaiter() : ftx_(0) {}

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    futex(&ftx_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    futex(&ftx_, FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(), nullptr, nullptr, 0);
  }

  uint32_t wait(uint32_t expectedEpoch) const {
    uint32_t current;
    // allow spurious wakeups
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      futex(&ftx_, FUTEX_WAIT_PRIVATE, expectedEpoch, nullptr, nullptr, 0);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t current() const {
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t waitFor(uint32_t expectedEpoch, uint32_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    struct timespec ts;
    ts.tv_sec = relTimeUs / 1000000;
    ts.tv_nsec = (relTimeUs - (ts.tv_sec * 1000000)) * 1000;

    // allow spurious wakeups
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      futex(&ftx_, FUTEX_WAIT_PRIVATE, current, &ts, nullptr, 0);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

 private:
  union {
    mutable int ftx_;
    std::atomic<uint32_t> epoch_;
  };
};

#elif defined(__MACH__)

class EpochWaiter {
 public:
  EpochWaiter() : epoch_(0) {
    semaphore_create(mach_task_self(), &sem_, SYNC_POLICY_FIFO, 0);
  }

  ~EpochWaiter() {
    semaphore_destroy(mach_task_self(), sem_);
  }

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_release);
    semaphore_signal(sem_);
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_release);
    semaphore_signal_all(sem_);
  }

  uint32_t wait(uint32_t expectedEpoch) const {
    mach_timespec_t ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 2000000; // 2 ms
    uint32_t current;
    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      // Here we use timedwait with medium-long wait time. This is because it is possible that we
      // can have the following sequence:
      // 1. We check the status_ condition for this loop
      // 2. notify() sets status_ to true
      // 3. notify() calls semaphore_signal_all
      // 4. We enter semaphore_wait here.  This would result in never being woken.

      // Although the sequencing above is unlikely, it is possible.  By chosing semaphore_timedwait
      // instead of semaphore_wait, we will ensure that we can always complete the wait.
      semaphore_timedwait(sem_, ts);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t waitFor(uint32_t expectedEpoch, uint32_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    mach_timespec_t ts;
    ts.tv_sec = relTimeUs / 1000000;
    ts.tv_nsec = static_cast<clock_res_t>((relTimeUs - (ts.tv_sec * 1000000)) * 1000);

    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      semaphore_timedwait(sem_, ts);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t current() const {
    return epoch_.load(std::memory_order_acquire);
  }

 private:
  semaphore_t sem_;
  std::atomic<uint32_t> epoch_;
};

#elif defined(_WIN32)

class EpochWaiter {
 public:
  EpochWaiter() : epoch_(0) {}

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressSingle(&epoch_);
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressAll(&epoch_);
  }

  uint32_t wait(uint32_t expectedEpoch) const {
    uint32_t current;
    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      WaitOnAddress(&epoch_, &current, sizeof(uint32_t), kInfiniteWin);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t current() const {
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t waitFor(uint32_t expectedEpoch, uint32_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    int msWait = std::max(1, static_cast<int>(relTimeUs / 1000));
    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      WaitOnAddress(&epoch_, &current, sizeof(uint32_t), msWait);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

 private:
  mutable std::atomic<uint32_t> epoch_;
};

#else

// Fallback C++11 implementation.
class EpochWaiter {
 public:
  EpochWaiter() : epoch_(0) {}

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    cv_.notify_one();
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    cv_.notify_all();
  }

  uint32_t wait(uint32_t expectedEpoch) const {
    uint32_t current;
    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait(lk);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t current() const {
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t waitFor(uint32_t expectedEpoch, uint32_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait_for(lk, std::chrono::microseconds(relTimeUs));
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex mtx_;
  mutable std::condition_variable cv_;
  std::atomic<uint32_t> epoch_;
};

#endif // platform

} // namespace detail
} // namespace dispenso
