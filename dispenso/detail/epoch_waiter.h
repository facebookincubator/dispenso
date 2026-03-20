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

  void bumpAndWakeN(int n, int totalWaiters) {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    (void)totalWaiters;
    futex(&ftx_, FUTEX_WAKE_PRIVATE, n, nullptr, nullptr, 0);
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

  uint32_t waitFor(uint32_t expectedEpoch, uint64_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    struct timespec ts;
    ts.tv_sec = static_cast<decltype(ts.tv_sec)>(relTimeUs / 1000000);
    ts.tv_nsec = static_cast<decltype(ts.tv_nsec)>((relTimeUs - (ts.tv_sec * 1000000)) * 1000);

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
  EpochWaiter() : epoch_(0) {}

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    mac_futex_wake_one(&epoch_, sizeof(uint32_t));
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    mac_futex_wake_all(&epoch_, sizeof(uint32_t));
  }

  void bumpAndWakeN(int n, int totalWaiters) {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    if (n >= totalWaiters) {
      mac_futex_wake_all(&epoch_, sizeof(uint32_t));
    } else {
      for (int i = 0; i < n; ++i) {
        mac_futex_wake_one(&epoch_, sizeof(uint32_t));
      }
    }
  }

  uint32_t wait(uint32_t expectedEpoch) const {
    uint32_t current;
    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      mac_futex_wait(&epoch_, expectedEpoch, sizeof(uint32_t));
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t current() const {
    return epoch_.load(std::memory_order_acquire);
  }

  uint32_t waitFor(uint32_t expectedEpoch, uint64_t relTimeUs) const {
    uint32_t current;
    if ((current = epoch_.load(std::memory_order_acquire)) != expectedEpoch) {
      return current;
    }

    // Allow spurious wake
    if ((current = epoch_.load(std::memory_order_acquire)) == expectedEpoch) {
      mac_futex_wait_for(&epoch_, current, sizeof(uint32_t), relTimeUs);
    } else {
      return current;
    }
    return epoch_.load(std::memory_order_acquire);
  }

 private:
  mutable std::atomic<uint32_t> epoch_;
};

#elif defined(_WIN32)

class EpochWaiter {
 public:
  EpochWaiter() : epoch_(0) {}

  void bumpAndWake() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    // Always WakeAll on Windows. WakeByAddressSingle and WakeByAddressAll
    // have comparable per-call kernel cost, but WakeAll keeps more threads
    // in their spin phase (kBackoffYield iterations) where they can pick up
    // subsequent work without another kernel wake transition.
    WakeByAddressAll(&epoch_);
  }

  void bumpAndWakeAll() {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressAll(&epoch_);
  }

  void bumpAndWakeN(int /*n*/, int totalWaiters) {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    // Only use WakeByAddressSingle when there is exactly one waiter —
    // it avoids waking threads that aren't waiting. Otherwise, WakeAll
    // is preferred: same single kernel transition, and keeps threads
    // spinning where they can absorb follow-up work without re-waking.
    if (totalWaiters <= 1) {
      WakeByAddressSingle(&epoch_);
    } else {
      WakeByAddressAll(&epoch_);
    }
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

  uint32_t waitFor(uint32_t expectedEpoch, uint64_t relTimeUs) const {
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

  void bumpAndWakeN(int n, int totalWaiters) {
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    if (n >= totalWaiters) {
      cv_.notify_all();
    } else {
      for (int i = 0; i < n; ++i) {
        cv_.notify_one();
      }
    }
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
