/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

// Implementation notes:
// 1. You should know what you're doing if you are directly using CompletionEventImpl.  Most people
//    should be using CompletionEvent, which is designed for safe use by folks without strong
//    knowledge of atomics, etc...
// 2. intrusiveStatus may be used to gain access to the internal atomic status
// 3. intrusiveStatus may set the status to anything... *but* once wait functions have been called,
//    if intrusiveStatus sets the status to completedStatus, it should not change the status again.
//    After notify, the status should not be modified.
// 4. It is not an error to set the status to completedStatus, and then call notify with
//    completedStatus, but it is very likely to be unnecessary.  Usually, while there may be threads
//    racing to get some statuses, and they should use compare_exchange functions to resolve those,
//    setting completed status should not typically be racy.

#include "notifier_common.h"

namespace dispenso {
namespace detail {

#define DISPENSO_COMPLETION_SUPPORTS_TRY_NOTIFY

#if defined(__linux__)

class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : ftx_(initStatus) {}

  void notify(int completedStatus) {
    status_.store(completedStatus, std::memory_order_release);
    futex(&ftx_, FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(), nullptr, nullptr, 0);
  }

  void tryNotify() {
    futex(&ftx_, FUTEX_WAKE_PRIVATE, std::numeric_limits<int>::max(), nullptr, nullptr, 0);
  }

  void wait(int completedStatus) const {
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      futex(&ftx_, FUTEX_WAIT_PRIVATE, current, nullptr, nullptr, 0);
    }
  }

  bool waitFor(int completedStatus, const std::chrono::duration<double>& relTime) const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }

    double relSeconds = relTime.count();
    if (relSeconds <= 0.0) {
      return false;
    }

    struct timespec ts;
    ts.tv_sec = static_cast<time_t>(relSeconds);
    relSeconds -= static_cast<double>(ts.tv_sec);
    ts.tv_nsec = static_cast<long>(1e9 * relSeconds);

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      if (futex(&ftx_, FUTEX_WAIT_PRIVATE, current, &ts, nullptr, 0) && errno == ETIMEDOUT) {
        // Intentionally not re-checking status: returning false on timeout is consistent with
        // std::condition_variable::wait_for semantics. The benign race where notify arrives
        // concurrently with timeout is handled by the caller retrying if needed.
        return false;
      }
    }
    return true;
  }

  template <class Clock, class Duration>
  bool waitUntil(int completedStatus, const std::chrono::time_point<Clock, Duration>& absTime)
      const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }

    return waitFor(completedStatus, absTime - Clock::now());
  }

  std::atomic<int>& intrusiveStatus() {
    return status_;
  }

  const std::atomic<int>& intrusiveStatus() const {
    return status_;
  }

 private:
  union {
    mutable int ftx_;
    std::atomic<int> status_;
  };
};

#elif defined(__MACH__) && defined(DISPENSO_HAS_OS_SYNC)

// Uses a 64-bit combined {status, waiterCount} to eliminate the Dekker race between notify() and
// wait(). notify() atomically sets status while observing waiterCount via CAS; wait() atomically
// increments waiterCount while observing status via CAS. The 8-byte futex compare detects changes
// to either field. intrusiveStatus() returns a reference to the status portion (lower 32 bits on
// little-endian) for direct CAS/fetch_sub by FutureImplBase and Latch.
class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus)
      : combined_(static_cast<uint64_t>(static_cast<uint32_t>(initStatus))) {}

  void notify(int completedStatus) {
    uint64_t old = combined_.load(std::memory_order_relaxed);
    uint64_t newVal;
    do {
      newVal = (old & kWaiterMask) | static_cast<uint32_t>(completedStatus);
    } while (!combined_.compare_exchange_weak(
        old, newVal, std::memory_order_release, std::memory_order_relaxed));
    if (old & kWaiterMask) {
      mac_futex_wake_all(&combined_, sizeof(uint64_t));
    }
  }

  void tryNotify() {
    if (combined_.load(std::memory_order_acquire) & kWaiterMask) {
      mac_futex_wake_all(&combined_, sizeof(uint64_t));
    }
  }

  void wait(int completedStatus) const {
    uint64_t val = combined_.load(std::memory_order_acquire);
    while (static_cast<int>(val) != completedStatus) {
      uint64_t newVal = val + kOneWaiter;
      if (!combined_.compare_exchange_weak(
              val, newVal, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      mac_futex_wait(&combined_, newVal, sizeof(uint64_t));
      combined_.fetch_sub(kOneWaiter, std::memory_order_acq_rel);
      val = combined_.load(std::memory_order_acquire);
    }
  }

  bool waitFor(int completedStatus, const std::chrono::duration<double>& relTime) const {
    if (static_cast<int>(combined_.load(std::memory_order_acquire)) == completedStatus) {
      return true;
    }

    double relSeconds = relTime.count();
    if (relSeconds <= 0.0) {
      return false;
    }

    uint64_t relTimeUs = static_cast<uint64_t>(relSeconds * 1e6);

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    uint64_t val = combined_.load(std::memory_order_acquire);
    while (static_cast<int>(val) != completedStatus) {
      uint64_t newVal = val + kOneWaiter;
      if (!combined_.compare_exchange_weak(
              val, newVal, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      int ret = mac_futex_wait_for(&combined_, newVal, sizeof(uint64_t), relTimeUs);
      combined_.fetch_sub(kOneWaiter, std::memory_order_acq_rel);
      if (ret && errno == ETIMEDOUT) {
        // Intentionally not re-checking status: returning false on timeout is consistent with
        // std::condition_variable::wait_for semantics. The benign race where notify arrives
        // concurrently with timeout is handled by the caller retrying if needed.
        return false;
      }
      val = combined_.load(std::memory_order_acquire);
    }
    return true;
  }

  template <class Clock, class Duration>
  bool waitUntil(int completedStatus, const std::chrono::time_point<Clock, Duration>& absTime)
      const {
    if (static_cast<int>(combined_.load(std::memory_order_acquire)) == completedStatus) {
      return true;
    }

    return waitFor(completedStatus, absTime - Clock::now());
  }

  std::atomic<int>& intrusiveStatus() {
    return parts_[0];
  }

  const std::atomic<int>& intrusiveStatus() const {
    return parts_[0];
  }

 private:
  static constexpr uint64_t kOneWaiter = 1ULL << 32;
  static constexpr uint64_t kWaiterMask = ~0xFFFFFFFFULL;

  union {
    mutable std::atomic<uint64_t> combined_;
    mutable std::atomic<int> parts_[2]; // [0] = status (little-endian)
  };
};

#elif defined(_WIN32)

class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : status_(initStatus) {}

  void notify(int completedStatus) {
    status_.store(completedStatus, std::memory_order_release);
    WakeByAddressAll(&status_);
  }

  void tryNotify() {
    WakeByAddressAll(&status_);
  }

  void wait(int completedStatus) const {
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      WaitOnAddress(&status_, &current, sizeof(int), kInfiniteWin);
    }
  }

  bool waitFor(int completedStatus, const std::chrono::duration<double>& relTime) const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }

    double relSeconds = relTime.count();
    if (relSeconds <= 0.0) {
      return false;
    }

    int msWait = std::max(1, static_cast<int>(relSeconds * 1000.0));

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      if (!WaitOnAddress(&status_, &current, sizeof(int), msWait) &&
          GetLastError() == kErrorTimeoutWin) {
        // Intentionally not re-checking status: returning false on timeout is consistent with
        // std::condition_variable::wait_for semantics. The benign race where notify arrives
        // concurrently with timeout is handled by the caller retrying if needed.
        return false;
      }
    }
    return true;
  }

  template <class Clock, class Duration>
  bool waitUntil(int completedStatus, const std::chrono::time_point<Clock, Duration>& absTime)
      const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }

    return waitFor(completedStatus, absTime - Clock::now());
  }

  std::atomic<int>& intrusiveStatus() {
    return status_;
  }

  const std::atomic<int>& intrusiveStatus() const {
    return status_;
  }

 private:
  mutable std::atomic<int> status_;
};

#else

#undef DISPENSO_COMPLETION_SUPPORTS_TRY_NOTIFY

// Fallback C++11 implementation.
class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : status_(initStatus) {}

  void notify(int completedStatus) {
    // See https://en.cppreference.com/w/cpp/thread/condition_variable
    // "Even if the shared variable is atomic, it must be modified under the mutex in order to
    // correctly publish the modification to the waiting thread."
    {
      std::lock_guard<std::mutex> lk(mtx_);
      status_.store(completedStatus, std::memory_order_release);
    }
    cv_.notify_all();
  }

  void wait(int completedStatus) const {
    if (status_.load(std::memory_order_acquire) != completedStatus) {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait(lk, [this, completedStatus]() {
        return status_.load(std::memory_order_relaxed) == completedStatus;
      });
    }
  }

  template <class Rep, class Period>
  bool waitFor(int completedStatus, const std::chrono::duration<Rep, Period>& relTime) const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }
    std::unique_lock<std::mutex> lk(mtx_);
    return cv_.wait_for(lk, relTime, [this, completedStatus] {
      return status_.load(std::memory_order_relaxed) == completedStatus;
    });
  }

  template <class Clock, class Duration>
  bool waitUntil(int completedStatus, const std::chrono::time_point<Clock, Duration>& absTime)
      const {
    if (status_.load(std::memory_order_acquire) == completedStatus) {
      return true;
    }
    std::unique_lock<std::mutex> lk(mtx_);
    return cv_.wait_until(lk, absTime, [this, completedStatus] {
      return status_.load(std::memory_order_relaxed) == completedStatus;
    });
  }

  std::atomic<int>& intrusiveStatus() {
    return status_;
  }

  const std::atomic<int>& intrusiveStatus() const {
    return status_;
  }

 private:
  mutable std::mutex mtx_;
  mutable std::condition_variable cv_;
  std::atomic<int> status_;
};
#endif // platform

} // namespace detail
} // namespace dispenso
