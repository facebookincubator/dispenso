// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <algorithm>
#include <atomic>
#include <chrono>

#if defined(__linux__)
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

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

namespace dispenso {
namespace detail {
static int
futex(int* uaddr, int futex_op, int val, const struct timespec* timeout, int* uaddr2, int val3) {
  return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr, val3);
}

class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : ftx_(initStatus) {}

  void notify(int completedStatus) {
    status_.store(completedStatus, std::memory_order_release);
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
    ts.tv_sec = relSeconds;
    relSeconds -= ts.tv_sec;
    ts.tv_nsec = 1e9 * relSeconds;

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      if (futex(&ftx_, FUTEX_WAIT_PRIVATE, current, &ts, nullptr, 0) && errno == ETIMEDOUT) {
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
} // namespace detail
} // namespace dispenso

#elif defined(__MACH__)
#include <mach/mach.h>

namespace dispenso {
namespace detail {
class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : status_(initStatus) {
    semaphore_create(mach_task_self(), &sem_, SYNC_POLICY_FIFO, 0);
  }

  ~CompletionEventImpl() {
    semaphore_destroy(mach_task_self(), sem_);
  }

  void notify(int completedStatus) {
    status_.store(completedStatus, std::memory_order_release);
    semaphore_signal_all(sem_);
  }

  void wait(int completedStatus) const {
    mach_timespec_t ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 2000000; // 2 ms
    while (status_.load(std::memory_order_acquire) != completedStatus) {
      // Here we use timedwait with medium-long wait time. This is because it is possible that we
      // can have the following sequence:
      // 1. We check the status_ condition for this loop
      // 2. notify() sets status_ to true
      // 3. notify() calls semaphore_signal_all
      // 4. We enter semaphore_wait here.  This would result in never being woken.

      // Although the sequencing above is unlikely, it is possible.  By chosing semaphore_timedwait
      // instead of semaphore_wait, we will ensure that we can always complete the wait.
      semaphore_timedwait(sem_, ts);
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

    mach_timespec_t ts;
    ts.tv_sec = relSeconds;
    relSeconds -= ts.tv_sec;
    ts.tv_nsec = 1e9 * relSeconds;

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    while (status_.load(std::memory_order_acquire) != completedStatus) {
      if (semaphore_timedwait(sem_, ts) == KERN_OPERATION_TIMED_OUT) {
        return status_.load(std::memory_order_acquire) == completedStatus;
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
  semaphore_t sem_;
  std::atomic<int> status_;
};
} // namespace detail
} // namespace dispenso
#elif defined(_WIN32)

#if (defined(_M_ARM64) || defined(_M_ARM)) && !defined(_ARM_)
#define _ARM_
#elif _WIN64
#define _AMD64_
#elif _WIN32
#define _X86_
#else
#error "No valid windows platform"
#endif // platform

#include <errhandlingapi.h>
#include <synchapi.h>

namespace dispenso {
namespace detail {

#ifndef ERROR_TIMEOUT
constexpr int ERROR_TIMEOUT = 0x000005B4;
#endif // ERROR_TIMEOUT
#ifndef INFINITE
constexpr unsigned long INFINITE = -1;
#endif // INFINITE

class CompletionEventImpl {
 public:
  CompletionEventImpl(int initStatus) : status_(initStatus) {}

  void notify(int completedStatus) {
    status_.store(completedStatus, std::memory_order_release);
    WakeByAddressAll(&status_);
  }

  void wait(int completedStatus) const {
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      WaitOnAddress(&status_, &current, sizeof(int), INFINITE);
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

    int msWait = std::max<int>(1, relSeconds * 1000.0);

    // TODO: determine if we should worry about reducing timeout time subsequent times through the
    // loop in the case of spurious wake.
    int current;
    while ((current = status_.load(std::memory_order_acquire)) != completedStatus) {
      if (!WaitOnAddress(&status_, &current, sizeof(int), msWait) &&
          GetLastError() == ERROR_TIMEOUT) {
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
} // namespace detail
} // namespace dispenso
#else

#include <condition_variable>
#include <mutex>

// Fallback C++11 implementation.
namespace dispenso {
namespace detail {
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
    if (status_.load(std::memory_order_acquire)) {
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

 private:
  mutable std::mutex mtx_;
  mutable std::condition_variable cv_;
  std::atomic<int> status_;
};
} // namespace detail
} // namespace dispenso
#endif // platform
