/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file thread_pool.h
 * @ingroup group_core
 * A file providing ThreadPool.  This is the heart of dispenso.  All other scheduling paradigms,
 * including TaskSets, Futures, pipelines, and parallel loops, are built on top of ThreadPool.
 **/

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

#include <moodycamel/concurrentqueue.h>

#include <dispenso/detail/epoch_waiter.h>
#include <dispenso/detail/per_thread_info.h>
#include <dispenso/once_function.h>
#include <dispenso/platform.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {

#if !defined(DISPENSO_WAKEUP_ENABLE)
#if defined(_WIN32) || defined(__linux__)
#define DISPENSO_WAKEUP_ENABLE 1
#else
// TODO(bbudge):  For now, only enable Linux and Windows.  On Mac still need to figure out how to
// wake more quickly (see e.g.
// https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/scheduler/scheduler.html)
#define DISPENSO_WAKEUP_ENABLE 0
#endif // Linux or Windows
#endif // DISPENSO_WAKEUP_ENABLE

#if !defined(DISPENSO_POLL_PERIOD_US)
#if defined(_WIN32)
#define DISPENSO_POLL_PERIOD_US 1000
#else
#if !(DISPENSO_WAKEUP_ENABLE)
#define DISPENSO_POLL_PERIOD_US 200
#else
#define DISPENSO_POLL_PERIOD_US (1 << 15) // Determined empirically good on dual Xeon Linux
#endif // DISPENSO_WAKEUP_ENABLE
#endif // PLATFORM
#endif // DISPENSO_POLL_PERIOD_US

constexpr uint32_t kDefaultSleepLenUs = DISPENSO_POLL_PERIOD_US;

constexpr bool kDefaultWakeupEnable = DISPENSO_WAKEUP_ENABLE;

/**
 * A simple tag specifier that can be fed to TaskSets and
 * ThreadPools to denote that the current thread should never immediately execute a functor, but
 * rather, the functor should always be placed in the ThreadPool's queue.
 **/
struct ForceQueuingTag {};

/**
 * The basic executor for dispenso.  It provides typical thread pool functionality, plus allows work
 * stealing by related types (e.g. TaskSet, Future, etc...), which prevents deadlock when waiting
 * for pool-recursive tasks.
 */
class alignas(kCacheLineSize) ThreadPool {
 public:
  /**
   * Construct a thread pool.
   *
   * @param n The number of threads to spawn at construction.
   * @param poolLoadMultiplier A parameter that specifies how overloaded the pool should be before
   * allowing the current thread to self-steal work.
   **/
  DISPENSO_DLL_ACCESS ThreadPool(size_t n, size_t poolLoadMultiplier = 32);

  /**
   * Enable or disable signaling wake functionality.  If enabled, this will try to ensure that
   * threads are woken up proactively when work has not been available and it becomes available.
   * This function is blocking and potentially very slow.  Repeated use is discouraged.
   *
   * @param enable If set true, turns on signaling wake.  If false, turns it off.
   * @param sleepDuration If enable is true, this is the length of time a thread will wait for a
   * signal before waking up.  If enable is false, this is the length of time a thread will sleep
   * between polling.
   *
   * @note It is highly recommended to leave signaling wake enabled on Windows platforms, as
   * sleeping/polling tends to perform poorly for intermittent workloads.  For Mac/Linux platforms,
   * it is okay to enable signaling wake, particularly if you wish to set a longer expected duration
   * between work.  If signaling wake is disabled, ensure sleepDuration is small (e.g. 200us) for
   * best performance.  Most users will not need to call this function, as defaults are reasonable.
   *
   *
   **/
  template <class Rep, class Period>
  void setSignalingWake(
      bool enable,
      const std::chrono::duration<Rep, Period>& sleepDuration =
          std::chrono::microseconds(kDefaultSleepLenUs)) {
    setSignalingWake(
        enable,
        static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(sleepDuration).count()));
  }

  /**
   * Change the number of threads backing the thread pool.  This is a blocking and potentially
   * slow operation, and repeatedly resizing is discouraged.
   *
   * @param n The number of threads in use after call completion
   **/
  DISPENSO_DLL_ACCESS void resize(ssize_t n) {
    std::lock_guard<std::mutex> lk(threadsMutex_);
    resizeLocked(n);
  }

  /**
   * Get the number of threads backing the pool.  If called concurrently to <code>resize</code>, the
   * number returned may be stale.
   *
   * @return The current number of threads backing the pool.
   **/
  ssize_t numThreads() const {
    return numThreads_.load(std::memory_order_relaxed);
  }

  /**
   * Schedule a functor to be executed.  If the pool's load factor is high, execution may happen
   * inline by the calling thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f);

  /**
   * Schedule a functor to be executed.  The functor will always be queued and executed by pool
   * threads.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag);

  /**
   * Destruct the pool.  This destructor is blocking until all queued work is completed.  It is
   * illegal to call the destructor while any other thread makes calls to the pool (as is generally
   * the case with C++ classes).
   **/
  DISPENSO_DLL_ACCESS ~ThreadPool();

 private:
  class PerThreadData {
   public:
    void setThread(std::thread&& t);

    bool running();

    void stop();

    ~PerThreadData();

   public:
    alignas(kCacheLineSize) std::thread thread_;
    std::atomic<bool> running_{true};
  };

  DISPENSO_DLL_ACCESS uint32_t wait(uint32_t priorEpoch);
  DISPENSO_DLL_ACCESS void wake();

  void setSignalingWake(bool enable, uint32_t sleepDurationUs) {
    std::lock_guard<std::mutex> lk(threadsMutex_);
    ssize_t currentPoolSize = numThreads();
    resizeLocked(0);
    enableEpochWaiter_.store(enable, std::memory_order_release);
    sleepLengthUs_.store(sleepDurationUs, std::memory_order_release);
    resizeLocked(currentPoolSize);
  }

  DISPENSO_DLL_ACCESS void resizeLocked(ssize_t n);

  void executeNext(OnceFunction work);

  DISPENSO_DLL_ACCESS void threadLoop(PerThreadData& threadData);

  bool tryExecuteNext();
  bool tryExecuteNextFromProducerToken(moodycamel::ProducerToken& token);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag);

  void conditionallyWake() {
    if (enableEpochWaiter_.load(std::memory_order_acquire)) {
      // A rare race to overwake is preferable to a race that underwakes.
      auto queuedWork = queuedWork_.fetch_add(1, std::memory_order_acq_rel) + 1;
      auto idle = idleButAwake_.load(std::memory_order_acquire);
      if (idle < queuedWork) {
        wake();
      }
    }
  }

 public:
  // If we are not yet C++17, we provide aligned new/delete to avoid false sharing.
#if __cplusplus < 201703L
  static void* operator new(size_t sz) {
    return detail::alignedMalloc(sz);
  }
  static void operator delete(void* ptr) {
    return detail::alignedFree(ptr);
  }
#endif // __cplusplus

 private:
  mutable std::mutex threadsMutex_;
  std::deque<PerThreadData> threads_;
  size_t poolLoadMultiplier_;
  std::atomic<ssize_t> poolLoadFactor_;
  std::atomic<ssize_t> numThreads_;

  moodycamel::ConcurrentQueue<OnceFunction> work_;

  alignas(kCacheLineSize) std::atomic<ssize_t> queuedWork_{0};
  alignas(kCacheLineSize) std::atomic<ssize_t> idleButAwake_{0};

  alignas(kCacheLineSize) std::atomic<ssize_t> workRemaining_{0};

  alignas(kCacheLineSize) detail::EpochWaiter epochWaiter_;
  alignas(kCacheLineSize) std::atomic<bool> enableEpochWaiter_{kDefaultWakeupEnable};
  std::atomic<uint32_t> sleepLengthUs_{kDefaultSleepLenUs};

#if defined DISPENSO_DEBUG
  alignas(kCacheLineSize) std::atomic<ssize_t> outstandingTaskSets_{0};
#endif // NDEBUG

  friend class ConcurrentTaskSet;
  friend class TaskSet;
  friend class TaskSetBase;
};

/**
 * Get access to the global thread pool.
 *
 * @return the global thread pool
 **/
DISPENSO_DLL_ACCESS ThreadPool& globalThreadPool();

/**
 * Change the number of threads backing the global thread pool.
 *
 * @param numThreads The number of threads to back the global thread pool.
 **/
DISPENSO_DLL_ACCESS void resizeGlobalThreadPool(size_t numThreads);

// ----------------------------- Implementation details -------------------------------------

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedule(F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedule(F&& f, ForceQueuingTag) {
  if (auto* token =
          static_cast<moodycamel::ProducerToken*>(detail::PerPoolPerThreadInfo::producer(this))) {
    schedule(*token, std::forward<F>(f), ForceQueuingTag());
    return;
  }

  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  bool enqueued = work_.enqueue({std::forward<F>(f)});
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  (void)(enqueued); // unused
  assert(enqueued);

  conditionallyWake();
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(token, std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag) {
  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  bool enqueued = work_.enqueue(token, {std::forward<F>(f)});
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  (void)(enqueued); // unused
  assert(enqueued);

  conditionallyWake();
}

inline bool ThreadPool::tryExecuteNext() {
  OnceFunction next;
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  bool dequeued = work_.try_dequeue(next);
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  if (dequeued) {
    executeNext(std::move(next));
    return true;
  }
  return false;
}

inline bool ThreadPool::tryExecuteNextFromProducerToken(moodycamel::ProducerToken& token) {
  OnceFunction next;
  if (work_.try_dequeue_from_producer(token, next)) {
    executeNext(std::move(next));
    return true;
  }
  return false;
}

inline void ThreadPool::executeNext(OnceFunction next) {
  next();
  workRemaining_.fetch_add(-1, std::memory_order_relaxed);
}

} // namespace dispenso
