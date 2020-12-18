// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <dispenso/thread_pool.h>

namespace dispenso {

namespace detail {
template <typename Result>
class FutureBase;

class LimitGatedScheduler;
} // namespace detail

/**
 * <code>TaskSet</code> is an object that allows scheduling multiple functors to a thread pool, and
 * allows to wait on that set of tasks.  <code>TaskSet</code> supplies more efficient schedule/wait
 * than <code>ConcurrentTaskSet</code>, but at the expense of only being usable from one thread at a
 * time.
 *
 * <code>TaskSet</code> is "thread-compatible".  This means that you can safely use
 * different <code>TaskSet</code> objects on different threads concurrently. Any given
 * <code>TaskSet</code> object may only be used from a single thread, so no concurrent use of that
 * object is allowed.
 **/
class TaskSet {
 public:
  /**
   * Construct a TaskSet with the given backing pool.
   *
   * @param pool The backing pool for this TaskSet
   * @param stealingLoadMultiplier An over-load factor.  If this factor of load is reached by the
   * underlying pool, scheduled tasks may run immediately in the calling thread.
   **/
  TaskSet(ThreadPool& p, int32_t stealingLoadMultiplier = 4)
      : pool_(p), token_(p.work_), taskSetLoadFactor_(stealingLoadMultiplier * p.numThreads()) {
#if defined DISPENSO_DEBUG
    assert(stealingLoadMultiplier > 0);
    pool_.outstandingTaskSets_.fetch_add(1, std::memory_order_acquire);
#endif
  }

  TaskSet(TaskSet&& other) = delete;
  TaskSet& operator=(TaskSet&& other) = delete;

  /**
   * Schedule a functor for execution on the underlying pool.  If the load on the
   * underlying pool is high, immediate inline execution may occur on the current thread.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   *
   * @note If <code>f</code> can throw exceptions, then <code>schedule</code> may throw if the task
   * is run inline.  Otherwise, exceptions will be caught on the running thread and best-effort
   * propagated to the <code>ConcurrentTaskSet</code>, where the first one from the set is rethrown
   * in <code>wait</code>.
   **/
  template <typename F>
  void schedule(F&& f) {
    if (outstandingTaskCount_.load(std::memory_order_relaxed) > taskSetLoadFactor_) {
      f();
    } else {
      outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
      pool_.schedule(token_, [this, f = std::move(f)]() mutable {
#if defined(__cpp_exceptions)
        try {
          f();
        } catch (...) {
          trySetCurrentException();
        }
#else
          f();
#endif // __cpp_exceptions
        outstandingTaskCount_.fetch_add(-1, std::memory_order_release);
      });
    }
  }

  /**
   * Schedule a functor for execution on the underlying pool.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   *
   * @note If <code>f</code> can throw exceptions, then exceptions will be caught on the running
   * thread and best-effort propagated to the <code>ConcurrentTaskSet</code>, where the first one
   * from the set is rethrown in <code>wait</code>.
   **/
  template <typename F>
  void schedule(F&& f, ForceQueuingTag fq) {
    outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    pool_.schedule(
        token_,
        [this, f = std::move(f)]() {
#if defined(__cpp_exceptions)
          try {
            f();
          } catch (...) {
            trySetCurrentException();
          }
#else
          f();
#endif // __cpp_exceptions
          outstandingTaskCount_.fetch_add(-1, std::memory_order_release);
        },
        fq);
  }

  /**
   * Wait for all currently scheduled functors to finish execution.  If exceptions are thrown
   * during execution of the set of tasks, <code>wait</code> will propagate the first exception.
   **/
  void wait();

  /**
   * See if the currently scheduled functors can be completed while stealing and executing at most
   * <code>maxToExecute</code> of them from the pool. If not used in conjunction with wait, there
   * may be cases that <code>tryWait</code> must be called multiple times with
   * <code>maxToExecute &gt 0</code> to prevent livelock/deadlock.  If exceptions have been
   * propagated since the last call to <code>wait</code> or <code>tryWait</code>,
   * <code>tryWait</code> will propagate the first of them.
   *
   * @param maxToExecute The maximum number of tasks to proactively execute on the current thread.
   *
   * @return <code>true</code> if all currently scheduled functors have been completed prior to
   * returning, and <code>false</code> otherwise.
   **/
  bool tryWait(size_t maxToExecute);

  /**
   * Get the number of threads backing the underlying thread pool.
   *
   * @return The number of threads in the pool.
   **/
  size_t numPoolThreads() const {
    return pool_.numThreads();
  }

  /**
   * Access the underlying pool.
   *
   * @return The thread pool.
   **/
  ThreadPool& pool() {
    return pool_;
  }

  /**
   * Destroy the TaskSet, first waiting for all currently scheduled functors to
   * finish execution.
   **/
  ~TaskSet() {
    wait();
#if defined DISPENSO_DEBUG
    pool_.outstandingTaskSets_.fetch_add(-1, std::memory_order_release);
#endif
  }

 private:
  void trySetCurrentException();
  void testAndResetException();

  alignas(kCacheLineSize) std::atomic<int32_t> outstandingTaskCount_{0};
  alignas(kCacheLineSize) ThreadPool& pool_;
  moodycamel::ProducerToken token_;
  const int32_t taskSetLoadFactor_;
#if defined(__cpp_exceptions)
  enum ExceptionState { kUnset, kSetting, kSet };
  std::atomic<ExceptionState> guardException_{kUnset};
  std::exception_ptr exception_;
#endif // __cpp_exceptions

  template <typename Result>
  friend class detail::FutureBase;
};

/**
 * <code>ConcurrentTaskSet</code> fulfills the same API as <code>TaskSet</code> with one minor
 * difference: It may be used to schedule tasks concurrently from multiple threads (see more below).
 * It is an object that allows scheduling multiple function-like objects to a thread pool, and
 * allows to wait on that set of tasks.
 *
 * <code>ConcurrentTaskSet</code> is "thread-compatible".  This means that you can safely use
 * different <code>ConcurrentTaskSet</code> objects on different threads concurrently.
 * <code>ConcurrentTaskSet</code> also allows multiple threads to concurrently schedule against it.
 * It is an error to call wait() concurrently with schedule() on the same
 * <code>ConcurrentTaskSet</code>.
 */
class ConcurrentTaskSet {
 public:
  /**
   * Construct a ConcurrentTaskSet with the given backing pool.
   *
   * @param pool The backing pool for this ConcurrentTaskSet
   * @param stealingLoadMultiplier An over-load factor.  If this factor of load is reached by the
   * underlying pool, scheduled tasks may run immediately in the calling thread.
   **/
  ConcurrentTaskSet(ThreadPool& pool, int32_t stealingLoadMultiplier = 4)
      : pool_(pool), taskSetLoadFactor_(stealingLoadMultiplier * pool.numThreads()) {
#if defined DISPENSO_DEBUG
    assert(stealingLoadMultiplier > 0);
    pool_.outstandingTaskSets_.fetch_add(1, std::memory_order_acquire);
#endif
  }

  ConcurrentTaskSet(ConcurrentTaskSet&& other) = delete;
  ConcurrentTaskSet& operator=(ConcurrentTaskSet&& other) = delete;

  /**
   * Schedule a functor for execution on the underlying pool.  If the load on the
   * underlying pool is high, immediate inline execution may occur on the current thread.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   *
   * @note If <code>f</code> can throw exceptions, then <code>schedule</code> may throw if the task
   * is run inline.  Otherwise, exceptions will be caught on the running thread and best-effort
   * propagated to the <code>ConcurrentTaskSet</code>, where the first one from the set is rethrown
   * in <code>wait</code>.
   **/
  template <typename F>
  void schedule(F&& f) {
    if (outstandingTaskCount_.load(std::memory_order_relaxed) > taskSetLoadFactor_) {
      f();
    } else {
      schedule(std::forward<F>(f), ForceQueuingTag());
    }
  }

  /**
   * Schedule a functor for execution on the underlying pool.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   *
   * @note If <code>f</code> can throw exceptions, then exceptions will be caught on the running
   * thread and best-effort propagated to the <code>ConcurrentTaskSet</code>, where the first one
   * from the set is rethrown in <code>wait</code>.
   **/
  template <typename F>
  void schedule(F&& f, ForceQueuingTag fq) {
    outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    pool_.schedule(
        [this, f = std::move(f)]() mutable {
#if defined(__cpp_exceptions)
          try {
            f();
          } catch (...) {
            trySetCurrentException();
          }
#else
          f();
#endif // __cpp_exceptions
          outstandingTaskCount_.fetch_add(-1, std::memory_order_release);
        },
        fq);
  }

  /**
   * Wait for all currently scheduled functors to finish execution.  If exceptions are thrown
   * during execution of the set of tasks, <code>wait</code> will propagate the first exception.
   **/
  void wait();

  /**
   * See if the currently scheduled functors can be completed while stealing and executing at most
   * <code>maxToExecute</code> of them from the pool. If not used in conjunction with wait, there
   * may be cases that <code>tryWait</code> must be called multiple times with
   * <code>maxToExecute &gt 0</code> to prevent livelock/deadlock.  If exceptions have been
   * propagated since the last call to <code>wait</code> or <code>tryWait</code>,
   * <code>tryWait</code> will propagate the first of them.
   *
   * @param maxToExecute The maximum number of tasks to proactively execute on the current thread.
   *
   * @return <code>true</code> if all currently scheduled functors have been completed prior to
   * returning, and <code>false</code> otherwise.
   **/
  bool tryWait(size_t maxToExecute);

  /**
   * Get the number of threads backing the underlying thread pool.
   *
   * @return The number of threads in the pool.
   **/
  size_t numPoolThreads() const {
    return pool_.numThreads();
  }

  /**
   * Access the underlying pool.
   *
   * @return The thread pool.
   **/
  ThreadPool& pool() {
    return pool_;
  }

  /**
   * Destroy the ConcurrentTaskSet, first waiting for all currently scheduled functors to
   * finish execution.
   **/
  ~ConcurrentTaskSet() {
    wait();
#if defined DISPENSO_DEBUG
    pool_.outstandingTaskSets_.fetch_add(-1, std::memory_order_release);
#endif
  }

 private:
  void trySetCurrentException();
  void testAndResetException();

  bool tryExecuteNext() {
    return pool_.tryExecuteNext();
  }

  std::atomic<int32_t> outstandingTaskCount_{0};
  alignas(kCacheLineSize) ThreadPool& pool_;
  const int32_t taskSetLoadFactor_;
#if defined(__cpp_exceptions)
  enum ExceptionState { kUnset, kSetting, kSet };
  std::atomic<ExceptionState> guardException_{kUnset};
  std::exception_ptr exception_;
#endif // __cpp_exceptions

  template <typename Result>
  friend class detail::FutureBase;

  friend class detail::LimitGatedScheduler;
};

} // namespace dispenso
