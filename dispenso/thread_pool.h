// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

#include <concurrentqueue.h>

#include <dispenso/detail/per_thread_info.h>
#include <dispenso/once_function.h>
#include <dispenso/platform.h>

namespace dispenso {

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
  ThreadPool(size_t n, size_t poolLoadMultiplier = 32)
      : poolLoadMultiplier_(poolLoadMultiplier),
        poolLoadFactor_(n * poolLoadMultiplier),
        numThreads_(n) {
#if defined DISPENSO_DEBUG
    assert(poolLoadMultiplier > 0);
#endif // DISPENSO_DEBUG
    for (size_t i = 0; i < n; ++i) {
      threads_.emplace_back();
      auto& back = threads_.back();
      back.running = true;
      back.thread = std::thread([this, &running = back.running]() { threadLoop(running); });
    }
  }

  /**
   * Change the number of threads backing the thread pool.  This is a blocking and potentially slow
   * operation, and repeatedly resizing is discouraged.
   *
   * @param n The number of threads in use after call completion
   **/
  void resize(size_t n);

  /**
   * Get the number of threads backing the pool.  If called concurrently to <code>resize</code>, the
   * number returned may be stale.
   *
   * @return The current number of threads backing the pool.
   **/
  size_t numThreads() const {
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
  void schedule(F&& f, ForceQueuingTag);

  /**
   * Destruct the pool.  This destructor is blocking until all queued work is completed.  It is
   * illegal to call the destructor while any other thread makes calls to the pool (as is generally
   * the case with C++ classes).
   **/
  ~ThreadPool();

 private:
  void executeNext(OnceFunction work);

  void threadLoop(std::atomic<bool>& running);

  bool tryExecuteNext();
  bool tryExecuteNextFromProducerToken(moodycamel::ProducerToken& token);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag);

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
  struct PerThreadData {
    alignas(kCacheLineSize) std::thread thread;
    std::atomic<bool> running;
  };

  mutable std::mutex threadsMutex_;
  std::deque<PerThreadData> threads_;
  size_t poolLoadMultiplier_;
  std::atomic<ssize_t> poolLoadFactor_;
  std::atomic<size_t> numThreads_;

  moodycamel::ConcurrentQueue<OnceFunction> work_;

  alignas(kCacheLineSize) std::atomic<ssize_t> workRemaining_{0};

#if defined DISPENSO_DEBUG
  alignas(kCacheLineSize) std::atomic<ssize_t> outstandingTaskSets_{0};
#endif // NDEBUG

  friend class ConcurrentTaskSet;
  friend class TaskSet;
};

/**
 * Get access to the global thread pool.
 *
 * @return the global thread pool
 **/
ThreadPool& globalThreadPool();

/**
 * Change the number of threads backing the global thread pool.
 *
 * @param numThreads The number of threads to back the global thread pool.
 **/
void resizeGlobalThreadPool(size_t numThreads);

// ----------------------------- Implementation details -------------------------------------

template <typename F>
inline void ThreadPool::schedule(F&& f) {
  size_t curWork = workRemaining_.load(std::memory_order_relaxed);
  size_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(std::move(f), ForceQueuingTag());
  }
}

template <typename F>
inline void ThreadPool::schedule(F&& f, ForceQueuingTag) {
  workRemaining_.fetch_add(1, std::memory_order_release);
  bool enqueued = work_.enqueue({std::forward<F>(f)});
  (void)(enqueued); // unused
  assert(enqueued);
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f) {
  size_t curWork = workRemaining_.load(std::memory_order_relaxed);
  size_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(token, std::move(f), ForceQueuingTag());
  }
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag) {
  workRemaining_.fetch_add(1, std::memory_order_release);
  bool enqueued = work_.enqueue(token, {std::forward<F>(f)});
  (void)(enqueued); // unused
  assert(enqueued);
}

inline bool ThreadPool::tryExecuteNext() {
  OnceFunction next;
  if (work_.try_dequeue(next)) {
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
