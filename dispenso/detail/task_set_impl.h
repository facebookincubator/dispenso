/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file task_set.h
 * A file providing TaskSet and ConcurrentTaskSet.  These interfaces allow the user to
 * submit/schedule multiple closures and then wait on them.
 **/

#pragma once

#include <dispenso/thread_pool.h>

namespace dispenso {

class TaskSetBase;

namespace detail {
template <typename Result>
class FutureBase;

class LimitGatedScheduler;

} // namespace detail

class TaskSetBase {
 public:
  TaskSetBase(ThreadPool& p, ssize_t stealingLoadMultiplier = 4)
      : pool_(p), taskSetLoadFactor_(stealingLoadMultiplier * p.numThreads()) {
#if defined DISPENSO_DEBUG
    assert(stealingLoadMultiplier > 0);
    pool_.outstandingTaskSets_.fetch_add(1, std::memory_order_acquire);
#endif
  }

  TaskSetBase(TaskSetBase&& other) = delete;
  TaskSetBase& operator=(TaskSetBase&& other) = delete;

  ssize_t numPoolThreads() const {
    return pool_.numThreads();
  }

  ThreadPool& pool() {
    return pool_;
  }

  ~TaskSetBase() {
#if defined DISPENSO_DEBUG
    pool_.outstandingTaskSets_.fetch_sub(1, std::memory_order_release);
#endif
  }

 protected:
  template <typename F>
  auto packageTask(F&& f) {
    outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    return [this, f = std::move(f)]() mutable {
#if defined(__cpp_exceptions)
      try {
        f();
      } catch (...) {
        trySetCurrentException();
      }
#else
      f();
#endif // __cpp_exceptions
      outstandingTaskCount_.fetch_sub(1, std::memory_order_release);
    };
  }

  DISPENSO_DLL_ACCESS void trySetCurrentException();
  void testAndResetException();

  alignas(kCacheLineSize) std::atomic<ssize_t> outstandingTaskCount_{0};
  alignas(kCacheLineSize) ThreadPool& pool_;
  const ssize_t taskSetLoadFactor_;
#if defined(__cpp_exceptions)
  enum ExceptionState { kUnset, kSetting, kSet };
  std::atomic<ExceptionState> guardException_{kUnset};
  std::exception_ptr exception_;
#endif // __cpp_exceptions
};

} // namespace dispenso
