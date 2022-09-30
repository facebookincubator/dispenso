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

DISPENSO_DLL_ACCESS void pushThreadTaskSet(TaskSetBase* tasks);
DISPENSO_DLL_ACCESS void popThreadTaskSet();

} // namespace detail

DISPENSO_DLL_ACCESS TaskSetBase* parentTaskSet();

class TaskSetBase {
 public:
  TaskSetBase(ThreadPool& p, ssize_t stealingLoadMultiplier = 4)
      : pool_(p), taskSetLoadFactor_(stealingLoadMultiplier * p.numThreads()) {
#if defined DISPENSO_DEBUG
    assert(stealingLoadMultiplier > 0);
    pool_.outstandingTaskSets_.fetch_add(1, std::memory_order_acquire);
#endif

    if (auto* pt = parentTaskSet()) {
      pt->registerChild(this);
      if (pt->canceled()) {
        canceled_.store(true, std::memory_order_release);
      }
    }
  }

  TaskSetBase(TaskSetBase&& other) = delete;
  TaskSetBase& operator=(TaskSetBase&& other) = delete;

  ssize_t numPoolThreads() const {
    return pool_.numThreads();
  }

  ThreadPool& pool() {
    return pool_;
  }

  void cancel() {
    canceled_.store(true, std::memory_order_release);
    cancelChildren();
  }

  bool canceled() const {
    return canceled_.load(std::memory_order_acquire);
  }

  ~TaskSetBase() {
#if defined DISPENSO_DEBUG
    pool_.outstandingTaskSets_.fetch_sub(1, std::memory_order_release);
#endif

    if (auto* p = parentTaskSet()) {
      p->unregisterChild(this);
    }
  }

 protected:
  template <typename F>
  auto packageTask(F&& f) {
    outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    return [this, f = std::move(f)]() mutable {
      detail::pushThreadTaskSet(this);
      if (!canceled_.load(std::memory_order_acquire)) {
#if defined(__cpp_exceptions)
        try {
          f();
        } catch (...) {
          trySetCurrentException();
        }
#else
        f();
#endif // __cpp_exceptions
      }
      detail::popThreadTaskSet();
      outstandingTaskCount_.fetch_sub(1, std::memory_order_release);
    };
  }

  DISPENSO_DLL_ACCESS void trySetCurrentException();
  bool testAndResetException();

  void registerChild(TaskSetBase* child) {
    std::lock_guard<std::mutex> lk(mtx_);

    child->prev_ = tail_;
    child->next_ = nullptr;
    if (tail_) {
      tail_->next_ = child;
      tail_ = child;
    } else {
      head_ = tail_ = child;
    }
  }

  void unregisterChild(TaskSetBase* child) {
    std::lock_guard<std::mutex> lk(mtx_);

    if (child->prev_) {
      child->prev_->next_ = child->next_;
    } else {
      // We're head
      assert(child == head_);
      head_ = child->next_;
    }
    if (child->next_) {
      child->next_->prev_ = child->prev_;
    } else {
      // We're tail
      assert(child == tail_);
      tail_ = child->prev_;
    }
  }

  void cancelChildren() {
    std::lock_guard<std::mutex> lk(mtx_);

    auto* node = head_;
    while (node) {
      node->cancel();
      node = node->next_;
    }
  }

  alignas(kCacheLineSize) std::atomic<ssize_t> outstandingTaskCount_{0};
  alignas(kCacheLineSize) ThreadPool& pool_;
  alignas(kCacheLineSize) std::atomic<bool> canceled_{false};
  const ssize_t taskSetLoadFactor_;
#if defined(__cpp_exceptions)
  enum ExceptionState { kUnset, kSetting, kSet };
  std::atomic<ExceptionState> guardException_{kUnset};
  std::exception_ptr exception_;
#endif // __cpp_exceptions

  // This mutex guards modifications/use of the intusive linked list between head_ and tail_
  std::mutex mtx_;
  TaskSetBase* head_{nullptr};
  TaskSetBase* tail_{nullptr};

  // prev_ and next_ are links in our *parent's* intrusive linked list.
  TaskSetBase* prev_;
  TaskSetBase* next_;
};

} // namespace dispenso
