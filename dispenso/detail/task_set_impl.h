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

#include <dispenso/detail/per_thread_info.h>
#include <dispenso/thread_pool.h>

namespace dispenso {

/**
 * Default pool-recursive inline load factor.  When a pool thread schedules
 * work and workRemaining exceeds numThreads * this factor, the work is
 * executed inline instead of queued.  Lower values (1.0) favor inlining
 * (good for pipeline data locality), higher values (3.0) favor distribution
 * (good for wide graph parallelism).
 **/
constexpr float kDefaultPoolRecursiveLoadFactor = 1.5f;
class TaskSetBase;

namespace detail {
template <typename Result>
class FutureBase;
template <typename Result>
class FutureImplBase;

class LimitGatedScheduler;

struct SchedulePlacedWrapper;

DISPENSO_DLL_ACCESS void pushThreadTaskSet(TaskSetBase* tasks);
DISPENSO_DLL_ACCESS void popThreadTaskSet();

} // namespace detail

DISPENSO_DLL_ACCESS TaskSetBase* parentTaskSet();

class TaskSetBase {
 public:
  TaskSetBase(
      ThreadPool& p,
      ParentCascadeCancel registerForParentCancel = ParentCascadeCancel::kOff,
      ssize_t stealingLoadMultiplier = 4)
      : pool_(p), taskSetLoadFactor_(stealingLoadMultiplier * p.numThreads()) {
#if defined DISPENSO_DEBUG
    assert(stealingLoadMultiplier > 0);
    pool_.outstandingTaskSets_.fetch_add(1, std::memory_order_acquire);
#endif

    parent_ = (registerForParentCancel == ParentCascadeCancel::kOn) ? parentTaskSet() : nullptr;

    if (parent_) {
      parent_->registerChild(this);
      if (parent_->canceled()) {
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

  /**
   * Check whether an exception has been captured by this task set.
   * When exceptions are disabled at compile time, this always returns false,
   * allowing the compiler to eliminate exception-related branches entirely.
   **/
#if defined(__cpp_exceptions)
  bool hasException() const {
    return guardException_.load(std::memory_order_acquire) != kUnset;
  }
#else
  constexpr bool hasException() const {
    return false;
  }
#endif

  ~TaskSetBase() {
#if defined DISPENSO_DEBUG
    pool_.outstandingTaskSets_.fetch_sub(1, std::memory_order_release);
#endif

    if (parent_) {
      parent_->unregisterChild(this);
    }
  }

 protected:
  template <typename F>
  auto packageTask(F&& f) {
    outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    return [this, f = std::move(f)]() mutable {
      // Skip push/pop if this TaskSet is already the current parent on this
      // thread. This happens with ConcurrentTaskSet self-recursion (scheduling
      // tasks from within tasks on the same set), which is normal and expected.
      // Only actual nesting of *different* TaskSets needs stack tracking.
      bool pushed = (parentTaskSet() != this);
      if (pushed) {
        detail::pushThreadTaskSet(this);
      }
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
      if (pushed) {
        detail::popThreadTaskSet();
      }
      outstandingTaskCount_.fetch_sub(1, std::memory_order_release);
    };
  }

  // Package a task without incrementing the counter (for bulk scheduling where
  // the counter is incremented once for the entire batch).
  template <typename F>
  auto packageTaskNoIncrement(F&& f) {
    return [this, f = std::move(f)]() mutable {
      // Same conditional guard as packageTask: skip push/pop if this TaskSet
      // is already the current parent (self-recursive bulk scheduling).
      bool pushed = (parentTaskSet() != this);
      if (pushed) {
        detail::pushThreadTaskSet(this);
      }
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
      if (pushed) {
        detail::popThreadTaskSet();
      }
      outstandingTaskCount_.fetch_sub(1, std::memory_order_release);
    };
  }

  // Run gen(i)() inline with exception handling. Shared by scheduleBulkImpl's
  // overloaded and at-capacity paths.
  template <typename Generator>
  DISPENSO_INLINE void invokeInline(Generator&& gen, size_t i) {
    detail::InlineDepthGuard depthGuard;
#if defined(__cpp_exceptions)
    try {
      gen(i)();
    } catch (...) {
      trySetCurrentException();
    }
#else
    gen(i)();
#endif // __cpp_exceptions
  }

  // Check pool-level overload: recursive load factor (tight, numThreads*1.5)
  // for pool-recursive callers, global poolLoadFactor_ (loose, numThreads*32)
  // for external callers. Separated from scheduleBulkImpl for CCN reduction.
  DISPENSO_INLINE bool
  shouldInlineBulk(ssize_t curWork, ssize_t numPool, float poolRecursiveLoadFactor) const {
    return (detail::PerPoolPerThreadInfo::isPoolRecursive(&pool_) &&
            curWork >
                static_cast<ssize_t>(static_cast<float>(numPool) * poolRecursiveLoadFactor)) ||
        curWork > pool_.poolLoadFactor_.load(std::memory_order_relaxed);
  }

  template <typename Generator>
  void scheduleBulkImpl(
      size_t count,
      Generator&& gen,
      moodycamel::ProducerToken* token,
      float poolRecursiveLoadFactor = kDefaultPoolRecursiveLoadFactor) {
    if (count == 0) {
      return;
    }

    ssize_t numPool = pool_.numThreads();

    // Ring fast path: for kStatic parallel_for where count ≈ numPool.
    // Push task i directly to ring i for deterministic thread-to-chunk affinity.
    if (count * 4 >= static_cast<size_t>(numPool) && count <= static_cast<size_t>(numPool) &&
        pool_.numRings_.load(std::memory_order_relaxed) >= count &&
        !detail::PerPoolPerThreadInfo::isPoolRecursive(&pool_) &&
        outstandingTaskCount_.load(std::memory_order_relaxed) <= taskSetLoadFactor_) {
      outstandingTaskCount_.fetch_add(static_cast<ssize_t>(count), std::memory_order_acquire);
      pool_.scheduleBulkToRings(
          count, [this, &gen](size_t j) { return packageTaskNoIncrement(gen(j)); }, token);
      return;
    }

    // Standard path: interleave enqueue and inline execution based on load.
    size_t chunkSize = static_cast<size_t>(numPool) + static_cast<size_t>(numPool) / 2;
    if (chunkSize < 1) {
      chunkSize = 1;
    }

    size_t i = 0;
    while (i < count) {
      if (canceled()) {
        break;
      }
      ssize_t outstanding = outstandingTaskCount_.load(std::memory_order_relaxed);
      ssize_t curWork = pool_.workRemaining_.load(std::memory_order_relaxed);
      ssize_t room = taskSetLoadFactor_ - outstanding;

      if ((room <= 0 || shouldInlineBulk(curWork, numPool, poolRecursiveLoadFactor)) &&
          detail::PerPoolPerThreadInfo::canInlineSchedule()) {
        // At/over our task set limit, or pool is overloaded — run inline.
        invokeInline(gen, i);
        ++i;
      } else {
        // If inline execution is capped by recursion depth, queue work even
        // when the task-set load factor says there is no room.
        size_t enqueueLimit = room > 0 ? std::min(chunkSize, static_cast<size_t>(room)) : chunkSize;
        size_t toEnqueue = std::min(count - i, enqueueLimit);
        outstandingTaskCount_.fetch_add(static_cast<ssize_t>(toEnqueue), std::memory_order_acquire);
        size_t base = i;
        pool_.scheduleBulkEnqueue(
            toEnqueue,
            [this, &gen, base](size_t j) { return packageTaskNoIncrement(gen(base + j)); },
            token);
        i += toEnqueue;
      }
    }
  }

  template <typename Generator>
  void scheduleBulkImplPlaced(
      size_t count,
      Generator&& gen,
      float poolRecursiveLoadFactor = kDefaultPoolRecursiveLoadFactor) {
    if (count == 0) {
      return;
    }

    ssize_t numPool = pool_.numThreads();
    size_t chunkSize = static_cast<size_t>(numPool) + static_cast<size_t>(numPool) / 2;
    if (chunkSize < 1) {
      chunkSize = 1;
    }

    size_t i = 0;
    while (i < count) {
      if (canceled()) {
        break;
      }
      ssize_t outstanding = outstandingTaskCount_.load(std::memory_order_relaxed);
      ssize_t curWork = pool_.workRemaining_.load(std::memory_order_relaxed);
      ssize_t room = taskSetLoadFactor_ - outstanding;

      if ((room <= 0 || shouldInlineBulk(curWork, numPool, poolRecursiveLoadFactor)) &&
          detail::PerPoolPerThreadInfo::canInlineSchedule()) {
        invokeInline(gen, i);
        ++i;
      } else {
        // If inline execution is capped by recursion depth, queue work even
        // when the task-set load factor says there is no room.
        size_t enqueueLimit = room > 0 ? std::min(chunkSize, static_cast<size_t>(room)) : chunkSize;
        size_t toEnqueue = std::min(count - i, enqueueLimit);
        outstandingTaskCount_.fetch_add(static_cast<ssize_t>(toEnqueue), std::memory_order_acquire);
        size_t base = i;
        pool_.scheduleBulkPlaced(toEnqueue, [this, &gen, base](size_t j) {
          return packageTaskNoIncrement(gen(base + j));
        });
        i += toEnqueue;
      }
    }
  }

  // Force-queue variant: enqueues every task to the pool, never runs inline on
  // the caller. Needed when the caller must not be trapped executing a task
  // before the rest are scheduled (e.g. adaptive parallel_for no-wait, where
  // an inline-executed worker would block on liveTasks waiting for peers that
  // haven't been queued yet).
  template <typename Generator>
  void scheduleBulkImplForceQueue(size_t count, Generator&& gen, moodycamel::ProducerToken* token) {
    if (count == 0) {
      return;
    }
    ssize_t numPool = pool_.numThreads();
    size_t chunkSize = static_cast<size_t>(numPool) + static_cast<size_t>(numPool) / 2;
    if (chunkSize < 1) {
      chunkSize = 1;
    }
    size_t i = 0;
    while (i < count) {
      if (canceled()) {
        break;
      }
      size_t toEnqueue = std::min(count - i, chunkSize);
      outstandingTaskCount_.fetch_add(static_cast<ssize_t>(toEnqueue), std::memory_order_acquire);
      size_t base = i;
      pool_.scheduleBulkEnqueue(
          toEnqueue,
          [this, &gen, base](size_t j) { return packageTaskNoIncrement(gen(base + j)); },
          token);
      i += toEnqueue;
    }
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
  // Always present to ensure stable ABI layout regardless of __cpp_exceptions.
  enum ExceptionState { kUnset, kSetting, kSet };
  std::atomic<ExceptionState> guardException_{kUnset};
  std::exception_ptr exception_;

  TaskSetBase* parent_;

  // This mutex guards modifications/use of the intusive linked list between head_ and tail_
  std::mutex mtx_;
  TaskSetBase* head_{nullptr};
  TaskSetBase* tail_{nullptr};

  // prev_ and next_ are links in our *parent's* intrusive linked list.
  TaskSetBase* prev_{nullptr};
  TaskSetBase* next_{nullptr};
};

} // namespace dispenso
