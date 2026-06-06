/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file task_set.h
 * @ingroup group_core
 * A file providing TaskSet and ConcurrentTaskSet.  These interfaces allow the user to
 * submit/schedule multiple closures and then wait on them.
 **/

#pragma once

namespace dispenso {
enum class ParentCascadeCancel { kOff, kOn };

/**
 * Hint to ConcurrentTaskSet about how much work each scheduled task does.
 * Affects how the task set distributes work across pool threads.
 *
 *   kHeavy (default) — Each task does meaningful work — typically anything
 *                      taking microseconds or longer (e.g. a parallel_for
 *                      chunk, a tree-build subproblem, image-tile processing,
 *                      a numerical kernel). Choose kHeavy if you want best
 *                      scaling across many cores, or if many threads will
 *                      concurrently submit tasks (e.g. fork-join recursion).
 *
 *   kLightweight     — Each task does very little work and you are submitting
 *                      a large number of them, where submission cost matters
 *                      more than scaling. Typical: short callbacks, small
 *                      counter updates, simple per-element operations from a
 *                      single producer.
 *
 * If unsure, leave at kHeavy. Picking kHeavy for genuinely tiny tasks costs
 * a small constant overhead per submission; picking kLightweight for heavy
 * parallel work can leave many cores idle.
 **/
// Implementation note: kHeavy routes through the pool's per-group steal rings
// (schedulePlaced / scheduleBulkPlaced), which avoids consumer-side contention
// on the central moodycamel queue at high core counts. kLightweight routes
// through the central queue, which has lower per-task overhead but does not
// scale past ~100 contending consumers.
enum class TaskCost { kLightweight, kHeavy };
} // namespace dispenso

#include <dispenso/detail/task_set_impl.h>

namespace dispenso {

constexpr ssize_t kDefaultStealingMultiplier = 4;

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
class TaskSet : public TaskSetBase {
 public:
  /**
   * Construct a TaskSet with the given backing pool.
   *
   * @param p The backing pool for this TaskSet
   * @param registerForParentCancel Whether to register for parent cancellation cascade.
   * @param stealingLoadMultiplier An over-load factor.  If this factor of load is reached by the
   * underlying pool, scheduled tasks may run immediately in the calling thread.
   **/
  TaskSet(
      ThreadPool& p,
      ParentCascadeCancel registerForParentCancel,
      ssize_t stealingLoadMultiplier = kDefaultStealingMultiplier)
      : TaskSetBase(p, registerForParentCancel, stealingLoadMultiplier),
        token_(makeToken(p.work_)) {}

  /** Construct a TaskSet with default options. @param p The backing pool. */
  TaskSet(ThreadPool& p) : TaskSet(p, ParentCascadeCancel::kOff, kDefaultStealingMultiplier) {}
  /** Construct a TaskSet with custom load multiplier. @param p The backing pool. @param
   * stealingLoadMultiplier The over-load factor. */
  TaskSet(ThreadPool& p, ssize_t stealingLoadMultiplier)
      : TaskSet(p, ParentCascadeCancel::kOff, stealingLoadMultiplier) {}

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
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f) {
    if (DISPENSO_EXPECT(canceled(), false)) {
      return;
    }
    if (outstandingTaskCount_.load(std::memory_order_relaxed) > taskSetLoadFactor_) {
      f();
    } else {
      pool_.schedule(token_, packageTask(std::forward<F>(f)));
    }
  }

  /**
   * Schedule a functor for execution on the underlying pool.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   * @param fq Tag to force queuing instead of potential inline execution.
   *
   * @note If <code>f</code> can throw exceptions, then exceptions will be caught on the running
   * thread and best-effort propagated to the <code>ConcurrentTaskSet</code>, where the first one
   * from the set is rethrown in <code>wait</code>.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag fq) {
    pool_.schedule(token_, packageTask(std::forward<F>(f)), fq);
  }

  /**
   * Schedule multiple functors for execution on the underlying pool in bulk.
   * This is more efficient than calling schedule() multiple times when you have many tasks to
   * submit, as it reduces atomic contention and allows for better thread wakeup behavior.
   *
   * @param count The number of functors to schedule.
   * @param gen A generator functor that takes an index and returns a functor to execute.
   *            gen(i) will be called for i in [0, count) to produce each task.
   *
   * @note Work is processed in chunks, interleaving enqueue and inline execution based on the
   *       task set's load factor, preventing both pool thread starvation and excessive overhead.
   **/
  template <typename Generator>
  void scheduleBulk(size_t count, Generator&& gen) {
    scheduleBulkImpl(count, std::forward<Generator>(gen), &token_);
  }

  /**
   * Schedule multiple functors for execution on the underlying pool in bulk, forcing each
   * task to be queued rather than potentially run inline on the calling thread.
   *
   * @param count The number of functors to schedule.
   * @param gen A generator functor that takes an index and returns a functor to execute.
   *            gen(i) will be called for i in [0, count) to produce each task.
   **/
  template <typename Generator>
  void scheduleBulk(size_t count, Generator&& gen, ForceQueuingTag) {
    scheduleBulkImplForceQueue(count, std::forward<Generator>(gen), &token_);
  }

  /**
   * Wait for all currently scheduled functors to finish execution.  If exceptions are thrown
   * during execution of the set of tasks, <code>wait</code> will propagate the first exception.
   *
   * @return true if the TaskSet was canceled, false otherwise
   **/
  DISPENSO_DLL_ACCESS bool wait();

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
   * returning, and <code>false</code> otherwise.  This includes returning false if the TaskSet was
   * cancelled.
   **/
  DISPENSO_DLL_ACCESS bool tryWait(size_t maxToExecute);

  /**
   * Set the TaskSet to canceled state.  No unexecuted tasks will execute once this is set.
   * Already executing tasks may check canceled() status to exit early.
   *
   **/
  void cancel() {
    TaskSetBase::cancel();
  }

  /**
   * Check the canceled status of the TaskSet.
   *
   * @return a boolean indicating whether or not the TaskSet has been canceled.
   **/
  bool canceled() const {
    return TaskSetBase::canceled();
  }

  /**
   * Destroy the TaskSet, first waiting for all currently scheduled functors to
   * finish execution.
   **/
  ~TaskSet() {
    wait();
  }

 private:
  DISPENSO_DLL_ACCESS moodycamel::ProducerToken makeToken(
      moodycamel::ConcurrentQueue<OnceFunction>& pool);

  moodycamel::ProducerToken token_;

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;
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
class ConcurrentTaskSet : public TaskSetBase {
 public:
  /**
   * Construct a ConcurrentTaskSet with the given backing pool.
   *
   * @param pool The backing pool for this ConcurrentTaskSet
   * @param registerForParentCancel Whether to register for parent cancellation cascade.
   * @param stealingLoadMultiplier An over-load factor.  If this factor of load is reached by the
   * underlying pool, scheduled tasks may run immediately in the calling thread.
   * @param cost Hint about per-task cost; see TaskCost.
   **/
  ConcurrentTaskSet(
      ThreadPool& pool,
      ParentCascadeCancel registerForParentCancel,
      ssize_t stealingLoadMultiplier = kDefaultStealingMultiplier,
      TaskCost cost = TaskCost::kHeavy)
      : TaskSetBase(pool, registerForParentCancel, stealingLoadMultiplier), cost_(cost) {}

  /** Construct a ConcurrentTaskSet with default options. @param p The backing pool. */
  ConcurrentTaskSet(ThreadPool& p)
      : ConcurrentTaskSet(p, ParentCascadeCancel::kOff, kDefaultStealingMultiplier) {}
  /** Construct a ConcurrentTaskSet with a task-cost hint. @param p The backing pool. @param cost
   * Hint about per-task cost; see TaskCost. */
  ConcurrentTaskSet(ThreadPool& p, TaskCost cost)
      : ConcurrentTaskSet(p, ParentCascadeCancel::kOff, kDefaultStealingMultiplier, cost) {}
  /** Construct a ConcurrentTaskSet with custom load multiplier. @param p The backing pool. @param
   * stealingLoadMultiplier The over-load factor. */
  ConcurrentTaskSet(ThreadPool& p, ssize_t stealingLoadMultiplier)
      : ConcurrentTaskSet(p, ParentCascadeCancel::kOff, stealingLoadMultiplier) {}
  /** Construct a ConcurrentTaskSet with a task-cost hint and custom load multiplier. */
  ConcurrentTaskSet(ThreadPool& p, TaskCost cost, ssize_t stealingLoadMultiplier)
      : ConcurrentTaskSet(p, ParentCascadeCancel::kOff, stealingLoadMultiplier, cost) {}

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
   * @param skipRecheck A poweruser knob that says that if we don't have enough outstanding tasks to
   * immediately work steal, we should bypass the similar check in the ThreadPool.
   *
   * @param poolRecursiveLoadFactor Controls how aggressively pool threads
   *   inline work.  Lower values (1.0) cause more inlining (pipeline-like),
   *   higher values (3.0) cause more distribution (graph-like).
   *
   * @note If <code>f</code> can throw exceptions, then <code>schedule</code> may throw if the task
   * is run inline.  Otherwise, exceptions will be caught on the running thread and best-effort
   * propagated to the <code>ConcurrentTaskSet</code>, where the first one from the set is rethrown
   * in <code>wait</code>.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(
      F&& f,
      bool skipRecheck = false,
      float poolRecursiveLoadFactor = kDefaultPoolRecursiveLoadFactor) {
    if (cost_ == TaskCost::kHeavy) {
      schedulePlaced(std::forward<F>(f), skipRecheck, poolRecursiveLoadFactor);
      return;
    }
    // Combined inline decision (mirrors scheduleBulkImpl):
    // 1. TaskSet-level overload
    // 2. Pool-recursive: pool threads inline when workRemaining_ exceeds
    //    numThreads * poolRecursiveLoadFactor
    // 3. Pool global: non-recursive callers inline at the loose poolLoadFactor_
    // After this check, use ForceQueuingTag to skip the redundant check
    // in ThreadPool::schedule.
    if (outstandingTaskCount_.load(std::memory_order_relaxed) > taskSetLoadFactor_ &&
        DISPENSO_EXPECT(!canceled(), true) && detail::PerPoolPerThreadInfo::canInlineSchedule()) {
      detail::InlineDepthGuard depthGuard;
      f();
      return;
    }
    if (!skipRecheck) {
      ssize_t curWork = pool_.workRemaining_.load(std::memory_order_relaxed);
      ssize_t quickFactor =
          static_cast<ssize_t>(static_cast<float>(pool_.numThreads()) * poolRecursiveLoadFactor);
      if ((detail::PerPoolPerThreadInfo::isPoolRecursive(&pool_) && curWork > quickFactor) ||
          curWork > pool_.poolLoadFactor_.load(std::memory_order_relaxed)) {
        if (!detail::PerPoolPerThreadInfo::canInlineSchedule()) {
          pool_.schedule(packageTask(std::forward<F>(f)), ForceQueuingTag());
          return;
        }
        detail::InlineDepthGuard depthGuard;
        f();
        return;
      }
    }
    pool_.schedule(packageTask(std::forward<F>(f)), ForceQueuingTag());
  }

  /**
   * Schedule a functor for execution on the underlying pool.
   *
   * @param f A functor matching signature <code>void()</code>.  Best performance will come from
   * passing lambdas, other concrete functors, or <code>OnceFunction</code>, but
   * <code>std::function</code> or similarly type-erased objects will also work.
   * @param fq Tag to force queuing instead of potential inline execution.
   *
   * @note If <code>f</code> can throw exceptions, then exceptions will be caught on the running
   * thread and best-effort propagated to the <code>ConcurrentTaskSet</code>, where the first one
   * from the set is rethrown in <code>wait</code>.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag fq) {
    if (cost_ == TaskCost::kHeavy) {
      pool_.schedulePlaced(packageTask(std::forward<F>(f)), fq);
      return;
    }
    pool_.schedule(packageTask(std::forward<F>(f)), fq);
  }

  /**
   * Schedule multiple functors for execution on the underlying pool in bulk.
   * This is more efficient than calling schedule() multiple times when you have many tasks to
   * submit, as it reduces atomic contention and allows for better thread wakeup behavior.
   *
   * @param count The number of functors to schedule.
   * @param gen A generator functor that takes an index and returns a functor to execute.
   *            gen(i) will be called for i in [0, count) to produce each task.
   *
   * @note Work is processed in chunks, interleaving enqueue and inline execution based on the
   *       task set's load factor, preventing both pool thread starvation and excessive overhead.
   **/
  template <typename Generator>
  void scheduleBulk(size_t count, Generator&& gen) {
    if (cost_ == TaskCost::kHeavy) {
      scheduleBulkImplPlaced(count, std::forward<Generator>(gen));
      return;
    }
    scheduleBulkImpl(count, std::forward<Generator>(gen), nullptr);
  }

  /**
   * Schedule multiple functors for execution on the underlying pool in bulk, forcing each
   * task to be queued rather than potentially run inline on the calling thread.
   *
   * @param count The number of functors to schedule.
   * @param gen A generator functor that takes an index and returns a functor to execute.
   *            gen(i) will be called for i in [0, count) to produce each task.
   **/
  template <typename Generator>
  void scheduleBulk(size_t count, Generator&& gen, ForceQueuingTag) {
    scheduleBulkImplForceQueue(count, std::forward<Generator>(gen), nullptr);
  }

  /**
   * Wait for all currently scheduled functors to finish execution.  If exceptions are thrown
   * during execution of the set of tasks, <code>wait</code> will propagate the first exception.
   **/
  DISPENSO_DLL_ACCESS bool wait();

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
   * returning, and <code>false</code> otherwise (including cancelled cases).
   **/
  DISPENSO_DLL_ACCESS bool tryWait(size_t maxToExecute);

  /**
   * Set the ConcurrentTaskSet to canceled state.  No unexecuted tasks will execute once this is
   * set.  Already executing tasks may check canceled() status to exit early.
   *
   * @note This will be reset automatically by wait.
   **/
  void cancel() {
    TaskSetBase::cancel();
  }

  /**
   * Check the canceled status of the ConcurrentTaskSet.
   *
   * @return a boolean indicating whether or not the ConcurrentTaskSet has been canceled.
   **/
  bool canceled() const {
    return TaskSetBase::canceled();
  }

  /**
   * Destroy the ConcurrentTaskSet, first waiting for all currently scheduled functors to
   * finish execution.
   **/
  ~ConcurrentTaskSet() {
    wait();
  }

 private:
  bool tryExecuteNext() {
    return pool_.tryExecuteNext();
  }

  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedulePlaced(
      F&& f,
      bool skipRecheck = false,
      float poolRecursiveLoadFactor = kDefaultPoolRecursiveLoadFactor) {
    ssize_t placedThreshold = std::max(pool_.numThreads() + 1, taskSetLoadFactor_ / 2);
    if (outstandingTaskCount_.load(std::memory_order_relaxed) > placedThreshold &&
        DISPENSO_EXPECT(!canceled(), true) && detail::PerPoolPerThreadInfo::canInlineSchedule()) {
      detail::InlineDepthGuard depthGuard;
      f();
      return;
    }
    if (!skipRecheck) {
      ssize_t curWork = pool_.workRemaining_.load(std::memory_order_relaxed);
      ssize_t quickFactor =
          static_cast<ssize_t>(static_cast<float>(pool_.numThreads()) * poolRecursiveLoadFactor);
      if ((detail::PerPoolPerThreadInfo::isPoolRecursive(&pool_) && curWork > quickFactor) ||
          curWork > pool_.poolLoadFactor_.load(std::memory_order_relaxed)) {
        if (!detail::PerPoolPerThreadInfo::canInlineSchedule()) {
          pool_.schedulePlaced(packageTask(std::forward<F>(f)), ForceQueuingTag());
          return;
        }
        detail::InlineDepthGuard depthGuard;
        f();
        return;
      }
    }
    pool_.schedulePlaced(packageTask(std::forward<F>(f)), ForceQueuingTag());
  }

  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedulePlaced(F&& f, ForceQueuingTag) {
    pool_.schedulePlaced(packageTask(std::forward<F>(f)), ForceQueuingTag());
  }

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;

  friend class detail::LimitGatedScheduler;

  TaskCost cost_{TaskCost::kHeavy};
};

/**
 * Get access to the parent task set that scheduled the currently running code. nullptr if called
 * outside the context of a (Concurrent)TaskSet schedule.
 *
 **/
DISPENSO_DLL_ACCESS TaskSetBase* parentTaskSet();

} // namespace dispenso
