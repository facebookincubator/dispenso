/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file timed_task.h
 * @ingroup group_async
 * Utilities for delaying a task until a future time and periodic scheduling.
 **/

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <queue>

#include <dispenso/detail/timed_task_impl.h>
#include <dispenso/priority.h>
#include <dispenso/timing.h>

namespace dispenso {

/**
 * For periodic tasks, this type will describe the behavior of how tasks are scheduled
 **/
enum class TimedTaskType {
  kNormal ///< Schedule the next task at period plus the time the first or most recent task ran
  ,
  kSteady ///< Schedule the next task at period plus the time the first or most recent task was
          ///< scheduled for
};

/**
 * A timed task type.  This encapsulates a function that will run at a time scheduled in the future,
 * and optionally periodically for a specified number of times.
 **/

class TimedTask {
 public:
  TimedTask(const TimedTask&) = delete;
  /** Move constructor. */
  TimedTask(TimedTask&& other) : impl_(std::move(other.impl_)) {}

  TimedTask& operator=(const TimedTask&) = delete;
  TimedTask& operator=(TimedTask&& other) {
    impl_ = std::move(other.impl_);
    return *this;
  }

  /**
   * Cancel the task.  No further runs to the underlying function will occur (though any calls in
   * progress will complete).
   **/
  void cancel() {
    impl_->timesToRun.store(0, std::memory_order_release);
    impl_->flags.fetch_or(detail::kFFlagsCancelled, std::memory_order_release);
  }

  /**
   * Detach this task so that it cannot block in destruction, and destruction does not cancel
   * further runs to the underlying function.  This is only to be used in cases where the underlying
   * schedulable and any function resources are expected to outlive any other copy of the task.
   **/
  void detach() {
    impl_->flags.fetch_or(detail::kFFlagsDetached, std::memory_order_release);
  }

  /**
   * See how many calls there have been to the underlying function have completed thus far.
   *
   * @return the count of calls made this far to the underlying function.
   **/
  size_t calls() const {
    return impl_->count.load(std::memory_order_acquire);
  }

  /**
   * Destroy the timed task.  If detach() has not been called, this will cancel further calls to the
   * underlying scheduled function, and will wait if necessary until the function is no longer in
   * progress.  This is to naturally protect resources that may be used by the function.  If
   * detach() was called, the destructor will not block.
   *
   **/
  ~TimedTask() {
    if (!impl_ || impl_->flags.load(std::memory_order_acquire) & detail::kFFlagsDetached) {
      return;
    }
    cancel();
    while (impl_->inProgress.load(std::memory_order_acquire)) {
    }
    // Now we can safely destroy the underlying function.  We do this here because we can't risk
    // that func may call code in it's destructor that may no longer be relevant after this
    // TimedTask destructor completes.
    impl_->func = {};
  }

 private:
  template <typename Schedulable, typename F>
  TimedTask(
      Schedulable& sched,
      F&& f,
      double nextRunAbs,
      double period = 0.0,
      size_t timesToRun = 1,
      TimedTaskType type = TimedTaskType::kNormal)
      : impl_(
            detail::make_shared<detail::TimedTaskImpl>(
                timesToRun,
                nextRunAbs,
                period,
                std::forward<F>(f),
                sched,
                type == TimedTaskType::kSteady)) {}

  std::shared_ptr<detail::TimedTaskImpl> impl_;

  friend class TimedTaskScheduler;
}; // namespace dispenso

/**
 * A timed-task scheduler running on a single thread. This allows multiple schedulers if necessary,
 * if e.g. you wish to schedule against InlineInvoker backing schedulable and you're running
 * a lot of relatively quick timed tasks.Most people should just use the global timed task
 * scheduler.
 **/
class TimedTaskScheduler {
 public:
  /**
   * Create a TimedTaskScheduler with specified priority
   *
   * @param priority The priority to set.  For highest level of periodicity accuracy, use kRealtime.
   * @note Priorities above kNormal should be used sparingly, and only short-running tasks should be
   * run inline to avoid bad OS responsivity.  When using a ThreadPool Schedulable, tasks can be
   * long running.
   **/
  DISPENSO_DLL_ACCESS explicit TimedTaskScheduler(
      ThreadPriority priority = ThreadPriority::kNormal);
  DISPENSO_DLL_ACCESS ~TimedTaskScheduler();

  /**
   * Set the priority for the backing thread.  See note for constructor.
   **/
  void setPriority(ThreadPriority priority) {
    std::lock_guard<std::mutex> lk(queueMutex_);
    priority_ = priority;
  }

  /**
   * Schedule a task
   *
   * @param sched A backing schedulable, such as ImmediateInvoker, NewThreadInvoker, TaskSet, or
   *        ThreadPool to run the function in when it is scheduled.
   * @param func A bool() function to run when scheduled.  The function may return true to indicate
   *        it should continue to be scheduled, or false to cancel.
   * @param nextRunAbs The absolute time to run the function.  Time scale is expected to match
   *        getTime() for absolute times.
   * @param period The period in seconds.
   * @param timesToRun The number of times to run the function.  After that number, the function
   *        will not be called again.
   * @param type The type of periodicity (if any).
   **/
  template <typename Schedulable, typename F>
  TimedTask schedule(
      Schedulable& sched,
      F&& func,
      double nextRunAbs,
      double period = 0.0,
      size_t timesToRun = 1,
      TimedTaskType type = TimedTaskType::kNormal) {
    TimedTask task(sched, std::forward<F>(func), nextRunAbs, period, timesToRun, type);
    addTimedTask(task.impl_);
    return task;
  }

  /**
   * Schedule a task to run once at a time in the future
   *
   * @param sched A backing schedulable, such as ImmediateInvoker, NewThreadInvoker, TaskSet, or
   *        ThreadPool to run the function in when it is scheduled.
   * @param func A bool() function to run when scheduled.  The function may return true to indicate
   *        it should continue to be scheduled, or false to cancel.
   * @param timeInFuture the amount of time from current at which to schedule the function
   **/
  template <typename Schedulable, typename Rep, typename Period, typename F>
  TimedTask
  schedule(Schedulable& sched, F&& func, const std::chrono::duration<Rep, Period>& timeInFuture) {
    return schedule(sched, std::forward<F>(func), toNextRun(timeInFuture));
  }

  /**
   * Schedule a task to run once at a time in the future
   *
   * @param sched A backing schedulable, such as ImmediateInvoker, NewThreadInvoker, TaskSet, or
   *        ThreadPool to run the function in when it is scheduled.
   * @param func A bool() function to run when scheduled.  The function may return true to indicate
   *        it should continue to be scheduled, or false to cancel.
   * @param nextRunTime An absolute time to run the function.  If in the past, func will run
   *        immediately.
   **/
  template <typename Schedulable, typename Clock, typename Duration, typename F>
  TimedTask schedule(
      Schedulable& sched,
      F&& func,
      const std::chrono::time_point<Clock, Duration>& nextRunTime) {
    return schedule(sched, std::forward<F>(func), toNextRun(nextRunTime));
  }

  /**
   * Schedule a task to run periodically
   *
   * @param sched A backing schedulable, such as ImmediateInvoker, NewThreadInvoker, TaskSet, or
   *        ThreadPool to run the function in when it is scheduled.
   * @param func A bool() function to run when scheduled.  The function may return true to indicate
   *        it should continue to be scheduled, or false to cancel.
   * @param timeInFuture the amount of time from current at which to schedule the function
   * @param period The period defining the run frequency
   * @param timesToRun The number of times to run the function.  After that number, the function
   *        will not be called again.
   * @param type The type of periodicity (if any).
   **/
  template <typename Schedulable, typename Rep, typename Period, typename F>
  TimedTask schedule(
      Schedulable& sched,
      F&& func,
      const std::chrono::duration<Rep, Period>& timeInFuture,
      const std::chrono::duration<Rep, Period>& period,
      size_t timesToRun = std::numeric_limits<size_t>::max(),
      TimedTaskType type = TimedTaskType::kNormal) {
    return schedule(
        sched, std::forward<F>(func), toNextRun(timeInFuture), toPeriod(period), timesToRun, type);
  }

  /**
   * Schedule a task to run periodically
   *
   * @param sched A backing schedulable, such as ImmediateInvoker, NewThreadInvoker, TaskSet, or
   *        ThreadPool to run the function in when it is scheduled.
   * @param func A bool() function to run when scheduled.  The function may return true to indicate
   *        it should continue to be scheduled, or false to cancel.
   * @param nextRunTime An absolute time to run the function.  If in the past, func will run
   *        immediately.
   * @param period The period defining the run frequency
   * @param timesToRun The number of times to run the function.  After that number, the function
   *        will not be called again.
   * @param type The type of periodicity (if any).
   **/
  template <
      typename Schedulable,
      typename Rep,
      typename Period,
      typename Clock,
      typename Duration,
      typename F>
  TimedTask schedule(
      Schedulable& sched,
      F&& func,
      const std::chrono::time_point<Clock, Duration>& nextRunTime,
      const std::chrono::duration<Rep, Period>& period,
      size_t timesToRun = std::numeric_limits<size_t>::max(),
      TimedTaskType type = TimedTaskType::kNormal) {
    return schedule(
        sched, std::forward<F>(func), toNextRun(nextRunTime), toPeriod(period), timesToRun, type);
  }

 private:
  template <class Rep, class Period>
  static double toNextRun(const std::chrono::duration<Rep, Period>& timeInFuture) {
    return getTime() + std::chrono::duration<double>(timeInFuture).count();
  }

  template <typename Clock, typename Duration>
  static double toNextRun(const std::chrono::time_point<Clock, Duration>& nextRunTime) {
    auto curTime = Clock::now();
    return toNextRun(nextRunTime - curTime);
  }

  template <class Rep, class Period>
  static double toPeriod(const std::chrono::duration<Rep, Period>& period) {
    return std::chrono::duration<double>(period).count();
  }
  DISPENSO_DLL_ACCESS void addTimedTask(std::shared_ptr<detail::TimedTaskImpl> task);
  void timeQueueRunLoop();

  void kickOffTask(std::shared_ptr<detail::TimedTaskImpl> next, double curTime);

  struct Compare {
    bool operator()(
        const std::shared_ptr<detail::TimedTaskImpl>& a,
        const std::shared_ptr<detail::TimedTaskImpl>& b) const {
      return a->nextAbsTime > b->nextAbsTime;
    }
  };

  // TODO(bbudge): Consider lock-free priority queue implementation.  I'd expect it to be minimally
  // beneficial for this use case though... timed tasks should rarely be super-high contention.
  std::mutex queueMutex_;
  std::priority_queue<
      std::shared_ptr<detail::TimedTaskImpl>,
      std::vector<std::shared_ptr<detail::TimedTaskImpl>>,
      Compare>
      tasks_;
  bool running_{true};
  detail::EpochWaiter epoch_;
  std::thread thread_;
  ThreadPriority priority_;
};

/**
 * Access the global timed task scheduler.  Most applications should only require one scheduler.
 *
 * @return A reference to the single instance global timed task scheduler.
 **/
DISPENSO_DLL_ACCESS TimedTaskScheduler& globalTimedTaskScheduler();

} // namespace dispenso
