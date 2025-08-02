/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include <dispenso/detail/quanta.h>
#include <dispenso/timed_task.h>

namespace dispenso {

TimedTaskScheduler::TimedTaskScheduler(ThreadPriority prio) : priority_(prio) {
  thread_ = std::thread([this, prio]() {
    detail::registerFineSchedulerQuanta();
    if (!setCurrentThreadPriority(prio)) {
      std::cerr << "Couldn't set thread priority" << std::endl;
    }
    timeQueueRunLoop();
  });
}
TimedTaskScheduler::~TimedTaskScheduler() {
  {
    std::lock_guard<std::mutex> lk(queueMutex_);
    running_ = false;
  }
  epoch_.bumpAndWake();
  thread_.join();
}

void TimedTaskScheduler::kickOffTask(std::shared_ptr<detail::TimedTaskImpl> next, double curTime) {
  size_t remaining = next->timesToRun.fetch_sub(1, std::memory_order_acq_rel);
  if (remaining == 1) {
    auto* np = next.get();
    np->func(std::move(next));
  } else if (remaining > 1) {
    next->func(next);

    if (next->steady) {
      next->nextAbsTime += next->period;
    } else {
      next->nextAbsTime = curTime + next->period;
    }
    std::lock_guard<std::mutex> lk(queueMutex_);
    tasks_.push(std::move(next));
  }
}

constexpr double kSmallTimeBuffer = 10e-6;

void TimedTaskScheduler::timeQueueRunLoop() {
#if defined(_WIN32)
  constexpr double kSpinYieldBuffer = 1e-3;
  constexpr double kSpinBuffer = 100e-6;
#else
  constexpr double kSpinYieldBuffer = 500e-6;
  constexpr double kSpinBuffer = 50e-6;
#endif // platform
  constexpr double kConvertToUs = 1e6;

  uint32_t curEpoch = epoch_.current();

  while (true) {
    {
      std::unique_lock<std::mutex> lk(queueMutex_);
      if (priority_ != getCurrentThreadPriority()) {
        setCurrentThreadPriority(priority_);
      }

      if (!running_) {
        break;
      }
      if (tasks_.empty()) {
        lk.unlock();
        curEpoch = epoch_.wait(curEpoch);
        continue;
      }
    }
    double curTime = getTime();
    double timeRemaining;
    std::unique_lock<std::mutex> lk(queueMutex_);
    timeRemaining = tasks_.top()->nextAbsTime - curTime;
    if (timeRemaining < kSmallTimeBuffer) {
      auto next = tasks_.top();
      tasks_.pop();
      lk.unlock();

      kickOffTask(std::move(next), curTime);
    } else if (timeRemaining < kSpinBuffer) {
      continue;
    } else if (timeRemaining < kSpinYieldBuffer) {
      lk.unlock();
      std::this_thread::yield();
      continue;
    } else {
      lk.unlock();
      curEpoch = epoch_.waitFor(
          curEpoch, static_cast<uint64_t>((timeRemaining - kSpinBuffer) * kConvertToUs));
    }
  }
}

void TimedTaskScheduler::addTimedTask(std::shared_ptr<detail::TimedTaskImpl> task) {
  double curTime = getTime();
  double timeRemaining;
  timeRemaining = task->nextAbsTime - curTime;
  if (timeRemaining < kSmallTimeBuffer) {
    kickOffTask(std::move(task), curTime);
  } else {
    std::lock_guard<std::mutex> lk(queueMutex_);
    tasks_.push(std::move(task));
  }
  epoch_.bumpAndWake();
}

TimedTaskScheduler& globalTimedTaskScheduler() {
  static TimedTaskScheduler scheduler;
  return scheduler;
}

} // namespace dispenso
