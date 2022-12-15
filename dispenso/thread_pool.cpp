/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "thread_pool.h"

#ifdef _WIN32
#include <Windows.h>
#include <timeapi.h>

namespace {
struct OsQuantaSetter {
  OsQuantaSetter() {
    timeBeginPeriod(1);
  }
  ~OsQuantaSetter() {
    timeEndPeriod(1);
  }
};
} // namespace
#else
namespace {
struct OsQuantaSetter {};
} // namespace

#endif // _WIN32

namespace dispenso {
void ThreadPool::PerThreadData::setThread(std::thread&& t) {
  thread_ = std::move(t);
}

void ThreadPool::PerThreadData::stop() {
  running_.store(false, std::memory_order_release);
}

uint32_t ThreadPool::wait(uint32_t currentEpoch) {
  return epochWaiter_.waitFor(currentEpoch, sleepLengthUs_.load(std::memory_order_acquire));
}
void ThreadPool::wake() {
  epochWaiter_.bumpAndWake();
}

inline bool ThreadPool::PerThreadData::running() {
  return running_.load(std::memory_order_acquire);
}

ThreadPool::ThreadPool(size_t n, size_t poolLoadMultiplier)
    : poolLoadMultiplier_(poolLoadMultiplier),
      poolLoadFactor_(static_cast<ssize_t>(n * poolLoadMultiplier)),
      numThreads_(static_cast<ssize_t>(n)) {
  static OsQuantaSetter quantaSetter;
  (void)quantaSetter;
#if defined DISPENSO_DEBUG
  assert(poolLoadMultiplier > 0);
#endif // DISPENSO_DEBUG
  for (size_t i = 0; i < n; ++i) {
    threads_.emplace_back();
    threads_.back().setThread(std::thread([this, &back = threads_.back()]() { threadLoop(back); }));
  }
}

ThreadPool::PerThreadData::~PerThreadData() {}

void ThreadPool::threadLoop(PerThreadData& data) {
  constexpr int kBackoffYield = 50;
  constexpr int kBackoffSleep = kBackoffYield + 5;

  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  OnceFunction next;

  int failCount = 0;
  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken);
  uint32_t epoch = epochWaiter_.current();

  if (enableEpochWaiter_) {
    bool idle = true;
    idleButAwake_.fetch_add(1, std::memory_order_acq_rel);

    while (data.running()) {
      while (true) {
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
        bool got = work_.try_dequeue(ctoken, next);
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
        if (!got) {
          break;
        }
        queuedWork_.fetch_sub(1, std::memory_order_acq_rel);
        if (idle) {
          idle = false;
          idleButAwake_.fetch_sub(1, std::memory_order_acq_rel);
        }
        executeNext(std::move(next));
        failCount = 0;
      }

      if (!idle) {
        idle = true;
        idleButAwake_.fetch_add(1, std::memory_order_acq_rel);
      }

      ++failCount;

      detail::cpuRelax();
      if (failCount > kBackoffSleep) {
        idleButAwake_.fetch_sub(1, std::memory_order_acq_rel);
        epoch = wait(epoch);
        idleButAwake_.fetch_add(1, std::memory_order_acq_rel);
      } else if (failCount > kBackoffYield) {
        std::this_thread::yield();
      }
    }
    idleButAwake_.fetch_sub(1, std::memory_order_acq_rel);
  } else {
    while (data.running()) {
      while (true) {
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
        bool got = work_.try_dequeue(ctoken, next);
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
        if (!got) {
          break;
        }
        executeNext(std::move(next));
        failCount = 0;
      }

      ++failCount;

      detail::cpuRelax();
      if (failCount > kBackoffSleep) {
        epoch = wait(epoch);
      } else if (failCount > kBackoffYield) {
        std::this_thread::yield();
      }
    }
  }
}

void ThreadPool::resizeLocked(ssize_t sn) {
  assert(sn >= 0);
  size_t n = static_cast<size_t>(sn);

  if (n < threads_.size()) {
    for (size_t i = n; i < threads_.size(); ++i) {
      threads_[i].stop();
    }

    while (threads_.size() > n) {
      wake();
      threads_.back().thread_.join();
      threads_.pop_back();
    }

  } else if (n > threads_.size()) {
    for (size_t i = threads_.size(); i < n; ++i) {
      threads_.emplace_back();
      threads_.back().setThread(
          std::thread([this, &back = threads_.back()]() { threadLoop(back); }));
    }
  }
  poolLoadFactor_.store(static_cast<ssize_t>(n * poolLoadMultiplier_), std::memory_order_relaxed);
  numThreads_.store(sn, std::memory_order_relaxed);

  if (!sn) {
    // Pool will run future tasks inline since we have no threads, but we still need to empty
    // current set of tasks
    while (tryExecuteNext()) {
    }
  }
}

ThreadPool::~ThreadPool() {
#if defined DISPENSO_DEBUG
  assert(outstandingTaskSets_.load(std::memory_order_acquire) == 0);
#endif // DISPENSO_DEBUG

  // Strictly speaking, it is unnecessary to lock this in the destructor; however, it could be a
  // useful diagnostic to learn that the mutex is already locked when we reach this point.
  std::unique_lock<std::mutex> lk(threadsMutex_, std::try_to_lock);
  assert(lk.owns_lock());
  for (auto& t : threads_) {
    t.stop();
    wake();
  }

  while (tryExecuteNext()) {
  }

  while (!threads_.empty()) {
    wake();
    threads_.back().thread_.join();
    threads_.pop_back();
  }

  while (tryExecuteNext()) {
  }
}
ThreadPool& globalThreadPool() {
  // It should be illegal to access globalThreadPool after exiting main.
  // We default to hardware threads minus one because the calling thread usually is involved in
  // computation.
  static ThreadPool pool(std::thread::hardware_concurrency() - 1);
  return pool;
}

void resizeGlobalThreadPool(size_t numThreads) {
  globalThreadPool().resize(static_cast<ssize_t>(numThreads));
}

} // namespace dispenso
