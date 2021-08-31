// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include "thread_pool.h"

#include <iostream>

namespace dispenso {

#if !defined(DISPENSO_POLL_PERIOD_US)
#define DISPENSO_POLL_PERIOD_US 200
#endif // DISPENSO_POLL_PERIOD_US

void ThreadPool::threadLoop(std::atomic<bool>& running) {
  using namespace std::chrono_literals;

  constexpr auto kSleepDuration = std::chrono::microseconds(DISPENSO_POLL_PERIOD_US);
  constexpr int kBackoffYield = 50;
  constexpr int kBackoffSleep = kBackoffYield + 5;
  OnceFunction next;

  int failCount = 0;

  detail::PerPoolPerThreadInfo::registerPool(this);

  moodycamel::ConsumerToken token(work_);
  while (running.load(std::memory_order_relaxed) ||
         workRemaining_.load(std::memory_order_relaxed)) {
    while (work_.try_dequeue(token, next)) {
      executeNext(std::move(next));
      failCount = 0;
    }
    ++failCount;
    detail::cpuRelax();
    if (failCount > kBackoffSleep) {
      std::this_thread::sleep_for(kSleepDuration);
    } else if (failCount > kBackoffYield) {
      std::this_thread::yield();
    }
  }
}

void ThreadPool::resize(ssize_t sn) {
  assert(sn > 0);
  size_t n = static_cast<size_t>(sn);

  std::lock_guard<std::mutex> lk(threadsMutex_);
  if (n < threads_.size()) {
    for (size_t i = n; i < threads_.size(); ++i) {
      threads_[i].running.store(false, std::memory_order_release);
    }
    for (size_t i = n; i < threads_.size(); ++i) {
      threads_[i].thread.join();
    }
    while (threads_.size() > n) {
      threads_.pop_back();
    }
  } else if (n > threads_.size()) {
    for (size_t i = threads_.size(); i < n; ++i) {
      threads_.emplace_back();
      auto& back = threads_.back();
      back.running = true;
      back.thread = std::thread([this, &running = back.running]() { threadLoop(running); });
    }
  }
  poolLoadFactor_.store(static_cast<ssize_t>(n * poolLoadMultiplier_), std::memory_order_relaxed);
  numThreads_.store(sn, std::memory_order_relaxed);
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
    t.running.store(false, std::memory_order_release);
  }
  while (tryExecuteNext()) {
  }
  for (auto& t : threads_) {
    t.thread.join();
  }
}

ThreadPool& globalThreadPool() {
  // It should be illegal to access globalThreadPool after exiting main.
  static ThreadPool pool(std::thread::hardware_concurrency());
  return pool;
}

void resizeGlobalThreadPool(size_t numThreads) {
  globalThreadPool().resize(static_cast<ssize_t>(numThreads));
}

} // namespace dispenso
