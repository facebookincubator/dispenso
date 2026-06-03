/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "task_set.h"

#include <cstdio>

namespace dispenso {

namespace detail {
// 64 depth is pretty ridiculous, but try not to step on anyone's feet.
constexpr int32_t kMaxTasksStackSize = 64;

DISPENSO_THREAD_LOCAL TaskSetBase* g_taskStack[kMaxTasksStackSize];
DISPENSO_THREAD_LOCAL int32_t g_taskStackSize = 0;

void pushThreadTaskSet(TaskSetBase* t) {
#ifndef NDEBUG
  if (g_taskStackSize < 0 || g_taskStackSize >= kMaxTasksStackSize) {
    fprintf(stderr, "TaskSet parent stack index is invalid when pushing: %d\n", g_taskStackSize);
    std::abort();
  }
#endif // NDEBUG
  g_taskStack[g_taskStackSize++] = t;
}
void popThreadTaskSet() {
#ifndef NDEBUG
  if (g_taskStackSize <= 0) {
    fprintf(stderr, "TaskSet parent stack index is invalid when popping: %d\n", g_taskStackSize);
    std::abort();
  }
#endif // NDEBUG
  --g_taskStackSize;
}
} // namespace detail

TaskSetBase* parentTaskSet() {
  using namespace detail;

#ifndef NDEBUG
  if (g_taskStackSize < 0 || g_taskStackSize >= kMaxTasksStackSize) {
    fprintf(stderr, "TaskSet parent stack index is invalid when accessing: %d\n", g_taskStackSize);
    std::abort();
  }
#endif // NDEBUG

  return g_taskStackSize ? g_taskStack[g_taskStackSize - 1] : nullptr;
}

void TaskSetBase::trySetCurrentException() {
#if defined(__cpp_exceptions)
  auto status = kUnset;
  if (guardException_.compare_exchange_strong(status, kSetting, std::memory_order_acq_rel)) {
    exception_ = std::current_exception();
    guardException_.store(kSet, std::memory_order_release);
    canceled_.store(true, std::memory_order_release);
  }
#endif // __cpp_exceptions
}

inline bool TaskSetBase::testAndResetException() {
#if defined(__cpp_exceptions)
  if (guardException_.load(std::memory_order_acquire) == kSet) {
    auto exception = std::move(exception_);
    guardException_.store(kUnset, std::memory_order_release);
    std::rethrow_exception(exception);
  }
#endif // __cpp_exceptions
  return canceled_.load(std::memory_order_acquire);
}

bool ConcurrentTaskSet::wait() {
  // Steal work until our set is unblocked.  Note that this is not the
  // fastest possible way to unblock the current set, but it will alleviate
  // deadlock, and should provide decent throughput for all waiters.

  // The deadlock scenario mentioned goes as follows:  N threads in the
  // ThreadPool.  Each thread is running code that is using TaskSets.  No
  // progress could be made without stealing.

  // Work may be in the central queue or in per-thread rings (via proactive
  // wake). Drain each source fully before switching to avoid oscillation.
  size_t startRing = 0;
  while (outstandingTaskCount_.load(std::memory_order_acquire)) {
    // Drain central queue
    while (pool_.tryExecuteNext()) {
    }
    // Drain rings — work may have been pushed via proactive wake
    while (pool_.tryExecuteNextFromRings(startRing)) {
    }
    // If neither had work, yield and retry
    if (outstandingTaskCount_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  return testAndResetException();
}

bool ConcurrentTaskSet::tryWait(size_t maxToExecute) {
  size_t startRing = 0;
  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExecute) {
    if (pool_.tryExecuteNext()) {
      --maxToExecute;
    } else if (pool_.tryExecuteNextFromRings(startRing)) {
      --maxToExecute;
    } else {
      break;
    }
  }

  // Must check completion prior to checking exceptions, otherwise there could be a case where
  // exceptions are checked, then an exception is propagated, and then we return whether all items
  // have been completed, thus dropping the exception.
  if (outstandingTaskCount_.load(std::memory_order_acquire)) {
    return false;
  }

  return !testAndResetException();
}

moodycamel::ProducerToken TaskSet::makeToken(moodycamel::ConcurrentQueue<OnceFunction>& pool) {
  return moodycamel::ProducerToken(pool);
}

bool TaskSet::wait() {
  // Steal work until our set is unblocked.
  // The deadlock scenario mentioned goes as follows:  N threads in the
  // ThreadPool.  Each thread is running code that is using TaskSets.  No
  // progress could be made without stealing.

  // First drain our own producer token (work we enqueued)
  while (pool_.tryExecuteNextFromProducerToken(token_)) {
  }

  // Then drain central queue, rings, repeat until done.
  size_t startRing = 0;
  while (outstandingTaskCount_.load(std::memory_order_acquire)) {
    while (pool_.tryExecuteNext()) {
    }
    while (pool_.tryExecuteNextFromRings(startRing)) {
    }
    if (outstandingTaskCount_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  return testAndResetException();
}

bool TaskSet::tryWait(size_t maxToExecute) {
  ssize_t maxToExe = static_cast<ssize_t>(maxToExecute);
  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExe) {
    if (pool_.tryExecuteNextFromProducerToken(token_)) {
      --maxToExe;
    } else {
      break;
    }
  }

  // Must check completion prior to checking exceptions, otherwise there could be a case where
  // exceptions are checked, then an exception is propagated, and then we return whether all items
  // have been completed, thus dropping the exception.

  size_t startRing = 0;
  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExe) {
    if (pool_.tryExecuteNext()) {
      --maxToExe;
    } else if (pool_.tryExecuteNextFromRings(startRing)) {
      --maxToExe;
    } else {
      break;
    }
  }

  if (outstandingTaskCount_.load(std::memory_order_acquire)) {
    return false;
  }

  return !testAndResetException();
}

} // namespace dispenso
