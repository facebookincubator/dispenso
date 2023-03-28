/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "task_set.h"

namespace dispenso {

namespace detail {
// 64 depth is pretty ridiculous, but try not to step on anyone's feet.
constexpr size_t kMaxTasksStackSize = 64;

DISPENSO_THREAD_LOCAL TaskSetBase* g_taskStack[kMaxTasksStackSize];
DISPENSO_THREAD_LOCAL size_t g_taskStackSize = 0;

void pushThreadTaskSet(TaskSetBase* t) {
  assert(g_taskStackSize < kMaxTasksStackSize - 1);
  g_taskStack[g_taskStackSize++] = t;
}
void popThreadTaskSet() {
  --g_taskStackSize;
}
} // namespace detail

TaskSetBase* parentTaskSet() {
  using namespace detail;
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
  while (outstandingTaskCount_.load(std::memory_order_acquire)) {
    if (!pool_.tryExecuteNext()) {
      std::this_thread::yield();
    }
  }

  return testAndResetException();
}

bool ConcurrentTaskSet::tryWait(size_t maxToExecute) {
  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExecute--) {
    if (!pool_.tryExecuteNext()) {
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
  while (pool_.tryExecuteNextFromProducerToken(token_)) {
  }

  while (outstandingTaskCount_.load(std::memory_order_acquire)) {
    if (!pool_.tryExecuteNext()) {
      std::this_thread::yield();
    }
  }

  return testAndResetException();
}

bool TaskSet::tryWait(size_t maxToExecute) {
  ssize_t maxToExe = static_cast<ssize_t>(maxToExecute);
  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExe--) {
    if (!pool_.tryExecuteNextFromProducerToken(token_)) {
      break;
    }
  }

  // Must check completion prior to checking exceptions, otherwise there could be a case where
  // exceptions are checked, then an exception is propagated, and then we return whether all items
  // have been completed, thus dropping the exception.

  maxToExe = std::max<ssize_t>(0, maxToExe);

  while (outstandingTaskCount_.load(std::memory_order_acquire) && maxToExe--) {
    if (!pool_.tryExecuteNext()) {
      std::this_thread::yield();
    }
  }

  if (outstandingTaskCount_.load(std::memory_order_acquire)) {
    return false;
  }

  return !testAndResetException();
}

} // namespace dispenso
