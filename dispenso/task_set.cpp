// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include "task_set.h"

namespace dispenso {

void ConcurrentTaskSet::wait() {
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

#if defined(__cpp_exceptions)
  if (guardException_.exchange(false, std::memory_order_acquire)) {
    std::rethrow_exception(exception_);
  }
#endif // __cpp_exceptions
}

void TaskSet::wait() {
  // Steal work until our set is unblocked.
  // The deadlock scenario mentioned goes as follows:  N threads in the
  // ThreadPool.  Each thread is running code that is using TaskSets.  No
  // progress could be made without stealing.
  while (pool_.tryExecuteNextFromProducerToken(token_)) {
  }
  while (outstandingTaskCount_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

#if defined(__cpp_exceptions)
  if (guardException_.load(std::memory_order_acquire)) {
    guardException_.store(false, std::memory_order_release);
    std::rethrow_exception(exception_);
  }
#endif // __cpp_exceptions
}

} // namespace dispenso
