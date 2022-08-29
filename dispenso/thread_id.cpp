/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_id.h>

namespace dispenso {

std::atomic<uint64_t> nextThread{0};
constexpr uint64_t kInvalidThread = std::numeric_limits<uint64_t>::max();
DISPENSO_THREAD_LOCAL uint64_t currentThread = kInvalidThread;

uint64_t threadId() {
  if (currentThread == kInvalidThread) {
    currentThread = nextThread.fetch_add(uint64_t{1}, std::memory_order_relaxed);
  }
  return currentThread;
}

} // namespace dispenso
