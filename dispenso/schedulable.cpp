/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/schedulable.h>

namespace dispenso {

namespace {
std::atomic<uintptr_t> g_waiter(0);
} // namespace

NewThreadInvoker::ThreadWaiter* NewThreadInvoker::getWaiter() {
  uintptr_t waiter;
  // Common case check first
  if ((waiter = g_waiter.load(std::memory_order_acquire)) > 1) {
    return reinterpret_cast<ThreadWaiter*>(waiter);
  }

  uintptr_t exp = 0;
  if (g_waiter.compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
    g_waiter.store(reinterpret_cast<uintptr_t>(new ThreadWaiter()), std::memory_order_release);
    std::atexit(destroyThreadWaiter);
  }

  while ((waiter = g_waiter.load(std::memory_order_acquire)) < 2) {
  }

  return reinterpret_cast<ThreadWaiter*>(waiter);
}

void NewThreadInvoker::destroyThreadWaiter() {
  delete reinterpret_cast<ThreadWaiter*>(g_waiter.load(std::memory_order_acquire));
}

} // namespace dispenso
