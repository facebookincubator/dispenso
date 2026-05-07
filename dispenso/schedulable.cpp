/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>

#include <dispenso/schedulable.h>

namespace dispenso {

NewThreadInvoker::ThreadWaiter* NewThreadInvoker::getWaiter() {
  // Controlled-leak Meyers singleton, mirroring SmallBufferGlobals. The waiter is allocated on
  // first use and never freed; an atexit handler is registered alongside the allocation to block
  // process exit until every outstanding NewThreadInvoker thread has called remove(). See the
  // comment in schedulable.h for why this matters and why we do not delete the waiter at exit.
  static ThreadWaiter* waiter = []() {
    auto* w = new ThreadWaiter();
    std::atexit([]() { getWaiter()->wait(); });
    return w;
  }();
  return waiter;
}

} // namespace dispenso
