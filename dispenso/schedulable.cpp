/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/schedulable.h>

namespace dispenso {

NewThreadInvoker::ThreadWaiter& NewThreadInvoker::getWaiter() {
  static ThreadWaiter waiter;
  return waiter;
}

} // namespace dispenso
