/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/per_thread_info.h>

namespace dispenso {
namespace detail {

namespace {
DISPENSO_THREAD_LOCAL PerThreadInfo g_perThreadInfo;
}
PerThreadInfo& PerPoolPerThreadInfo::info() {
  return g_perThreadInfo;
}

} // namespace detail
} // namespace dispenso
