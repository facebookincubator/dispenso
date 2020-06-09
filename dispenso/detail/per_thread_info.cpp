// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/detail/per_thread_info.h>

namespace dispenso {
namespace detail {

DISPENSO_THREAD_LOCAL PerThreadInfo PerPoolPerThreadInfo::info_;

} // namespace detail
} // namespace dispenso
