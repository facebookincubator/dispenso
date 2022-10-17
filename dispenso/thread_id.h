/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file Utilities for getting a unique thread identifier
 **/

#pragma once

#include <dispenso/platform.h>

namespace dispenso {

/**
 * Get the current thread's identifier, unique within the current process.
 *
 * @return An integer representing the current thread.
 *
 * @note Thread IDs are assumed to not be reused over the lifetime of a process, but this should
 * still enable processes running for thousands of years, even with very poor spawn/kill thread
 * patterns.
 *
 * @note If thread ID is needed for cross-process synchronization, one must fall back on
 * system-specific thread IDs.
 **/
DISPENSO_DLL_ACCESS uint64_t threadId();

} // namespace dispenso
