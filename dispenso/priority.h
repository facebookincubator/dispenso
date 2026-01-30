/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file priority.h
 * @ingroup group_util
 *
 * Utilities for getting and setting thread priority.  This is an attempt to unify concepts for
 * thread priority usefully across multiple platforms.  For finer control, use platform specific
 * functionality.
 *
 * @note When using higher-than-normal priority, use caution!  Too many threads running at too high
 * priority can have a strong negative impact on the responsivity of the machine.  Prefer to use
 * realtime priority only for short running tasks that need to be very responsively run.
 **/

#pragma once

#include <dispenso/platform.h>

namespace dispenso {

/**
 * A thread priority setting.  Enum values in increasing order of priority.
 **/
enum class ThreadPriority { kLow, kNormal, kHigh, kRealtime };

/**
 * Access the current thread priority as set by setCurrentThreadPriority.
 *
 * @return The priority of the current thread
 *
 * @note If the current thread priority has been set via a platform-specific mechanism, this may
 * return an incorrect value.
 **/
DISPENSO_DLL_ACCESS ThreadPriority getCurrentThreadPriority();

/**
 * Set the current thread's priority
 *
 * @param prio The priority to set to
 *
 * @return true if the priority was modified, false otherwise.
 **/
DISPENSO_DLL_ACCESS bool setCurrentThreadPriority(ThreadPriority prio);

} // namespace dispenso
