/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file timing.h
 * @ingroup group_util
 * Utilities for getting the current time.
 **/

#pragma once

#include <dispenso/platform.h>

namespace dispenso {

/**
 * Get elapsed time in seconds since program start.
 *
 * Uses high-resolution timing when available (e.g., RDTSC on x86).
 *
 * @return Elapsed time in seconds as a double.
 */
DISPENSO_DLL_ACCESS double getTime();

} // namespace dispenso
