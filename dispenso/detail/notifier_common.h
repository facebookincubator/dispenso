/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// For fallback path
#include <condition_variable>
#include <mutex>

#if defined(__linux__)
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace dispenso {
namespace detail {
static int futex(
    int* uaddr,
    int futex_op,
    int val,
    const struct timespec* timeout,
    int* /*uaddr2*/,
    int val3) {
  return static_cast<int>(syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr, val3));
}
} // namespace detail
} // namespace dispenso

#elif defined(__MACH__)
#include <mach/mach.h>

#elif defined(_WIN32)

#if (defined(_M_ARM64) || defined(_M_ARM)) && !defined(_ARM_)
#define _ARM_
#elif _WIN64
#define _AMD64_
#elif _WIN32
#define _X86_
#else
#error "No valid windows platform"
#endif // platform

#include <errhandlingapi.h>
#include <synchapi.h>

namespace dispenso {
namespace detail {

constexpr int kErrorTimeoutWin = 0x000005B4;
constexpr unsigned long kInfiniteWin = static_cast<unsigned long>(-1);

} // namespace detail
} // namespace dispenso

#endif // PLATFORM
