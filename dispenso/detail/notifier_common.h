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
#include <Availability.h>
#include <errno.h>
#include <mach/mach_time.h>

// Detect os_sync_wait_on_address availability (macOS 14.4+)
#if defined(__has_include)
#if __has_include(<os/os_sync_wait_on_address.h>)
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 140400
#define DISPENSO_HAS_OS_SYNC 1
#include <os/os_sync_wait_on_address.h>
#endif
#endif
#endif

#ifndef DISPENSO_HAS_OS_SYNC
// __ulock APIs are available since macOS 10.12 (Sierra). On older versions (e.g. PPC/10.5 builds
// for MacPorts), fall through to the std::mutex fallback.
#if !defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
#define DISPENSO_HAS_ULOCK 1
extern "C" int __ulock_wait(uint32_t operation, void* addr, uint64_t value, uint32_t timeout_us);
extern "C" int __ulock_wake(uint32_t operation, void* addr, uint64_t value);

#ifndef UL_COMPARE_AND_WAIT
#define UL_COMPARE_AND_WAIT 1
#endif
#ifndef ULF_WAKE_ALL
#define ULF_WAKE_ALL 0x00000100
#endif
#endif // macOS >= 10.12
#endif // DISPENSO_HAS_OS_SYNC

// DISPENSO_HAS_MAC_FUTEX is set when any mac futex-like API is available.
#if defined(DISPENSO_HAS_OS_SYNC) || defined(DISPENSO_HAS_ULOCK)
#define DISPENSO_HAS_MAC_FUTEX 1
#endif

#ifdef DISPENSO_HAS_MAC_FUTEX
namespace dispenso {
namespace detail {

inline void mac_futex_wait(void* addr, uint64_t expected, size_t size) {
#ifdef DISPENSO_HAS_OS_SYNC
  os_sync_wait_on_address(addr, expected, size, OS_SYNC_WAIT_ON_ADDRESS_NONE);
#else
  (void)size;
  __ulock_wait(UL_COMPARE_AND_WAIT, addr, expected, 0);
#endif
}

inline int mac_futex_wait_for(void* addr, uint64_t expected, size_t size, uint64_t relTimeUs) {
#ifdef DISPENSO_HAS_OS_SYNC
  static mach_timebase_info_data_t sTimebaseInfo = []() {
    mach_timebase_info_data_t i;
    mach_timebase_info(&i);
    return i;
  }();
  uint64_t ns = relTimeUs * 1000;
  uint64_t timeout = ns * sTimebaseInfo.denom / sTimebaseInfo.numer;
  return os_sync_wait_on_address_with_timeout(
      addr, expected, size, OS_SYNC_WAIT_ON_ADDRESS_NONE, OS_CLOCK_MACH_ABSOLUTE_TIME, timeout);
#else
  (void)size;
  // __ulock_wait takes a uint32_t timeout in microseconds, which wraps at ~4295 seconds (~72 min).
  // Timeouts beyond that cause a spurious early wake, which is harmless since callers re-check
  // status in a loop.
  return __ulock_wait(UL_COMPARE_AND_WAIT, addr, expected, static_cast<uint32_t>(relTimeUs));
#endif
}

inline void mac_futex_wake_one(void* addr, size_t size) {
#ifdef DISPENSO_HAS_OS_SYNC
  os_sync_wake_by_address_any(addr, size, OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
  (void)size;
  __ulock_wake(UL_COMPARE_AND_WAIT, addr, 0);
#endif
}

inline void mac_futex_wake_all(void* addr, size_t size) {
#ifdef DISPENSO_HAS_OS_SYNC
  os_sync_wake_by_address_all(addr, size, OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
  (void)size;
  __ulock_wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL, addr, 0);
#endif
}

} // namespace detail
} // namespace dispenso
#endif // DISPENSO_HAS_MAC_FUTEX

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
