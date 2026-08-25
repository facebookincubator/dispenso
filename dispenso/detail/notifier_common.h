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

// Architectures the kernel added after 5.1 are 64-bit-time_t only: they do not
// define __NR_futex at all, just __NR_futex_time64. riscv32 is one. Where both
// exist they are not interchangeable -- SYS_futex takes a 32-bit timespec on
// those targets while SYS_futex_time64 takes a 64-bit one -- so prefer
// SYS_futex and fall back only where it is absent, which is precisely where
// timespec is already 64-bit and the layouts agree.
#if defined(SYS_futex)
#define DISPENSO_SYS_FUTEX SYS_futex
#elif defined(SYS_futex_time64)
#define DISPENSO_SYS_FUTEX SYS_futex_time64
#else
#error "Linux build with neither SYS_futex nor SYS_futex_time64"
#endif

namespace dispenso {
namespace detail {
static int futex(
    int* uaddr,
    int futex_op,
    int val,
    const struct timespec* timeout,
    int* /*uaddr2*/,
    int val3) {
  return static_cast<int>(syscall(DISPENSO_SYS_FUTEX, uaddr, futex_op, val, timeout, uaddr, val3));
}
} // namespace detail
} // namespace dispenso

#elif defined(__FreeBSD__)
#include <errno.h>
#include <sys/types.h>
#include <sys/umtx.h>

#ifndef FUTEX_WAIT_PRIVATE
#define FUTEX_WAIT_PRIVATE 128
#endif
#ifndef FUTEX_WAKE_PRIVATE
#define FUTEX_WAKE_PRIVATE 129
#endif

namespace dispenso {
namespace detail {
static int futex(
    int* uaddr,
    int futex_op,
    int val,
    const struct timespec* timeout,
    int* /*uaddr2*/,
    int val3) {
  (void)val3;
  if (futex_op == FUTEX_WAIT_PRIVATE) {
    // Zero-extend the compare value: the kernel compares the 32-bit value at
    // uaddr zero-extended against the full u_long val, so sign-extending values
    // with bit 31 set (e.g. epochs past 2^31) would never match and the wait
    // would return immediately instead of sleeping.
    const u_long uval = static_cast<uint32_t>(val);
    if (timeout == nullptr) {
      return _umtx_op(uaddr, UMTX_OP_WAIT_UINT_PRIVATE, uval, nullptr, nullptr);
    } else {
      struct _umtx_time t;
      t._timeout = *timeout;
      t._flags = 0; // relative timeout
      t._clockid = CLOCK_MONOTONIC;
      return _umtx_op(
          uaddr,
          UMTX_OP_WAIT_UINT_PRIVATE,
          uval,
          reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(t))),
          &t);
    }
  } else if (futex_op == FUTEX_WAKE_PRIVATE) {
    // Unlike Linux futex(FUTEX_WAKE), _umtx_op returns 0 on success rather than
    // the count of threads woken; `val` still caps how many are woken.
    return _umtx_op(uaddr, UMTX_OP_WAKE_PRIVATE, static_cast<u_long>(val), nullptr, nullptr);
  }
  errno = ENOTSUP;
  return -1;
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

#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(_ARM64_)
#define _ARM64_
#elif (defined(_M_ARM) || defined(__arm__)) && !defined(_ARM_)
#define _ARM_
#elif (defined(_M_AMD64) || defined(__x86_64__) || defined(_WIN64)) && !defined(_AMD64_)
#define _AMD64_
#elif !defined(_X86_)
#define _X86_
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
