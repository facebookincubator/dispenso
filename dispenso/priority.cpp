/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/priority.h>

#if (defined(__unix__) || defined(unix)) && !defined(USG)
#include <sys/param.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__MACH__)
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <Windows.h>
#elif defined(BSD)
#include <sys/rtprio.h>
#include <sys/types.h>
#endif

namespace dispenso {

namespace {
DISPENSO_THREAD_LOCAL ThreadPriority g_threadPriority = ThreadPriority::kNormal;
} // namespace

ThreadPriority getCurrentThreadPriority() {
  return g_threadPriority;
}

#ifdef __MACH__
bool setCurrentThreadPriority(ThreadPriority prio) {
  mach_port_t threadport = pthread_mach_thread_np(pthread_self());
  if (prio == ThreadPriority::kRealtime) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double msToAbsTime = ((double)info.denom / (double)info.numer) * 1000000.0;
    thread_time_constraint_policy_data_t time_constraints;
    time_constraints.period = 0;
    time_constraints.computation = static_cast<uint32_t>(1.0 * msToAbsTime);
    time_constraints.constraint = static_cast<uint32_t>(10.0 * msToAbsTime);
    time_constraints.preemptible = 0;

    if (thread_policy_set(
            threadport,
            THREAD_TIME_CONSTRAINT_POLICY,
            (thread_policy_t)&time_constraints,
            THREAD_TIME_CONSTRAINT_POLICY_COUNT) != KERN_SUCCESS) {
      return false;
    }
  }

  // https://fergofrog.com/code/cbowser/xnu/osfmk/kern/sched.h.html#_M/MAXPRI_USER
  struct thread_precedence_policy ttcpolicy;

  switch (prio) {
    case ThreadPriority::kLow:
      ttcpolicy.importance = 20;
      break;
    case ThreadPriority::kNormal:
      ttcpolicy.importance = 37;
      break;
    case ThreadPriority::kHigh: // fallthrough
    case ThreadPriority::kRealtime:
      ttcpolicy.importance = 63;
      break;
  }

  if (thread_policy_set(
          threadport,
          THREAD_PRECEDENCE_POLICY,
          (thread_policy_t)&ttcpolicy,
          THREAD_PRECEDENCE_POLICY_COUNT) != KERN_SUCCESS) {
    return false;
  }

  g_threadPriority = prio;
  return true;
}
#elif defined(_WIN32)
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadpriority
bool setCurrentThreadPriority(ThreadPriority prio) {
  if (prio == ThreadPriority::kRealtime) {
    if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
      return false;
    }
  }

  if (prio == ThreadPriority::kHigh) {
    // Best effort
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
  }

  bool success = false;
  switch (prio) {
    case ThreadPriority::kLow:
      success = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
      break;
    case ThreadPriority::kNormal:
      success = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
      break;
    case ThreadPriority::kHigh: // fallthrough
    case ThreadPriority::kRealtime:
      success = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
      break;
  }

  if (!success) {
    return false;
  }

  g_threadPriority = prio;
  return true;
}
#elif defined(__linux__)
bool setCurrentThreadPriority(ThreadPriority prio) {
  if (prio == ThreadPriority::kRealtime) {
    struct sched_param param;
    param.sched_priority = 99;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param)) {
      return false;
    }
  }

  switch (prio) {
    case ThreadPriority::kLow:
      errno = 0;
      (void)!nice(10);
      break;
    case ThreadPriority::kNormal:
      errno = 0;
      (void)!nice(0);
      break;
    case ThreadPriority::kHigh: // fallthrough
    case ThreadPriority::kRealtime: {
      struct rlimit rlim;
      getrlimit(RLIMIT_NICE, &rlim);
      if (rlim.rlim_max <= 20) {
        return false;
      }
      rlim.rlim_cur = rlim.rlim_max;
      setrlimit(RLIMIT_NICE, &rlim);
      errno = 0;
      (void)!nice(static_cast<int>(20 - rlim.rlim_max));
    }
  }
  if (errno != 0) {
    return false;
  }
  g_threadPriority = prio;
  return true;
}
#elif defined(__FreeBSD__)
// TODO: Find someone who has a FreeBSD system to test this code.
bool setCurrentThreadPriority(ThreadPriority prio) {
  struct rtprio rtp;

  if (prio == ThreadPriority::kRealtime) {
    rtp.type = RTP_PRIO_REALTIME;
    rtp.prio = 10;
    if (rtprio_thread(RTP_SET, 0, &rtp)) {
      return false;
    }
  } else {
    rtp.type = RTP_PRIO_NORMAL;
    switch (prio) {
      case ThreadPriority::kLow:
        rtp.prio = 31;
        break;
      case ThreadPriority::kNormal:
        rtp.prio = 15;
        break;
      case ThreadPriority::kHigh: // fallthrough
      case ThreadPriority::kRealtime:
        rtp.prio = 0;
        break;
    }
    if (rtprio_thread(RTP_SET, 0, &rtp)) {
      return false;
    }
  }
  g_threadPriority = prio;
  return true;
}
#else
bool setCurrentThreadPriority(ThreadPriority prio) {
  return false;
}

#endif // platform

} // namespace dispenso
