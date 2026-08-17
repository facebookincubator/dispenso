/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <dispenso/schedulable.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace dispenso {

#if defined(_WIN32)
namespace {
// Pin the module that physically contains dispenso's code — the .exe, dispenso's own DLL, or a host
// DLL that statically linked dispenso (FROM_ADDRESS resolves whichever it is) — so it can never be
// unmapped while a NewThreadInvoker thread is still executing inside it. joinAll() waits for
// threads to terminate, but that wait is bounded; this pin makes a straggler that outlives the
// deadline (and is therefore detached) non-fatal. Process-lifetime pin, taken only if
// NewThreadInvoker is ever actually used. Also covers a host FreeLibrary() at runtime, which the
// atexit drain cannot.
void pinModuleForNewThread() {
  HMODULE hmod = nullptr;
  const BOOL pinned = GetModuleHandleExW(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
      reinterpret_cast<LPCWSTR>(&pinModuleForNewThread),
      &hmod);
  // Pinning our own code address should never fail; if it somehow does we lose the unmap guarantee
  // that lets the atexit drain be bounded, so surface it loudly in debug builds. Release proceeds
  // (the bounded wait remains a best-effort safety net).
  assert(pinned && "Failed to pin dispenso module for NewThreadInvoker");
  (void)pinned;
}
} // namespace
#endif // _WIN32

namespace {
// Wait up to `deadline` for a NewThreadInvoker thread to fully terminate, then
// reclaim it. We wait on the OS thread itself, not a dispenso-level signal,
// because only the kernel marks the thread done AFTER its final synchronization
// call and OS thread-exit (see the rationale on ThreadTracker in schedulable.h).
// The wait is bounded on Windows so a thread that cannot make progress — e.g. one
// parked on the loader lock during static-CRT DLL_PROCESS_DETACH — cannot wedge
// shutdown.
void reapThreadAtExit(std::thread& t, std::chrono::steady_clock::time_point deadline) {
#if defined(_WIN32)
  const auto now = std::chrono::steady_clock::now();
  const DWORD ms = now < deadline
      ? static_cast<DWORD>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count())
      : 0;
  if (WaitForSingleObject(reinterpret_cast<HANDLE>(t.native_handle()), ms) == WAIT_OBJECT_0) {
    t.join();
  } else {
    // Straggler that missed the deadline. Detach rather than join so shutdown is
    // never wedged; pinModuleForNewThread() keeps our code mapped so the lingering
    // thread's eventual termination by the OS stays benign.
    t.detach();
  }
#else
  // POSIX has neither the code-unmap-under-running-thread nor the loader-lock
  // hazard, so an unbounded join is both correct and simplest.
  (void)deadline;
  t.join();
#endif
}
} // namespace

size_t NewThreadInvoker::ThreadTracker::joinAll() {
  // One deadline shared across all threads AND all drain passes below, so a burst
  // outstanding at exit is bounded overall rather than per-thread or per-pass.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  // Drain until entries_ is observed empty under the lock. schedule() can still
  // run while we reap -- from a thread that is still live, or from a static
  // destructor sequenced after this atexit handler (see getTracker) -- and such a
  // thread lands in a fresh entries_ that a single swap would leave unjoined.
  // Past the deadline reapThreadAtExit detaches immediately, so late arrivals
  // cannot extend shutdown beyond the bound.
  // Count only threads whose functor had not finished. A thread that has run to
  // completion is still tracked until some later schedule() reaps it, so the
  // most recently scheduled thread is almost always still here even when the
  // caller waited on every result -- counting those would flag well-behaved
  // code.
  size_t stillRunning = 0;
  for (;;) {
    std::vector<Entry> local;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (entries_.empty()) {
        break;
      }
      local.swap(entries_);
    }
    // Sample the whole batch before reaping any of it: joining the first
    // entries takes long enough for later ones to finish, which would otherwise
    // undercount how many were running when shutdown began.
    for (const auto& e : local) {
      if (!e.done->load(std::memory_order_acquire)) {
        ++stillRunning;
      }
    }
    for (auto& e : local) {
      reapThreadAtExit(e.thread, deadline);
    }
  }
  return stillRunning;
}

namespace detail {
void drainNewThreadInvokerThreads() {
  // joinAll() drains until it observes an empty tracker, so calling it from
  // both the registrar and the atexit handler is harmless; whichever runs
  // first does the work and the other finds nothing left.
  const auto start = std::chrono::steady_clock::now();
  const size_t outstanding = NewThreadInvoker::getTracker()->joinAll();
  const long long blockedMs =
      static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count());

  // Requiring that the drain actually *blocked* is what keeps this quiet for
  // correct code. Waiting on a Future returns from inside the functor, so the
  // worker is legitimately still winding down when main reaches exit; counting
  // those flags well-behaved callers a third of the time. Work that nothing
  // waited on is what holds exit open measurably.
  constexpr long long kReportThresholdMs = 5;
  if (outstanding == 0 || blockedMs < kReportThresholdMs) {
    return;
  }
  // Reaching here means these threads were still running with nothing waiting
  // on them -- the process only survived because this drain caught them. Report
  // it in every build: the message is rare by construction, and staying silent
  // hides a pattern that is merely unlucky here and fatal in a process that
  // uses dispenso solely from shared libraries, where no drain runs early
  // enough to help.
  std::fprintf(
      stderr,
      "dispenso: %zu NewThreadInvoker thread(s) were still running at process exit; "
      "draining them held exit open for %lldms.\n"
      "  Nothing had synchronized on their completion. Wait on the work before exiting;\n"
      "  this rescue is unavailable when dispenso is used only from shared libraries.\n",
      outstanding,
      blockedMs);
  std::fflush(stderr);
}
} // namespace detail

NewThreadInvoker::ThreadTracker* NewThreadInvoker::getTracker() {
  // Controlled-leak Meyers singleton. The atexit handler joins every outstanding
  // NewThreadInvoker thread (waiting for true termination) before module teardown.
  // The tracker is intentionally never freed (see schedulable.h) so that a
  // schedule() from a late static destructor still finds a valid tracker.
  static ThreadTracker* tracker = []() {
    auto* t = new ThreadTracker();
#if defined(_WIN32)
    pinModuleForNewThread();
#endif
    std::atexit([]() { detail::drainNewThreadInvokerThreads(); });
    return t;
  }();
  return tracker;
}

} // namespace dispenso
