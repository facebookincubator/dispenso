/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
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

void NewThreadInvoker::ThreadTracker::joinAll() {
  // One deadline shared across all threads AND all drain passes below, so a burst
  // outstanding at exit is bounded overall rather than per-thread or per-pass.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  // Drain until entries_ is observed empty under the lock. schedule() can still
  // run while we reap -- from a thread that is still live, or from a static
  // destructor sequenced after this atexit handler (see getTracker) -- and such a
  // thread lands in a fresh entries_ that a single swap would leave unjoined.
  // Past the deadline reapThreadAtExit detaches immediately, so late arrivals
  // cannot extend shutdown beyond the bound.
  for (;;) {
    std::vector<Entry> local;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (entries_.empty()) {
        break;
      }
      local.swap(entries_);
    }
    for (auto& e : local) {
      reapThreadAtExit(e.thread, deadline);
    }
  }
}

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
    std::atexit([]() { getTracker()->joinAll(); });
    return t;
  }();
  return tracker;
}

} // namespace dispenso
