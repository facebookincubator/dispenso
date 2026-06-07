/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdlib>

#include <dispenso/schedulable.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace dispenso {

#if defined(_WIN32)
namespace {
// Pin the module that physically contains dispenso's code — the .exe, dispenso's own DLL, or a host
// DLL that statically linked dispenso (FROM_ADDRESS resolves whichever it is) — so it can never be
// unmapped while a detached NewThreadInvoker thread is still executing inside it. Process-lifetime
// pin, taken only if NewThreadInvoker is ever actually used. Also covers a host FreeLibrary() at
// runtime, which the atexit drain cannot. See the rationale block in schedulable.h.
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

NewThreadInvoker::ThreadWaiter* NewThreadInvoker::getWaiter() {
  // Controlled-leak Meyers singleton, mirroring SmallBufferGlobals. The waiter is allocated on
  // first use and never freed; an atexit handler is registered alongside the allocation to block
  // process exit until every outstanding NewThreadInvoker thread has called remove(). See the
  // comment in schedulable.h for why this matters and why we do not delete the waiter at exit.
  static ThreadWaiter* waiter = []() {
    auto* w = new ThreadWaiter();
#if defined(_WIN32)
    pinModuleForNewThread();
#endif
    std::atexit([]() { getWaiter()->wait(); });
    return w;
  }();
  return waiter;
}

} // namespace dispenso
