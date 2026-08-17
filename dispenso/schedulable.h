/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file schedulable.h
 * @ingroup group_core
 * Classes providing simple schedulables that match scheduling interfaces of *TaskSet and ThreadPool
 *
 **/

#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <dispenso/detail/completion_event_impl.h>
#include <dispenso/task_set.h>

namespace dispenso {

/**
 * A class fullfilling the Schedulable concept that immediately invokes the functor.  This can be
 * used in place of <code>ThreadPool</code> or <code>TaskSet</code> with <code>Future</code>s at
 * construction or through <code>then</code>, or it may be used in TimedTask scheduling for
 * short-running tasks.
 **/
class ImmediateInvoker {
 public:
  /**
   * Schedule a functor to be executed.  It will be invoked immediately.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f) const {
    f();
  }

  /**
   * Schedule a functor to be executed.  It is a bit oxymoronical to call this function, since
   * ForceQueuingTag will have no effect, and it's use is discouraged.
   *
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag) const {
    f();
  }
};

constexpr ImmediateInvoker kImmediateInvoker;

/**
 * A class fullfilling the Schedulable concept that always invokes on a new thread.  This can be
 * used in place of <code>ThreadPool</code> or <code>TaskSet</code> with <code>Future</code>s at
 * construction or through <code>then</code>.
 **/
namespace detail {
// Drains every outstanding NewThreadInvoker thread. Idempotent; defined in
// schedulable.cpp.
DISPENSO_DLL_ACCESS void drainNewThreadInvokerThreads();

// Runs the drain from the destructor of a static living in *the caller's*
// module, which is the whole point of it being in a header.
//
// dispenso also registers the drain with atexit(), but in a shared build that
// registration belongs to the dispenso DLL and therefore runs at
// DLL_PROCESS_DETACH. ExitProcess reaches DLL_PROCESS_DETACH only after it has
// already terminated every other thread in the process, so a drain registered
// there can never see the threads it is meant to join -- they have been killed
// mid-execution, which is exactly the access violation this avoids. Static
// destructors in the executable run earlier, during ordinary exit processing
// and before ExitProcess, while the threads are still alive and joinable.
struct NewThreadDrainRegistrar {
  ~NewThreadDrainRegistrar() {
    drainNewThreadInvokerThreads();
  }
};

inline void ensureNewThreadDrainRegistered() {
  static NewThreadDrainRegistrar registrar;
  (void)registrar;
}
} // namespace detail

class NewThreadInvoker {
 public:
  /**
   * Schedule a functor to be executed on a new thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   *
   * <code>schedule</code> hands back no handle, so threads still running when the process exits
   * are joined automatically. Prefer synchronizing on completion yourself where you can: the
   * automatic drain runs with the static destructors of whichever module called
   * <code>schedule</code>, so it only runs early enough to help when that module is the
   * executable. A process that reaches <code>NewThreadInvoker</code> solely from within shared
   * libraries has no drain that beats process teardown.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f) const {
    schedule(std::forward<F>(f), ForceQueuingTag());
  }
  /**
   * Schedule a functor to be executed on a new thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag) const {
    // The thread is retained (not detached) and joined at process exit; see
    // ThreadTracker for why detaching is unsafe on Windows. `done` is set by the
    // thread as its very last act so schedule() can reap already-finished threads
    // and keep retention bounded across a long-running process.
    // Must precede thread creation: the registrar's destructor is what drains
    // this thread at exit in a shared build.
    detail::ensureNewThreadDrainRegistered();
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread thread([f = std::move(f), done]() {
      f();
      done->store(true, std::memory_order_release);
    });
    getTracker()->add(std::move(thread), std::move(done));
  }

 private:
  /// @cond INTERNAL
  // NewThreadInvoker spawns one std::thread per schedule(). On Windows shared-lib
  // builds a *detached* thread that is still executing during process exit faults
  // (EXCEPTION_ACCESS_VIOLATION): it runs a synchronization primitive
  // (e.g. WakeByAddressAll, from CompletionEvent::notify) after ntdll has begun
  // tearing down its wait machinery at shutdown. See T282829604.
  //
  // The fix is to wait for each thread to FULLY terminate before shutdown proceeds
  // -- not merely for its functor to return. Only the OS thread handle signals true
  // termination (after the thread's last sync call and OS thread-exit), so we retain
  // the threads joinable and join them from an atexit handler, which runs before
  // module teardown. On Windows the wait is BOUNDED (see joinAll): a thread that
  // cannot terminate -- e.g. one parked on the loader lock during a static-CRT
  // DLL_PROCESS_DETACH -- must not wedge shutdown, so it is detached and left for
  // process termination to reclaim (pinModuleForNewThread keeps our code mapped so
  // that stays benign).
  //
  // The tracker is a controlled-leak singleton (schedulable.cpp); it is never
  // destroyed, so a schedule() from a late static destructor still finds it valid.
  struct ThreadTracker {
    struct Entry {
      std::thread thread;
      // Set true by the thread as its last act. Lets add() reap finished threads
      // without blocking; shared so the store outlives an entries_ reallocation.
      std::shared_ptr<std::atomic<bool>> done;
    };

    std::mutex mtx_;
    std::vector<Entry> entries_;

    void add(std::thread&& t, std::shared_ptr<std::atomic<bool>> done)
        DISPENSO_NO_THREAD_SAFETY_ANALYSIS {
      // Opportunistically reap already-finished threads so entries_ does not grow
      // unbounded over the lifetime of a long-running process. The joins happen
      // after mtx_ is released: `done` only means the functor returned, and the
      // OS-level teardown that follows is precisely the window this class does not
      // trust, so it must not block every other schedule() behind the lock.
      std::vector<std::thread> finished;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        for (size_t i = 0; i < entries_.size();) {
          if (entries_[i].done->load(std::memory_order_acquire)) {
            finished.push_back(std::move(entries_[i].thread));
            entries_[i] = std::move(entries_.back());
            entries_.pop_back();
          } else {
            ++i;
          }
        }
        entries_.push_back(Entry{std::move(t), std::move(done)});
      }
      for (std::thread& thread : finished) {
        thread.join(); // already finished -> returns promptly
      }
    }

    // Defined in schedulable.cpp; joins each thread, bounded on Windows.
    // Returns how many were still mid-functor, which is the count that says
    // nothing had synchronized on them. Threads that had finished but were not
    // yet reaped do not count.
    size_t joinAll() DISPENSO_NO_THREAD_SAFETY_ANALYSIS;
  };

  DISPENSO_DLL_ACCESS static ThreadTracker* getTracker();

  friend void detail::drainNewThreadInvokerThreads();
  /// @endcond
};

constexpr NewThreadInvoker kNewThreadInvoker;

} // namespace dispenso
