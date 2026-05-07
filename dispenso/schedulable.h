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

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>

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
class NewThreadInvoker {
 public:
  /**
   * Schedule a functor to be executed on a new thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
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
    auto* waiter = getWaiter();
    waiter->add();
    std::thread thread([f = std::move(f), waiter]() {
      // RAII so remove() runs even if f() throws. In practice an uncaught exception out of a
      // std::thread function calls std::terminate -> abort, which kills the process before the
      // atexit waiter could block, but the guard documents the invariant and covers exotic
      // terminate-handler setups.
      RemoveGuard guard{waiter};
      f();
    });
    thread.detach();
  }

 private:
  // DO NOT REMOVE WITHOUT READING THE FOLLOWING:
  //
  // schedule() above spawns a detached std::thread per call. On Windows shared-lib builds, if the
  // process exits while a detached thread is mid-execution inside dispenso's code, the OS unmaps
  // dispenso.dll's pages during DLL_PROCESS_DETACH while the thread is still executing them →
  // EXCEPTION_ACCESS_VIOLATION (0xC0000005). On any platform, the detached thread also has a
  // non-trivial window between Future::status notify(kReady) and the final
  // decRefCountMaybeDestroy + thread-local-storage teardown, which extends past the user-visible
  // future.get() return.
  //
  // ThreadWaiter blocks the atexit handler until every detached thread has called remove(), so
  // no thread is mid-execution when _cexit / DLL unload runs. The SmallBufferAllocator
  // controlled-leak fix (small_buffer_allocator.cpp) addresses a different hazard (returning
  // small buffers to a destroyed central store) and is NOT a substitute for this. Both
  // mitigations are needed.
  //
  // The relevant tests are future_test_sans_exceptions and future_shared_test, the
  // Future.AsyncNotAsyncSpecifyNewThread / NewThreadInvoker / AsyncSpecifyNewThread cases.
  //
  // The ThreadWaiter object itself is intentionally leaked (controlled-leak singleton in
  // schedulable.cpp). Deleting it on shutdown would create a UAF window for any post-atexit
  // schedule() call (external thread, static destructor) that hits a freed waiter.
  struct ThreadWaiter {
    int count_ = 0;
    std::mutex mtx_;
    std::condition_variable cond_;

    void add() {
      std::lock_guard<std::mutex> lk(mtx_);
      ++count_;
    }

    void remove() {
      std::lock_guard<std::mutex> lk(mtx_);
      assert(count_ > 0 && "remove() called without matching add()");
      if (--count_ == 0) {
        cond_.notify_one();
      }
    }

    void wait() {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this]() { return count_ == 0; });
    }
  };

  struct RemoveGuard {
    ThreadWaiter* w;
    ~RemoveGuard() {
      w->remove();
    }
  };

  DISPENSO_DLL_ACCESS static ThreadWaiter* getWaiter();
};

constexpr NewThreadInvoker kNewThreadInvoker;

} // namespace dispenso
