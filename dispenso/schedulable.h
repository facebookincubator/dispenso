/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file schedulable.h
 * Classes providing simple schedulables that match scheduling interfaces of *TaskSet and ThreadPool
 *
 **/

#pragma once

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
  void schedule(F&& f) const {
    f();
  }

  /**
   * Schedule a functor to be executed.  It is a bit oxymoronical to call this function, since
   * ForceQueuingTag will have no effect, and it's use is discouraged.
   *
   **/
  template <typename F>
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
  void schedule(F&& f, ForceQueuingTag) const {
    static ThreadWaiter waiter;
    waiter.add();
    std::thread thread([f = std::move(f)]() {
      f();
      waiter.remove();
    });
    thread.detach();
  }

  // This is to protect against accessing stale memory after main exits.  This was encountered on
  // occasion in conjunction with Futures in tests where the work was stolen locally long before the
  // thread could be launched, and the process already is exiting when the thread is executing.
  // Because it was after shutdown, backing memory for things could be no longer available.
  struct ThreadWaiter {
    detail::CompletionEventImpl impl_{0};

    void add() {
      impl_.intrusiveStatus().fetch_add(1, std::memory_order_acq_rel);
    }

    void remove() {
      if (impl_.intrusiveStatus().fetch_sub(1, std::memory_order_acq_rel) == 1) {
        impl_.notify(0);
      }
    }

    ~ThreadWaiter() {
      impl_.wait(0);
    }
  };
};

constexpr NewThreadInvoker kNewThreadInvoker;

} // namespace dispenso
