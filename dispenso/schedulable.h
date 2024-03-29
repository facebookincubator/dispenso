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
    auto* waiter = getWaiter();
    waiter->add();
    std::thread thread([f = std::move(f), waiter]() {
      f();
      waiter->remove();
    });
    thread.detach();
  }

 private:
  // This is to protect against accessing stale memory after main exits.  This was encountered on
  // occasion in conjunction with Futures in tests where the work was stolen locally long before the
  // thread could be launched, and the process already is exiting when the thread is executing.
  // Because it was after shutdown, backing memory for things could be no longer available.
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
      if (--count_ == 0) {
        cond_.notify_one();
      }
    }

    ~ThreadWaiter() {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this]() { return count_ == 0; });
    }
  };
  DISPENSO_DLL_ACCESS static ThreadWaiter* getWaiter();

  static void destroyThreadWaiter();
};

constexpr NewThreadInvoker kNewThreadInvoker;

} // namespace dispenso
