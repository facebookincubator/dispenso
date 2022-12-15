/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file completion_event.h
 * A file providing a CompletionEvent type, which gives a way to signal to waiting threads that some
 * event has been completed.
 **/

#pragma once

#include <dispenso/platform.h>

#include <dispenso/detail/completion_event_impl.h>

namespace dispenso {

/**
 * A class which can be used for one-time notify/wait scenarios.  It is basically a way to signal to
 * any waiting threads that some event has completed.  There must be a single publisher thread
 * and zero or more waiters on arbitrary threads.  <code>reset</code> may be called to restart a
 * sequence (e.g. after <code>notify</code> occurs and all waiters have successfully exited
 * <code>wait*</code>).
 **/
class CompletionEvent {
 public:
  /**
   * Notify any waiting threads that the event has completed.  It is safe for this to be called
   * before threads call <code>wait</code>.
   **/
  void notify() {
    impl_.notify(1);
  }

  /**
   * Wait for another thread to <code>notify</code>
   **/
  void wait() const {
    impl_.wait(1);
  }

  /**
   * Peek to see if the event has been notified in any thread
   **/
  bool completed() const {
    return impl_.intrusiveStatus().load(std::memory_order_acquire);
  }

  /**
   * Wait for another thread to <code>notify</code> or for the relative timeout to expire, whichever
   * is first.
   *
   * @return true if status is "completed", false if timed out.
   **/
  template <class Rep, class Period>
  bool waitFor(const std::chrono::duration<Rep, Period>& relTime) const {
    return impl_.waitFor(1, relTime);
  }

  /**
   * Wait for another thread to <code>notify</code> or for the absolute timeout to expire, whichever
   * is first.
   *
   * @return true if status is "completed", false if timed out.
   **/
  template <class Clock, class Duration>
  bool waitUntil(const std::chrono::time_point<Clock, Duration>& absTime) const {
    return impl_.waitUntil(1, absTime);
  }

  /**
   * Resets the event to "not-completed".  This should not be called while an active
   * <code>wait*\/notify</code> sequence is still currently in play.
   **/
  void reset() {
    impl_.intrusiveStatus().store(0, std::memory_order_seq_cst);
  }

 private:
  detail::CompletionEventImpl impl_{0};
};

} // namespace dispenso
