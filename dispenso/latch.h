/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file latch.h
 * @ingroup group_sync
 * A file providing a Latch barrier type, which gives a way for threads to wait until all expected
 * threads have reached this point.  This is intended to match API and behavior of C++20 std::latch.
 **/

#pragma once

#include <dispenso/platform.h>

#include <dispenso/detail/completion_event_impl.h>

namespace dispenso {

/**
 * A class which can be used for barrier scenarios.  See e.g.
 * https://en.cppreference.com/w/cpp/thread/latch
 **/
class Latch {
 public:
  /**
   * Construct a latch with expected number of threads to wait on.
   *
   * @param threadGroupCount The number of threads in the group.
   **/
  explicit Latch(uint32_t threadGroupCount) noexcept : impl_(threadGroupCount) {}

  /**
   * Decrement the counter in a non-blocking manner.
   **/
  void count_down(uint32_t n = 1) noexcept {
    if (impl_.intrusiveStatus().fetch_sub(n, std::memory_order_acq_rel) == 1) {
      impl_.notify(0);
    }
  }

  /**
   * See if the count has been reduced to zero, indicating all necessary threads
   * have synchronized.
   *
   * @note try_wait is a misnomer, as the function never blocks.  We kept the name to match C++20
   * API.
   * @return true only if the internal counter has reached zero.
   **/
  bool try_wait() const noexcept {
    return impl_.intrusiveStatus().load(std::memory_order_acquire) == 0;
  }

  /**
   * Wait for all threads to have synchronized.
   **/
  void wait() const noexcept {
    impl_.wait(0);
  }

  /**
   * Decrement the counter and wait
   **/
  void arrive_and_wait() noexcept {
    if (impl_.intrusiveStatus().fetch_sub(1, std::memory_order_acq_rel) > 1) {
      impl_.wait(0);
    } else {
      impl_.notify(0);
    }
  }

 private:
  detail::CompletionEventImpl impl_;
};

} // namespace dispenso
