/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file async_request.h
 * @ingroup group_async
 * A file providing AsyncRequest.  This is a bit like a lightweight channel for storing updates to
 * one object, mostly intended to be used as a single producer, single consumer update mechanism.
 **/

#pragma once

#if __cplusplus >= 201703L
#include <optional>
#else
#include <dispenso/detail/op_result.h>
#endif // C++17

#include <dispenso/platform.h>

namespace dispenso {

/**
 * A type for making async requests.  Although it is safe to use from multiple producers and
 * consumers, it is primarily intended to be used from single producer, single consumer.
 *
 * Typically the consumer will request an update of the value from thread 0, and the producer will
 * look whether an update was requested from thread 1.  Once the producer determines an update was
 * requested (updateRequested() returns true), it calls tryEmplaceUpdate() to update the underlying
 * data.  Then when the consumer on thread 0 next calls getUpdate(), an optional wrapper to the
 * updated data is returned, and the AsyncRequest object is reset (it no longer has valid data, and
 * no update will have yet been requested for the next update).
 **/
template <typename T>
class AsyncRequest {
 public:
  // A lightweight std::optional-like type with a subset of functionality.
#if __cplusplus >= 201703L
  using OpResult = std::optional<T>;
#else
  using OpResult = detail::OpResult<T>;
#endif // C++17

  /**
   * The consumer can call this to request an update to the underlying data.  If request has already
   * been made or fulfilled, this is a no-op.
   **/
  void requestUpdate() {
    RequestState state = kNone;
    state_.compare_exchange_strong(state, kNeedsUpdate, std::memory_order_acq_rel);
  }

  /**
   * The producer can check this to determine if an update is needed.
   *
   * @return true if an update is required, false otherwise.
   **/
  bool updateRequested() const {
    return state_.load(std::memory_order_acquire) == kNeedsUpdate;
  }

  /**
   * The producer can try to emplace a new T object in response to a request.
   * @param args The arguments to emplace.
   * @return true if the underlying data was updated.  false if the underlying data is not in need
   * of an update.
   * @note For cases where calling this superflously could be expensive, it is wise to check
   * updateRequested() first.
   **/
  template <typename... Args>
  bool tryEmplaceUpdate(Args&&... args) {
    RequestState state = kNeedsUpdate;
    if (!state_.compare_exchange_strong(state, kUpdating, std::memory_order_acq_rel)) {
      return false;
    }
    obj_.emplace(std::forward<Args>(args)...);
    state_.store(kReady, std::memory_order_release);
    return true;
  }

  /**
   * The consumer can attempt to get an update.
   * @return An optional wrapper to the underlying data.  If no update is ready, nullopt is
   * returned. Once an update has been returned, the AsyncRequest object is returned to a state with
   * no underlying data.
   **/
  OpResult getUpdate() {
    if (state_.load(std::memory_order_acquire) == kReady) {
      auto obj = std::move(obj_);
      state_.store(kNone, std::memory_order_release);
      return obj;
    }
    return {};
  }

 private:
  enum RequestState { kNone, kNeedsUpdate, kUpdating, kReady };
  alignas(kCacheLineSize) std::atomic<RequestState> state_ = {kNone};
  OpResult obj_;
};

} // namespace dispenso
