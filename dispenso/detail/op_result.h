/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>
#include <utility>

namespace dispenso {
namespace detail {

template <typename T>
class OpResult {
 public:
  OpResult() : ptr_(nullptr) {}

  template <
      typename U,
      typename = typename std::enable_if<
          !std::is_same<typename std::decay<U>::type, OpResult<T>>::value>::type>
  OpResult(U&& u) : ptr_(new (buf_) T(std::forward<U>(u))) {}

  OpResult(const OpResult<T>& oth) : ptr_(oth ? new (buf_) T(*oth.ptr_) : nullptr) {}

  // Like std::optional, moving leaves the source engaged, holding a moved-from T that is destroyed
  // by the source's destructor (or a later assignment/emplace) as usual.
  OpResult(OpResult<T>&& oth) : ptr_(oth ? new (buf_) T(std::move(*oth.ptr_)) : nullptr) {}

  OpResult& operator=(const OpResult& oth) {
    if (&oth == this) {
      return *this;
    }
    destroy();
    if (oth) {
      ptr_ = new (buf_) T(*oth.ptr_);
    }
    return *this;
  }

  OpResult& operator=(OpResult&& oth) {
    if (&oth == this) {
      return *this;
    }
    destroy();
    if (oth) {
      ptr_ = new (buf_) T(std::move(*oth.ptr_));
    }
    return *this;
  }

  ~OpResult() {
    if (ptr_) {
      ptr_->~T();
    }
  }

  template <typename... Args>
  T& emplace(Args&&... args) {
    destroy();
    ptr_ = new (buf_) T(std::forward<Args>(args)...);
    return *ptr_;
  }

  operator bool() const {
    return ptr_;
  }

  bool has_value() const {
    return ptr_;
  }

  T& value() {
    return *ptr_;
  }

 private:
  // Leaves *this disengaged before any replacement value is constructed, so that if that
  // construction throws, *this is simply empty (as with std::optional) rather than still pointing
  // at the already-destroyed old value.
  void destroy() {
    if (ptr_) {
      ptr_->~T();
      ptr_ = nullptr;
    }
  }

  alignas(T) char buf_[sizeof(T)];
  T* ptr_;
};

} // namespace detail
} // namespace dispenso
