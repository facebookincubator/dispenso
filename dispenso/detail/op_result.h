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

  OpResult(OpResult<T>&& oth) : ptr_(oth ? new (buf_) T(std::move(*oth.ptr_)) : nullptr) {
    oth.ptr_ = nullptr;
  }

  OpResult& operator=(const OpResult& oth) {
    if (&oth == this) {
      return *this;
    }
    if (ptr_) {
      ptr_->~T();
    }

    if (oth) {
      ptr_ = new (buf_) T(*oth.ptr_);
    } else {
      ptr_ = nullptr;
    }
    return *this;
  }

  OpResult& operator=(OpResult&& oth) {
    if (&oth == this) {
      return *this;
    }
    if (ptr_) {
      ptr_->~T();
    }

    if (oth) {
      ptr_ = new (buf_) T(std::move(*oth.ptr_));
      oth.ptr_ = nullptr;
    } else {
      ptr_ = nullptr;
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
    if (ptr_) {
      ptr_->~T();
    }
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
  alignas(T) char buf_[sizeof(T)];
  T* ptr_;
};

} // namespace detail
} // namespace dispenso
