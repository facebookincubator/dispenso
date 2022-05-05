/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file once_function.h
 * A file providing OnceFunction, a class providing void() signature for closure to be called only
 * once.  It is built to be cheap to create and move.
 **/

#pragma once

#include <utility>

#include <dispenso/detail/once_callable_impl.h>

namespace dispenso {
namespace detail {
template <typename Result>
class FutureBase;
template <typename Result>
class FutureImplBase;
} // namespace detail

/**
 * A class fullfilling the void() signature, and operator() must be called exactly once for valid
 * <code>OnceFunction</code>s.  This class can be much more efficient than std::function for type
 * erasing functors without too much state (currently < ~250 bytes).
 * @note The wrapped type-erased functor in OnceFunction is *not* deleted upon destruction, but
 * rather when operator() is called.  It is the user's responsibility to ensure that operator() is
 * called.
 *
 **/
class OnceFunction {
 public:
  /**
   * Construct a <code>OnceFunction</code> with invalid state.
   **/
  OnceFunction()
#if defined DISPENSO_DEBUG
      : onceCallable_(nullptr)
#endif // DISPENSO_DEBUG
  {
  }

  /**
   * Construct a <code>OnceFunction</code> with a valid functor.
   *
   * @param f A functor with signature void().  Ideally this should be a concrete functor (e.g. from
   * lambda), though it will work with e.g. std::function.  The downside in the latter case is extra
   * overhead for double type erasure.
   **/
  template <typename F>
  OnceFunction(F&& f) : onceCallable_(detail::createOnceCallable(std::forward<F>(f))) {}

  OnceFunction(const OnceFunction& other) = delete;

  OnceFunction(OnceFunction&& other) : onceCallable_(other.onceCallable_) {
#if defined DISPENSO_DEBUG
    other.onceCallable_ = nullptr;
#endif // DISPENSO_DEBUG
  }

  OnceFunction& operator=(OnceFunction&& other) {
    onceCallable_ = other.onceCallable_;
#if defined DISPENSO_DEBUG
    if (&other != this) {
      other.onceCallable_ = nullptr;
    }
#endif // DISPENSO_DEBUG
    return *this;
  }

  /**
   * Invoke the type-erased functor.  This function must be called exactly once.  Fewer will result
   * in a leak, while more will invoke on an invalid object.
   **/
  void operator()() const {
#if defined DISPENSO_DEBUG
    assert(onceCallable_ != nullptr && "Must not use OnceFunction more than once!");
#endif // DISPENSO_DEBUG

    onceCallable_->run();

#if defined DISPENSO_DEBUG
    onceCallable_ = nullptr;
#endif // DISPENSO_DEBUG
  }

 private:
  OnceFunction(detail::OnceCallable* func, bool) : onceCallable_(func) {}

  mutable detail::OnceCallable* onceCallable_;

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;
};

} // namespace dispenso
