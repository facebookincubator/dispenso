/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file once_function.h
 * @ingroup group_core
 * A file providing OnceFunction, a class providing void() signature for closure to be called only
 * once.  It is built to be cheap to create and move.
 **/

#pragma once

#include <cstring>
#include <utility>

#include <dispenso/detail/once_callable_impl.h>
#include <dispenso/platform.h>

namespace dispenso {

#if DISPENSO_HAS_CONCEPTS
/**
 * @concept OnceCallableFunc
 * @brief A callable suitable for wrapping in OnceFunction or scheduling as a task.
 *
 * The callable must be invocable with no arguments. The return value (if any) is discarded.
 * This is the fundamental requirement for functors passed to OnceFunction,
 * ThreadPool::schedule, TaskSet::schedule, and similar scheduling interfaces.
 **/
template <typename F>
concept OnceCallableFunc = std::invocable<F>;
#endif // DISPENSO_HAS_CONCEPTS

namespace detail {
template <typename Result>
class FutureBase;
template <typename Result>
class FutureImplBase;
} // namespace detail

/**
 * A class fullfilling the void() signature, and operator() must be called exactly once for valid
 * <code>OnceFunction</code>s.  This class can be much more efficient than std::function for type
 * erasing functors without too much state (currently 56 bytes inline buffer).
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
      : invoke_(nullptr)
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
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  OnceFunction(F&& f) {
    auto callable = detail::createOnceCallable(std::forward<F>(f), buf_);
    invoke_ = callable.invoke;
  }

  OnceFunction(const OnceFunction& other) = delete;

  /** Move constructor.  Copies the full 64-byte object (one cache line). */
  OnceFunction(OnceFunction&& other) noexcept {
    std::memcpy(static_cast<void*>(this), &other, sizeof(OnceFunction));
#if defined DISPENSO_DEBUG
    other.invoke_ = nullptr;
#endif // DISPENSO_DEBUG
  }

  OnceFunction& operator=(OnceFunction&& other) noexcept {
    if (this != &other) {
#if defined DISPENSO_DEBUG
      assert(
          invoke_ == nullptr &&
          "OnceFunction must be invoked or cleanupNotRun() before reassignment");
#endif // DISPENSO_DEBUG
      std::memcpy(static_cast<void*>(this), &other, sizeof(OnceFunction));
#if defined DISPENSO_DEBUG
      other.invoke_ = nullptr;
#endif // DISPENSO_DEBUG
    }
    return *this;
  }

  /**
   * Destroy the type-erased functor and release its resources without invoking it.
   * Use this when a OnceFunction will not be called but its resources must still be freed.
   * Like operator(), this must be called at most once, and the OnceFunction must not be used after.
   **/
  void cleanupNotRun() {
#if defined DISPENSO_DEBUG
    assert(invoke_ != nullptr && "Must not cleanup an invalid OnceFunction!");
    invoke_(buf_, false);
    invoke_ = nullptr;
#else
    invoke_(buf_, false);
#endif // DISPENSO_DEBUG
  }

  /**
   * Invoke the type-erased functor.  This function must be called exactly once.  Fewer will result
   * in a leak, while more will invoke on an invalid object.
   **/
  void operator()() const {
#if defined DISPENSO_DEBUG
    assert(invoke_ != nullptr && "Must not use OnceFunction more than once!");
#endif // DISPENSO_DEBUG

    invoke_(buf_, true);

#if defined DISPENSO_DEBUG
    invoke_ = nullptr;
#endif // DISPENSO_DEBUG
  }

 private:
  // 64 bytes total (one cache line): 56 bytes inline storage + 8 bytes invoke ptr.
  // Inline: buf_ holds the functor directly.
  // Spill: buf_[0..7] holds a void* to pool-allocated storage.
  // Moves are a 64-byte memcpy — no pointer fixup needed.
  alignas(64) mutable char buf_[detail::kOnceFunctionInlineSize];
  mutable void (*invoke_)(void*, bool);

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;
};

} // namespace dispenso
