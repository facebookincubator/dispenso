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
  OnceFunction() : ops_(&detail::EmptyOnceCallableOps<>::kOps) {}

  /**
   * Construct a <code>OnceFunction</code> with a valid functor.
   *
   * @param f A functor with signature void().  Ideally this should be a concrete functor (e.g. from
   * lambda), though it will work with e.g. std::function.  The downside in the latter case is extra
   * overhead for double type erasure.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  OnceFunction(F&& f) : ops_(detail::createOnceCallable(std::forward<F>(f), buf_)) {}

  OnceFunction(const OnceFunction& other) = delete;

  /**
   * Move constructor.  Copies the full 64-byte object (one cache line), except for inline functors
   * that are not trivially copyable, which are move-constructed into place.
   **/
  OnceFunction(OnceFunction&& other) noexcept {
    moveFrom(other);
  }

  OnceFunction& operator=(OnceFunction&& other) noexcept {
    if (this != &other) {
#if defined DISPENSO_DEBUG
      assert(
          ops_->invoke == nullptr &&
          "OnceFunction must be invoked or cleanupNotRun() before reassignment");
#endif // DISPENSO_DEBUG
      moveFrom(other);
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
    assert(ops_->invoke != nullptr && "Must not cleanup an invalid OnceFunction!");
#endif
    consumeOps()->invoke(buf_, false);
  }

  /**
   * Invoke the type-erased functor.  This function must be called exactly once.  Fewer will result
   * in a leak, while more will invoke on an invalid object.
   **/
  void operator()() const {
#if defined DISPENSO_DEBUG
    assert(ops_->invoke != nullptr && "Must not use OnceFunction more than once!");
#endif // DISPENSO_DEBUG
    consumeOps()->invoke(buf_, true);
  }

 private:
  // 64 bytes total (one cache line): 56 bytes inline storage + 8 bytes operations ptr.
  // Inline: buf_ holds the functor directly.
  // Spill: buf_[0..7] holds a void* to pool-allocated storage.
  // Moves are a 64-byte memcpy, except for inline functors that are not trivially copyable, which
  // the operations' relocate entry move-constructs into place.
  void moveFrom(OnceFunction& other) noexcept {
    if (DISPENSO_EXPECT(other.ops_->relocate == nullptr, true)) {
      std::memcpy(static_cast<void*>(this), &other, sizeof(OnceFunction));
    } else {
      ops_ = other.ops_;
      ops_->relocate(other.buf_, buf_);
      other.ops_ = &detail::EmptyOnceCallableOps<>::kOps;
    }
#if defined DISPENSO_DEBUG
    other.ops_ = &detail::EmptyOnceCallableOps<>::kOps;
#endif // DISPENSO_DEBUG
  }

  // Marks this object consumed before the call, because the invoked functor may free the memory
  // holding this OnceFunction.  Leaves no stale relocate entry for a later move to act on.
  const detail::OnceCallableOps* consumeOps() const {
    const detail::OnceCallableOps* ops = ops_;
    ops_ = &detail::EmptyOnceCallableOps<>::kOps;
    return ops;
  }

  alignas(64) mutable char buf_[detail::kOnceFunctionInlineSize];
  mutable const detail::OnceCallableOps* ops_;

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;
};

} // namespace dispenso
