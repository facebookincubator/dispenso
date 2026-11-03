/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include <dispenso/detail/math.h>
#include <dispenso/platform.h>
#include <dispenso/small_buffer_allocator.h>

namespace dispenso {
namespace detail {

// Inline storage size: 64-byte OnceFunction minus 8-byte operations pointer.
// Covers most parallel_for lambdas (up to 7 pointers of captures).
//
// A callable is stored inline when it fits and moving it cannot throw. Moving a
// OnceFunction byte-copies the whole object unless the callable's operations
// provide `relocate`, which is set only for inline callables that are not
// trivially copyable: those are move-constructed into the destination and the
// source is destroyed. Spilled callables are reached through a pointer, so a
// byte copy moves them.
constexpr size_t kOnceFunctionInlineSize = 56;

struct OnceCallableOps {
  void (*invoke)(void* buf, bool run);
  void (*relocate)(void* src, void* dst);
};

// Invoke for inline-stored functors: functor lives directly in the buffer.
template <typename F>
void invokeInline(void* buf, bool run) {
  F* f = static_cast<F*>(buf);
  if (DISPENSO_EXPECT(run, true)) {
    (*f)();
  }
  f->~F();
}

template <typename F>
void relocateInline(void* src, void* dst) noexcept {
  F* f = static_cast<F*>(src);
  new (dst) F(std::move(*f));
  f->~F();
}

// Invoke for spilled functors: buffer holds a void* pointing to pool allocation.
template <size_t kBufferSize, typename F>
void invokeSpill(void* buf, bool run) {
  void* ptr = *static_cast<void**>(buf);
  F* f = static_cast<F*>(ptr);
  if (DISPENSO_EXPECT(run, true)) {
    (*f)();
  }
  f->~F();
  deallocSmallBuffer<kBufferSize>(ptr);
}

// One operations table per callable type. Function-local statics in inline functions are shared
// across translation units, and constexpr makes them constant-initialized, with no guard.
template <typename F>
inline const OnceCallableOps* inlineOps() {
  static constexpr OnceCallableOps kOps = {
      &invokeInline<F>, std::is_trivially_copyable<F>::value ? nullptr : &relocateInline<F>};
  return &kOps;
}

template <size_t kBufferSize, typename F>
inline const OnceCallableOps* spillOps() {
  static constexpr OnceCallableOps kOps = {&invokeSpill<kBufferSize, F>, nullptr};
  return &kOps;
}

// The operations of an empty or consumed OnceFunction. Never invoked; relocate is null, so moving
// one is a plain byte copy.
inline const OnceCallableOps* emptyOps() {
  static constexpr OnceCallableOps kOps = {nullptr, nullptr};
  return &kOps;
}

template <typename F>
inline const OnceCallableOps*
createOnceCallableImpl(F&& f, void* inlineBuf, std::true_type /*inline*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  new (inlineBuf) FNoRef(std::forward<F>(f));
  return inlineOps<FNoRef>();
}

template <typename F>
inline const OnceCallableOps*
createOnceCallableImpl(F&& f, void* inlineBuf, std::false_type /*spill*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  constexpr size_t kAllocSize =
      static_cast<size_t>(nextPow2(std::max(sizeof(FNoRef), alignof(FNoRef))));
  void* ptr = allocSmallBuffer<kAllocSize>();
#if defined(__cpp_exceptions)
  try {
    new (ptr) FNoRef(std::forward<F>(f));
  } catch (...) {
    deallocSmallBuffer<kAllocSize>(ptr);
    throw;
  }
#else
  new (ptr) FNoRef(std::forward<F>(f));
#endif // __cpp_exceptions
  *static_cast<void**>(inlineBuf) = ptr;
  return spillOps<kAllocSize, FNoRef>();
}

template <typename F>
inline const OnceCallableOps* createOnceCallable(F&& f, void* inlineBuf) {
  using FNoRef = typename std::remove_reference<F>::type;
  return createOnceCallableImpl(
      std::forward<F>(f),
      inlineBuf,
      std::integral_constant<
          bool,
          (sizeof(FNoRef) <= kOnceFunctionInlineSize && alignof(FNoRef) <= 64 &&
           (std::is_trivially_copyable<FNoRef>::value ||
            std::is_nothrow_move_constructible<FNoRef>::value))>{});
}

} // namespace detail
} // namespace dispenso
