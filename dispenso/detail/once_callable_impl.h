/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

template <typename F>
struct InlineOnceCallableOps {
  static constexpr OnceCallableOps kOps = {
      &invokeInline<F>,
      std::is_trivially_copyable<F>::value ? nullptr : &relocateInline<F>};
};
template <typename F>
constexpr OnceCallableOps InlineOnceCallableOps<F>::kOps;

template <size_t kBufferSize, typename F>
struct SpillOnceCallableOps {
  static constexpr OnceCallableOps kOps = {&invokeSpill<kBufferSize, F>, nullptr};
};
template <size_t kBufferSize, typename F>
constexpr OnceCallableOps SpillOnceCallableOps<kBufferSize, F>::kOps;

// A template so that the out-of-line definition C++14 needs can live in this header.
template <typename Unused = void>
struct EmptyOnceCallableOps {
  static constexpr OnceCallableOps kOps = {nullptr, nullptr};
};
template <typename Unused>
constexpr OnceCallableOps EmptyOnceCallableOps<Unused>::kOps;

template <typename F>
inline const OnceCallableOps*
createOnceCallableImpl(F&& f, void* inlineBuf, std::true_type /*inline*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  new (inlineBuf) FNoRef(std::forward<F>(f));
  return &InlineOnceCallableOps<FNoRef>::kOps;
}

template <typename F>
inline const OnceCallableOps*
createOnceCallableImpl(F&& f, void* inlineBuf, std::false_type /*spill*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  constexpr size_t kAllocSize =
      static_cast<size_t>(nextPow2(std::max(sizeof(FNoRef), alignof(FNoRef))));
  void* ptr = allocSmallBuffer<kAllocSize>();
  new (ptr) FNoRef(std::forward<F>(f));
  *static_cast<void**>(inlineBuf) = ptr;
  return &SpillOnceCallableOps<kAllocSize, FNoRef>::kOps;
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
