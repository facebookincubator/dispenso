/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/math.h>
#include <dispenso/platform.h>
#include <dispenso/small_buffer_allocator.h>

namespace dispenso {
namespace detail {

// Inline storage size: 64-byte OnceFunction minus 8-byte invoke pointer.
// Covers most parallel_for lambdas (up to 7 pointers of captures).
//
// IMPORTANT: Inline-stored functors are relocated via memcpy (see OnceFunction
// move ctor/assign). This is safe for types with position-independent value
// representation: scalars, pointers, references, POD aggregates, and typical
// lambdas (including captures of std::string, std::vector, std::shared_ptr,
// and std::function — their internal storage is memcpy-safe so long as the
// captured value within them is itself position-independent). It is NOT safe
// for types with self-referential pointers — e.g., std::list (intrusive
// sentinel), or any struct that holds a pointer back into itself. C++ has no
// standard is_trivially_relocatable trait yet (P1144), so this contract is
// implicit — callers must not store self-referential types in OnceFunction.
constexpr size_t kOnceFunctionInlineSize = 56;

struct OnceCallableData {
  void (*invoke)(void*, bool run);
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
inline OnceCallableData createOnceCallableImpl(F&& f, void* inlineBuf, std::true_type /*inline*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  new (inlineBuf) FNoRef(std::forward<F>(f));
  return {&invokeInline<FNoRef>};
}

template <typename F>
inline OnceCallableData createOnceCallableImpl(F&& f, void* inlineBuf, std::false_type /*spill*/) {
  using FNoRef = typename std::remove_reference<F>::type;
  constexpr size_t kAllocSize =
      static_cast<size_t>(nextPow2(std::max(sizeof(FNoRef), alignof(FNoRef))));
  void* ptr = allocSmallBuffer<kAllocSize>();
  new (ptr) FNoRef(std::forward<F>(f));
  *static_cast<void**>(inlineBuf) = ptr;
  return {&invokeSpill<kAllocSize, FNoRef>};
}

template <typename F>
inline OnceCallableData createOnceCallable(F&& f, void* inlineBuf) {
  using FNoRef = typename std::remove_reference<F>::type;
  return createOnceCallableImpl(
      std::forward<F>(f),
      inlineBuf,
      std::integral_constant<
          bool,
          (sizeof(FNoRef) <= kOnceFunctionInlineSize && alignof(FNoRef) <= 64)>{});
}

} // namespace detail
} // namespace dispenso
