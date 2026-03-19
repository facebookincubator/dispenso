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

class OnceCallable {
 public:
  virtual void run() = 0;
  virtual void destroyOnly() = 0;
  virtual ~OnceCallable() = default;
};

struct OnceCallableData {
  void* data;
  void (*invoke)(void*, bool run);
};

template <size_t kBufferSize, typename F>
void invokeImpl(void* ptr, bool run) {
  F* f = static_cast<F*>(ptr);
  if (DISPENSO_EXPECT(run, true)) {
    (*f)();
  }
  f->~F();
  deallocSmallBuffer<kBufferSize>(ptr);
}

template <typename F>
inline OnceCallableData createOnceCallable(F&& f) {
  using FNoRef = typename std::remove_reference<F>::type;

  constexpr size_t kAllocSize =
      static_cast<size_t>(nextPow2(std::max(sizeof(FNoRef), alignof(FNoRef))));

  void* buf = allocSmallBuffer<kAllocSize>();
  new (buf) FNoRef(std::forward<F>(f));
  return {buf, &invokeImpl<kAllocSize, FNoRef>};
}

inline void runOnceCallable(void* ptr, bool run) {
  auto* callable = static_cast<OnceCallable*>(ptr);
  if (DISPENSO_EXPECT(run, true)) {
    callable->run();
  } else {
    callable->destroyOnly();
  }
}

} // namespace detail
} // namespace dispenso
