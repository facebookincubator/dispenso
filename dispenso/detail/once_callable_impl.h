// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/small_buffer_allocator.h>

namespace dispenso {
namespace detail {

class OnceCallable {
 public:
  virtual void run() = 0;
  virtual ~OnceCallable() = default;
};

template <size_t kBufferSize, typename F>
class OnceCallableImpl : public OnceCallable {
 public:
  template <typename G>
  OnceCallableImpl(G&& f) : f_(std::forward<G>(f)) {}

  void run() override {
    f_();
    // This is admittedly playing nasty games here; however, the base class is empty, and we
    // completely control our own polymorphic existence.  No need to make the virtual base class
    // destructor get called (optimization).
    this->OnceCallableImpl::~OnceCallableImpl();
    deallocSmallBuffer<kBufferSize>(this);
  }

  ~OnceCallableImpl() override = default;

 private:
  F f_;
};

template <typename F>
class OnceCallableMalloc : public OnceCallable {
 public:
  template <typename G>
  OnceCallableMalloc(G&& f) {
    new (getF()) F(std::forward<G>(f));
  }
  void run() override {
    F* f = getF();
    (*f)();
    f->~F();
    ::free(this);
  }

  ~OnceCallableMalloc() override = default;

 private:
  F* getF() {
    // Ensure proper alignement of f
    constexpr uintptr_t kAlignMask = alignof(F) - 1;
    char* b = buffer_;
    b += kAlignMask;
    b = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(b) & (~kAlignMask));
    return reinterpret_cast<F*>(b);
  }

  char buffer_[sizeof(F) + std::max<ssize_t>(0, alignof(F) - sizeof(void*))];
};

constexpr uint32_t nextPowerOfTwo(uint32_t v) {
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

template <typename F>
inline OnceCallable* createOnceCallable(F&& f) {
  using FNoRef = typename std::remove_reference<F>::type;

  constexpr size_t kImplSize = nextPowerOfTwo(sizeof(OnceCallableImpl<16, FNoRef>));
  if (sizeof(OnceCallableImpl<kImplSize, FNoRef>) <= kMaxSmallBufferSize) {
    return new (allocSmallBuffer<kImplSize>())
        OnceCallableImpl<kImplSize, FNoRef>(std::forward<F>(f));
  }

  return new (::malloc(sizeof(OnceCallableMalloc<FNoRef>)))
      OnceCallableMalloc<FNoRef>(std::forward<F>(f));
}

} // namespace detail
} // namespace dispenso
