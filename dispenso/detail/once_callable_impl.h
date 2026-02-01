/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/math.h>
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
inline OnceCallable* createOnceCallable(F&& f) {
  using FNoRef = typename std::remove_reference<F>::type;

  constexpr size_t kImplSize = static_cast<size_t>(nextPow2(sizeof(OnceCallableImpl<16, FNoRef>)));

  return new (allocSmallBuffer<kImplSize>())
      OnceCallableImpl<kImplSize, FNoRef>(std::forward<F>(f));
}

} // namespace detail
} // namespace dispenso
