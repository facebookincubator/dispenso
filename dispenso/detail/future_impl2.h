// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

namespace dispenso {
namespace detail {
template <typename Result>
template <typename RetResult, typename F, typename Schedulable>
FutureImplBase<RetResult>* FutureBase<Result>::thenImpl(
    F&& f,
    Schedulable& sched,
    std::launch asyncPolicy,
    std::launch deferredPolicy) {
  Future<Result> copy(*this);
  auto func = [f = std::move(f), copy = std::move(copy)]() mutable -> RetResult {
    copy.wait();
    return f(std::move(copy));
  };

  auto* retImpl = createFutureImpl<RetResult>(
      std::move(func), (deferredPolicy & std::launch::deferred) == std::launch::deferred);
  impl_->addToThenChainOrExecute(retImpl, sched, asyncPolicy);
  return retImpl;
}
} // namespace detail
} // namespace dispenso
