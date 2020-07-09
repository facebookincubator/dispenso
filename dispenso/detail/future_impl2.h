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

template <typename VecType>
struct WhenAllShared {
  VecType vec;
  std::atomic<size_t> count;
  OnceFunction f;

  template <typename InputIt>
  WhenAllShared(InputIt first, InputIt last) : vec(first, last), count(vec.size()) {}
};

struct InterceptionInvoker {
  void schedule(OnceFunction f) {
    savedOffFn = std::move(f);
  }
  void schedule(OnceFunction f, ForceQueuingTag) {
    savedOffFn = std::move(f);
  }
  OnceFunction savedOffFn;
};

} // namespace detail

template <typename InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>> when_all(
    InputIt first,
    InputIt last) {
  using VecType = std::vector<typename std::iterator_traits<InputIt>::value_type>;
  using ResultFuture = Future<VecType>;

  if (first == last) {
    return make_ready_future(VecType());
  }

  // TODO(bbudge): Can write something faster than make_shared using SmallBufferAllocator.
  auto shared = std::make_shared<detail::WhenAllShared<VecType>>(first, last);

  auto whenComplete = [shared]() -> VecType {
    for (auto& f : shared->vec) {
      if (0 == shared->count.load(std::memory_order_relaxed)) {
        break;
      }
      f.wait();
    }
    return std::move(shared->vec);
  };

  detail::InterceptionInvoker interceptor;
  ResultFuture res(std::move(whenComplete), interceptor);

  shared->f = std::move(interceptor.savedOffFn);
  for (auto& s : shared->vec) {
    s.then(
        [shared](auto&&) {
          if (shared->count.fetch_sub(1, std::memory_order_relaxed) == 1) {
            shared->f();
          }
        },
        kImmediateInvoker);
  }

  return res;
}
} // namespace dispenso
