/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace dispenso {
namespace detail {

// Implementation note regarding then():
//
// At some point we tried to add Future::then()&&.  The impetus was that we figured we could avoid
// making an extra copy of the *this, avoiding primarily an atomic reference increment and
// decrement.  This lead to a 15% speedup in benchmarks.  The problem was that we were retaining a
// copy of the bare pointer impl_ in order to call addToThenChainOrExecute.  This could lead to a
// case where our impl_ only had one reference going into addToThenChainOrExecute, and internal to
// that, the func is scheduled, which creates a race between the execution and potential cleanup of
// the future internals, against completion of addToThenChainOrExecute.
//
// In reality this leads to *requiring* a bump of the reference count prior to
// addToThenChainOrExecute, and a decrement afterwards.  This isn't at all cheaper (in benchmarks)
// than just making the copy.  The moral of the story?  Don't bother with Future::then()&&.

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
      std::move(func), (deferredPolicy & std::launch::deferred) == std::launch::deferred, nullptr);
  impl_->addToThenChainOrExecute(retImpl, sched, asyncPolicy);
  return retImpl;
}

template <typename Result>
template <typename RetResult, typename F>
FutureImplBase<RetResult>* FutureBase<Result>::thenImpl(
    F&& f,
    TaskSet& sched,
    std::launch asyncPolicy,
    std::launch deferredPolicy) {
  Future<Result> copy(*this);
  auto func = [f = std::move(f), copy = std::move(copy)]() mutable -> RetResult {
    copy.wait();
    return f(std::move(copy));
  };

  sched.outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
  auto* retImpl = createFutureImpl<RetResult>(
      std::move(func),
      (deferredPolicy & std::launch::deferred) == std::launch::deferred,
      &sched.outstandingTaskCount_);
  impl_->addToThenChainOrExecute(retImpl, sched.pool(), asyncPolicy);
  return retImpl;
}

template <typename Result>
template <typename RetResult, typename F>
FutureImplBase<RetResult>* FutureBase<Result>::thenImpl(
    F&& f,
    ConcurrentTaskSet& sched,
    std::launch asyncPolicy,
    std::launch deferredPolicy) {
  Future<Result> copy(*this);
  auto func = [f = std::move(f), copy = std::move(copy)]() mutable -> RetResult {
    copy.wait();
    return f(std::move(copy));
  };

  sched.outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
  auto* retImpl = createFutureImpl<RetResult>(
      std::move(func),
      (deferredPolicy & std::launch::deferred) == std::launch::deferred,
      &sched.outstandingTaskCount_);
  impl_->addToThenChainOrExecute(retImpl, sched.pool(), asyncPolicy);
  return retImpl;
}

template <size_t index, typename... Ts>
struct ForEachApply {
  template <typename F>
  void operator()(std::tuple<Ts...>& t, F f) {
    if (f(std::get<index>(t))) {
      ForEachApply<index - 1, Ts...>{}(t, f);
    }
  }
};

template <typename... Ts>
struct ForEachApply<size_t{0}, Ts...> {
  template <typename F>
  void operator()(std::tuple<Ts...>& t, F f) {
    f(std::get<0>(t));
  }
};

template <typename F, typename... Ts>
void forEach(std::tuple<Ts...>& t, F f) {
  constexpr size_t size = std::tuple_size<std::tuple<Ts...>>::value;
  ForEachApply<size - 1, Ts...>{}(t, f);
}

template <typename VecType>
struct WhenAllSharedVec {
  VecType vec;
  std::atomic<size_t> count;
  OnceFunction f;

  template <typename InputIt, typename T = typename std::iterator_traits<InputIt>::value_type>
  WhenAllSharedVec(InputIt first, InputIt last) : vec(first, last), count(vec.size()) {}
};

template <typename Tuple>
struct WhenAllSharedTuple {
  Tuple tuple;
  std::atomic<size_t> count;
  OnceFunction f;
  template <typename... Types>
  WhenAllSharedTuple(Types&&... args)
      : tuple(std::make_tuple(std::forward<Types>(args)...)),
        count(std::tuple_size<Tuple>::value) {}
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

template <typename VecType>
struct WhenAnySharedVec {
  VecType vec;
  std::atomic<size_t> winner;
  OnceFunction f;

  template <typename InputIt>
  WhenAnySharedVec(InputIt first, InputIt last) : vec(first, last), winner(SIZE_MAX) {}
};

template <typename Tuple>
struct WhenAnySharedTuple {
  Tuple tuple;
  std::atomic<size_t> winner;
  OnceFunction f;
  template <typename... Types>
  WhenAnySharedTuple(Types&&... args)
      : tuple(std::make_tuple(std::forward<Types>(args)...)), winner(SIZE_MAX) {}
};

template <typename Invoker, typename... Futures>
auto whenAllTuple(Invoker& invoker, Futures&&... futures)
    -> Future<std::tuple<std::decay_t<Futures>...>> {
  using TupleType = std::tuple<std::decay_t<Futures>...>;
  using ResultFuture = Future<TupleType>;

  // TODO(bbudge): Can write something faster than make_shared using SmallBufferAllocator.
  auto shared =
      std::make_shared<detail::WhenAllSharedTuple<TupleType>>(std::forward<Futures>(futures)...);

  auto whenComplete = [shared]() -> TupleType {
    forEach(shared->tuple, [&shared](auto& future) {
      if (0 == shared->count.load(std::memory_order_acquire)) {
        return false;
      }
      future.wait();
      return true;
    });
    return std::move(shared->tuple);
  };

  ResultFuture res(std::move(whenComplete), invoker);

  shared->f = std::move(invoker.savedOffFn);
  // Avoid sequencing issue by getting the reference prior to the std::move.
  auto& tuple = shared->tuple;
  forEach(tuple, [shared = std::move(shared)](auto& future) {
    future.then(
        [shared](auto&&) {
          if (shared->count.fetch_sub(1, std::memory_order_release) == 1) {
            shared->f();
          }
        },
        kImmediateInvoker);
    return true;
  });

  return res;
}

template <typename Invoker, typename InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>>
whenAllIterators(Invoker& invoker, InputIt first, InputIt last) {
  using VecType = std::vector<typename std::iterator_traits<InputIt>::value_type>;
  using ResultFuture = Future<VecType>;

  if (first == last) {
    return make_ready_future(VecType());
  }

  // TODO(bbudge): Can write something faster than make_shared using SmallBufferAllocator.
  auto shared = std::make_shared<detail::WhenAllSharedVec<VecType>>(first, last);

  auto whenComplete = [shared]() -> VecType {
    for (auto& f : shared->vec) {
      if (0 == shared->count.load(std::memory_order_acquire)) {
        break;
      }
      f.wait();
    }
    return std::move(shared->vec);
  };

  ResultFuture res(std::move(whenComplete), invoker);

  shared->f = std::move(invoker.savedOffFn);
  for (auto& s : shared->vec) {
    s.then(
        [shared](auto&&) {
          if (shared->count.fetch_sub(1, std::memory_order_release) == 1) {
            shared->f();
          }
        },
        kImmediateInvoker);
  }

  return res;
}

template <typename Invoker, typename... Futures>
auto whenAnyTuple(Invoker& invoker, Futures&&... futures) -> Future<size_t> {
  using TupleType = std::tuple<std::decay_t<Futures>...>;
  using ResultFuture = Future<size_t>;

  auto shared =
      std::make_shared<detail::WhenAnySharedTuple<TupleType>>(std::forward<Futures>(futures)...);

  auto whenComplete = [shared]() -> size_t {
    size_t w = shared->winner.load(std::memory_order_acquire);
    if (w != SIZE_MAX) {
      return w;
    }
    // Inline-execution path: see whenAnyIterators for the rationale. Wait on a
    // single input (the first one forEach visits) and claim it if no .then
    // callback has already picked a winner. The --idx scheme matches the
    // registration loop below, so the index we assign is consistent with what
    // that input's .then would have set. We do not force strictly-first
    // completion here.
    size_t idx = std::tuple_size<TupleType>::value;
    forEach(shared->tuple, [&shared, &idx](auto& future) {
      --idx;
      future.wait();
      size_t expected = SIZE_MAX;
      shared->winner.compare_exchange_strong(expected, idx, std::memory_order_acq_rel);
      return false; // one input resolved ⇒ winner is now set; stop iterating.
    });
    return shared->winner.load(std::memory_order_acquire);
  };

  ResultFuture res(std::move(whenComplete), invoker);

  shared->f = std::move(invoker.savedOffFn);
  size_t idx = std::tuple_size<TupleType>::value;
  auto& tuple = shared->tuple;
  forEach(tuple, [shared, &idx](auto& future) {
    --idx;
    size_t myIdx = idx;
    future.then(
        [shared, myIdx](auto&&) {
          size_t expected = SIZE_MAX;
          if (shared->winner.compare_exchange_strong(expected, myIdx, std::memory_order_acq_rel)) {
            shared->f();
          }
        },
        kImmediateInvoker);
    return true;
  });

  return res;
}

template <typename Invoker, typename InputIt>
Future<size_t> whenAnyIterators(Invoker& invoker, InputIt first, InputIt last) {
  using VecType = std::vector<typename std::iterator_traits<InputIt>::value_type>;
  using ResultFuture = Future<size_t>;

  if (first == last) {
    // Empty range: no winner possible. Return SIZE_MAX as sentinel.
    return make_ready_future(static_cast<size_t>(SIZE_MAX));
  }

  auto shared = std::make_shared<detail::WhenAnySharedVec<VecType>>(first, last);

  auto whenComplete = [shared]() -> size_t {
    size_t w = shared->winner.load(std::memory_order_acquire);
    if (w != SIZE_MAX) {
      return w;
    }
    // Inline-execution path: get() ran the result inline before any .then
    // callback fired shared->f. Block until at least one input is resolved,
    // then return the winner. We deliberately wait on a single input rather
    // than racing all of them for the strictly-first completion: when_any makes
    // no such promise, and a wait-any over the inputs would only pessimize this
    // rare fallback. If a .then callback claims an earlier-completing winner
    // while we wait, our CAS fails and we return that winner instead. vec is
    // non-empty here (the empty range returned above).
    shared->vec[0].wait();
    size_t expected = SIZE_MAX;
    shared->winner.compare_exchange_strong(expected, size_t{0}, std::memory_order_acq_rel);
    return shared->winner.load(std::memory_order_acquire);
  };

  ResultFuture res(std::move(whenComplete), invoker);

  shared->f = std::move(invoker.savedOffFn);
  for (size_t i = 0; i < shared->vec.size(); ++i) {
    auto& s = shared->vec[i];
    s.then(
        [shared, i](auto&&) {
          size_t expected = SIZE_MAX;
          if (shared->winner.compare_exchange_strong(expected, i, std::memory_order_acq_rel)) {
            shared->f();
          }
        },
        kImmediateInvoker);
  }

  return res;
}

} // namespace detail

template <typename InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>> when_all(
    InputIt first,
    InputIt last) {
  detail::InterceptionInvoker interceptor;
  return whenAllIterators(interceptor, first, last);
}

template <typename InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>>
when_all(TaskSet& taskSet, InputIt first, InputIt last) {
  detail::TaskSetInterceptionInvoker<TaskSet> interceptor(taskSet);
  return whenAllIterators(interceptor, first, last);
}

template <typename InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>>
when_all(ConcurrentTaskSet& taskSet, InputIt first, InputIt last) {
  detail::TaskSetInterceptionInvoker<ConcurrentTaskSet> interceptor(taskSet);
  return whenAllIterators(interceptor, first, last);
}

inline auto when_all() -> Future<std::tuple<>> {
  return make_ready_future(std::tuple<>());
}

inline auto when_all(TaskSet&) -> Future<std::tuple<>> {
  return make_ready_future(std::tuple<>());
}

inline auto when_all(ConcurrentTaskSet&) -> Future<std::tuple<>> {
  return make_ready_future(std::tuple<>());
}

template <class... Futures>
auto when_all(Futures&&... futures) -> Future<std::tuple<std::decay_t<Futures>...>> {
  detail::InterceptionInvoker interceptor;
  return whenAllTuple(interceptor, std::forward<Futures>(futures)...);
}

template <typename... Futures>
auto when_all(TaskSet& taskSet, Futures&&... futures)
    -> Future<std::tuple<std::decay_t<Futures>...>> {
  detail::TaskSetInterceptionInvoker<TaskSet> interceptor(taskSet);
  return whenAllTuple(interceptor, std::forward<Futures>(futures)...);
}

template <typename... Futures>
auto when_all(ConcurrentTaskSet& taskSet, Futures&&... futures)
    -> Future<std::tuple<std::decay_t<Futures>...>> {
  detail::TaskSetInterceptionInvoker<ConcurrentTaskSet> interceptor(taskSet);
  return whenAllTuple(interceptor, std::forward<Futures>(futures)...);
}

template <typename InputIt, typename>
Future<size_t> when_any(InputIt first, InputIt last) {
  detail::InterceptionInvoker interceptor;
  return detail::whenAnyIterators(interceptor, first, last);
}

template <typename InputIt, typename>
Future<size_t> when_any(TaskSet& taskSet, InputIt first, InputIt last) {
  detail::TaskSetInterceptionInvoker<TaskSet> interceptor(taskSet);
  return detail::whenAnyIterators(interceptor, first, last);
}

template <typename InputIt, typename>
Future<size_t> when_any(ConcurrentTaskSet& taskSet, InputIt first, InputIt last) {
  detail::TaskSetInterceptionInvoker<ConcurrentTaskSet> interceptor(taskSet);
  return detail::whenAnyIterators(interceptor, first, last);
}

inline Future<size_t> when_any() {
  return make_ready_future(static_cast<size_t>(SIZE_MAX));
}

inline Future<size_t> when_any(TaskSet&) {
  return make_ready_future(static_cast<size_t>(SIZE_MAX));
}

inline Future<size_t> when_any(ConcurrentTaskSet&) {
  return make_ready_future(static_cast<size_t>(SIZE_MAX));
}

template <typename... Futures>
auto when_any(Futures&&... futures) -> Future<size_t> {
  detail::InterceptionInvoker interceptor;
  return detail::whenAnyTuple(interceptor, std::forward<Futures>(futures)...);
}

template <typename... Futures>
auto when_any(TaskSet& taskSet, Futures&&... futures) -> Future<size_t> {
  detail::TaskSetInterceptionInvoker<TaskSet> interceptor(taskSet);
  return detail::whenAnyTuple(interceptor, std::forward<Futures>(futures)...);
}

template <typename... Futures>
auto when_any(ConcurrentTaskSet& taskSet, Futures&&... futures) -> Future<size_t> {
  detail::TaskSetInterceptionInvoker<ConcurrentTaskSet> interceptor(taskSet);
  return detail::whenAnyTuple(interceptor, std::forward<Futures>(futures)...);
}

} // namespace dispenso
