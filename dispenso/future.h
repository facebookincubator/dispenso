// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <functional>

#include <dispenso/detail/future_impl.h>

namespace dispenso {

// See https://en.cppreference.com/w/cpp/experimental/future for details on the API.

// TODO(bbudge): Implement when_all(), when_any(), ?unwrapping constructor? functionality.

/**
 *  A <code>std::launch</code> policy specifying we won't force asynchronicity.  Opposite of
 * <code>std::launch::async</code>
 **/
constexpr std::launch kNotAsync = static_cast<std::launch>(0);
/**
 * A <code>std::launch</code> policy specifying we won't allow <code>Future::wait_for</code> and
 * <code>Future::wait_util</code> to invoke the Future's underlying functor.  Opposite of
 * <code>std::launch::deferred</code>
 **/
constexpr std::launch kNotDeferred = static_cast<std::launch>(0);

/**
 * A class fullfilling the Schedulable concept that immediately invokes the functor.  This can be
 * used in place of <code>ThreadPool</code> or <code>TaskSet</code> with <code>Future</code>s at
 * construction or through <code>then</code>.
 **/
class ImmediateInvoker {
 public:
  /**
   * Schedule a functor to be executed.  It will be invoked immediately.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  void schedule(F&& f) const {
    f();
  }

  /**
   * Schedule a functor to be executed.  It is a bit oxymoronical to call this function, since
   * ForceQueuingTag will have no effect, and it's use is discouraged.
   *
   **/
  template <typename F>
  void schedule(F&& f, ForceQueuingTag) const {
    // TODO(bbudge): Are there actually reasons we may want to allow this function to be called?
    // assert(false);
    f();
  }
};

constexpr ImmediateInvoker kImmediateInvoker;

/**
 * A class fullfilling the Schedulable concept that always invokes on a new thread.  This can be
 * used in place of <code>ThreadPool</code> or <code>TaskSet</code> with <code>Future</code>s at
 * construction or through <code>then</code>.
 **/
class NewThreadInvoker {
 public:
  /**
   * Schedule a functor to be executed on a new thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  void schedule(F&& f) const {
    schedule(std::forward<F>(f), ForceQueuingTag());
  }
  /**
   * Schedule a functor to be executed on a new thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  void schedule(F&& f, ForceQueuingTag) const {
    std::thread thread(std::forward<F>(f));
    thread.detach();
  }
};

constexpr NewThreadInvoker kNewThreadInvoker;

/**
 * A class that implements a hybrid of the interfaces for std::experimental::future, and
 * std::experimental::shared_future.  <code>Future</code> acts like a <code>shared_future</code>,
 * but includes the <code>share</code> function in order to enable generic code that may use the
 * function. See https://en.cppreference.com/w/cpp/experimental/future for more details on the API,
 * but some differences are notable.
 *
 * 1. <code>dispenso::Future</code> are created and set into executors through their constructor, or
 *    through <code>dispenso::async</code>
 * 2. <code>dispenso::future</code> can be shared around like
 *    <code>std::experimental::shared_future</code>, and wait/get functions may be called
 *    concurrently from multiple threads.
 *
 * Future is thread-safe to call into, with the usual caveats (one thread may not be calling
 * anything on the Future while another thread assigns to it or destructs it).  It is thread-safe to
 * assign or destruct on one copy of a Future while making calls on another copy of a Future
 * with the same backing state.
 *
 * Because Future is designed to work well with the dispenso TaskSet and ThreadPool, Future
 * aggressively work steals in <code>get</code> and <code>wait</code> functions.  This prevents
 * deadlock due to thread pool resource starvation, similar to how TaskSets avoid that flavor of
 * deadlock.  It is important to note that deadlock is still possible due to traditional cyclic
 * dependency, e.g. Future A and Future B which wait on each other.  Just as with a cyclic mutex
 * locking requirement, cyclic Future waits are considered programmer error.
 **/
template <typename Result>
class Future : detail::FutureBase<Result> {
  using Base = detail::FutureBase<Result>;

 public:
  /**
   * Construct an invalid Future.
   **/
  Future() noexcept : Base() {}

  /**
   * Move constructor
   *
   * @param f The future to move from.
   **/
  Future(Future&& f) noexcept : Base(std::move(f)) {}
  Future(Base&& f) noexcept : Base(std::move(f)) {}

  /**
   * Copy construct a Future.
   *
   * @param f The existing future to reference.  This essentially increments the reference count on
   * the future's backing state.
   **/
  Future(const Future& f) noexcept : Base(f) {}
  Future(const Base& f) noexcept : Base(f) {}

  /**
   * Construct a Future with callable and a schedulable (e.g. a ThreadPool or TaskSet), and ensure
   * it is scheduled according to the launch policy.
   *
   * @param f a functor with signature <code>Result(void)</code> to be invoked in order for
   * <code>get</code> to return a valid value.
   * @param schedulable A <code>TaskSet</code>, <code>ThreadPool</code>, or similar type that has
   * function <code>schedule</code> that takes a functor with signature <code>void()</code> and
   * <code>ForceQueuingTag</code>.
   * @param asyncPolicy If <code>std::launch::async</code>, the functor will be scheduled through to
   * the underlying thread pool work queue.
   * @param deferredPolicy If <code>std::launch::deferred</code>, <code>wait_for</code> and
   * <code>wait_until</code> may invoke the functor from their calling thread.
   **/
  template <typename F, typename Schedulable>
  Future(
      F&& f,
      Schedulable& schedulable,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred)
      : Base(std::forward<F>(f), schedulable, asyncPolicy, deferredPolicy) {}

  /**
   * Move a Future
   *
   * @param f The Future whose backing state will be transferred to this Future.
   **/
  Future& operator=(Future&& f) noexcept {
    Base::move(reinterpret_cast<Base&&>(f));
    return *this;
  }
  /**
   * Copy a Future, which increments the underlying state reference count.
   *
   * @param f The Future whose backing state will be referenced by this Future.
   **/
  Future& operator=(const Future& f) {
    Base::copy(f);
    return *this;
  }

  /**
   * Destruct a Future.  This decrements the shared reference count (if any), and ensures the
   * backing state is destroyed when no more references exist to the backing state.
   **/
  ~Future() = default;

  /**
   * Is this Future valid?
   *
   * @return <code>true</code> if the Future was constructed with a functor/schedulable, with result
   * value, or indirectly from a Future that was constructed one of those ways.  <code>false</code>
   * if the Future was constructed via the default constructor, or indrectly from a Future that was
   * constructed that way.
   **/
  bool valid() const noexcept {
    return Base::valid();
  }

  /**
   * Is the Future ready?
   *
   * @return <code>true</code> if the value associated with this Future has already be computed, and
   * <code>get</code> can return the value immediately.  Returns <false> if the functor is logically
   * queued, or is in progress.
   **/
  bool is_ready() const {
    return Base::is_ready();
  }

  /**
   * Wait until <code>is_ready</code> is <code>true</code>.
   *
   * @note This function will invoke the functor if it is still logically queued.
   **/
  void wait() const {
    Base::wait();
  }

  /**
   * Wait until <code>is_ready</code> is <code>true</code> or until timing out.
   *
   * @param timeoutDuration The length of time to wait until timing out.  This is relative to the
   * current point in time.
   * @return <code>std::future_status::ready</code> if <code>get</code> can return immediately,
   * <code>std::future_status::timeout</code> if the function timed out while waiting.
   *
   * @note This function may invoke the functor if it is still logically queued, and
   * <code>std::launch::deferred</code> was specified at construction.
   **/
  template <class Rep, class Period>
  std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeoutDuration) const {
    return Base::wait_for(timeoutDuration);
  }

  /**
   * Wait until <code>is_ready</code> is <code>true</code> or until timing out.
   *
   * @param timeoutTime The absolute time to wait until timing out.
   * @return <code>std::future_status::ready</code> if <code>get</code> can return immediately,
   * <code>std::future_status::timeout</code> if the function timed out while waiting.
   *
   * @note This function may invoke the functor if it is still logically queued, and
   * <code>std::launch::deferred</code> was specified at construction.
   **/
  template <class Clock, class Duration>
  std::future_status wait_until(const std::chrono::time_point<Clock, Duration>& timeoutTime) const {
    return Base::wait_until(timeoutTime);
  }

  /**
   * Provide a shared future.  <code>share</code> is here only to provide compatible api with
   * <code>std::experimental::future</code>, but Future already works like std::shared_future.
   **/
  Future share() {
    return std::move(*this);
  }

  /**
   * Get the result of the future functor.  This function blocks until the result is ready.
   *
   * @return A const reference to the result's value.
   **/
  const Result& get() const {
    wait();
    return this->impl_->result();
  }

  /**
   * Schedule a functor to be invoked upon reaching <code>is_ready<code> status, and return
   * a future that will hold the result of the functor.
   *
   * @param f The functor to be executed whose result will be available in the returned
   * <code>Future</code>.  This should have signature <code>Unpecified(Future<Result>&&)</code>.
   * @param sched The Schedulable in which to run the functor.
   * @param asyncPolicy if <code>std::launch::async</code> then the functor will attempt to be
   * queued on the backing work queue of the Schedulable (if any).
   * @param deferredPolicy if <code>std::launch::deferred</code>, <code>wait_for</code> and
   * <code>wait_until</code> of the returned future will be allowed to invoke the functor, otherwise
   * not.
   *
   * @return A future containing the result of the functor.
   **/
  template <typename F, typename Schedulable>
  Future<detail::AsyncResultOf<F, Future<Result>&&>> then(
      F&& f,
      Schedulable& sched,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred) {
    Future<detail::AsyncResultOf<F, Future<Result>&&>> retFuture;
    retFuture.impl_ = this->template thenImpl<detail::AsyncResultOf<F, Future<Result>&&>>(
        std::forward<F>(f), sched, asyncPolicy, deferredPolicy);
    return retFuture;
  }
  template <typename F>
  Future<detail::AsyncResultOf<F, Future<Result>&&>> then(F&& f) {
    return then(std::forward<F>(f), globalThreadPool(), kNotAsync, std::launch::deferred);
  }

 private:
  template <typename T>
  Future(T&& t, detail::ReadyTag) {
    this->impl_ = detail::createValueFutureImplReady<Result>(std::forward<T>(t));
  }

  template <typename T>
  friend Future<std::decay_t<T>> make_ready_future(T&& t);

  template <typename R>
  friend class Future;
};

template <typename Result>
class Future<Result&> : detail::FutureBase<Result&> {
  using Base = detail::FutureBase<Result&>;

 public:
  Future() noexcept : Base() {}
  Future(Future&& f) noexcept : Base(std::move(f)) {}
  Future(Base&& f) noexcept : Base(std::move(f)) {}
  Future(const Future& f) noexcept : Base(f) {}
  Future(const Base& f) noexcept : Base(f) {}
  template <typename F, typename Schedulable>
  Future(
      F&& f,
      Schedulable& schedulable,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred)
      : Base(std::forward<F>(f), schedulable, asyncPolicy, deferredPolicy) {}
  Future& operator=(Future&& f) noexcept {
    Base::move(reinterpret_cast<Base&&>(f));
    return *this;
  }
  Future& operator=(const Future& f) {
    Base::copy(f);
    return *this;
  }
  ~Future() = default;
  using Base::is_ready;
  using Base::valid;
  using Base::wait;
  using Base::wait_for;
  using Base::wait_until;

  Future share() {
    return std::move(*this);
  }

  /**
   * Get the result of the future functor.  This function blocks until the result is ready.
   *
   * @return Access to the underlying reference.
   **/
  Result& get() const {
    wait();
    return this->impl_->result();
  }

  template <typename F, typename Schedulable>
  Future<detail::AsyncResultOf<F, Future<Result&>&&>> then(
      F&& f,
      Schedulable& sched,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred) {
    Future<detail::AsyncResultOf<F, Future<Result&>&&>> retFuture;
    retFuture.impl_ = this->template thenImpl<detail::AsyncResultOf<F, Future<Result&>&&>>(
        std::forward<F>(f), sched, asyncPolicy, deferredPolicy);
    return retFuture;
  }
  template <typename F>
  Future<detail::AsyncResultOf<F, Future<Result&>&&>> then(F&& f) {
    return then(std::forward<F>(f), globalThreadPool(), kNotAsync, std::launch::deferred);
  }

 private:
  template <typename T>
  Future(std::reference_wrapper<T> t, detail::ReadyTag) {
    this->impl_ = detail::createRefFutureImplReady<Result>(t);
  }

  template <typename X>
  friend Future<X&> make_ready_future(std::reference_wrapper<X> x);

  template <typename R>
  friend class Future;
};

template <>
class Future<void> : detail::FutureBase<void> {
  using Base = detail::FutureBase<void>;

 public:
  Future() noexcept : Base() {}
  Future(Future&& f) noexcept : Base(std::move(f)) {}
  Future(Base&& f) noexcept : Base(std::move(f)) {}
  Future(const Future& f) noexcept : Base(f) {}
  Future(const Base& f) noexcept : Base(f) {}
  template <typename F, typename Schedulable>
  Future(
      F&& f,
      Schedulable& schedulable,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred)
      : Base(std::forward<F>(f), schedulable, asyncPolicy, deferredPolicy) {}
  Future& operator=(Future&& f) noexcept {
    Base::move(reinterpret_cast<Base&&>(f));
    return *this;
  }
  Future& operator=(const Future& f) {
    Base::copy(f);
    return *this;
  }
  ~Future() = default;
  using Base::is_ready;
  using Base::valid;
  using Base::wait;
  using Base::wait_for;
  using Base::wait_until;

  Future share() {
    return std::move(*this);
  }

  /**
   * Block until the functor has been called.
   **/
  void get() const {
    wait();
    this->impl_->result();
  }

  template <typename F, typename Schedulable>
  Future<detail::AsyncResultOf<F, Future<void>&&>> then(
      F&& f,
      Schedulable& sched,
      std::launch asyncPolicy = kNotAsync,
      std::launch deferredPolicy = std::launch::deferred) {
    Future<detail::AsyncResultOf<F, Future<void>&&>> retFuture;
    retFuture.impl_ = this->template thenImpl<detail::AsyncResultOf<F, Future<void>&&>>(
        std::forward<F>(f), sched, asyncPolicy, deferredPolicy);
    return retFuture;
  }
  template <typename F>
  Future<detail::AsyncResultOf<F, Future<void>&&>> then(F&& f) {
    return then(std::forward<F>(f), globalThreadPool(), kNotAsync, std::launch::deferred);
  }

 private:
  Future(detail::ReadyTag) {
    impl_ = detail::createVoidFutureImplReady();
  }

  friend Future<void> make_ready_future();

  template <typename R>
  friend class Future;
};

// TODO(bbudge): Determine if we should
// a. Expand launch policies, and logically inherit from std::launch and
// b. Whether async should truly mean on a new thread.
// For now we will treat std::launch::async such that we pass ForceQueuingTag

/**
 * Invoke a functor through the global dispenso thread pool.
 *
 * @param policy The bitmask policy for when/how the functor can be invoked.
 * <code>std::launch::async</code> will result in the functor being forced onto a ThreadPool work
 * queue.  <code>std::launch::deferred</code> specifies that <code>Future::wait_for</code> and
 * <code>Future::wait_until</code> may invoke the functor.
 * @param f The functor to be passed, or a function to be executed
 * @param args The remaining arguments that will be passed to <code>f</code>
 **/
template <class F, class... Args>
inline Future<detail::AsyncResultOf<F, Args...>> async(std::launch policy, F&& f, Args&&... args) {
  return Future<detail::AsyncResultOf<F, Args...>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...), globalThreadPool(), policy);
}

/**
 * Invoke a functor through the global dispenso thread pool.
 *
 * @param f The functor to be passed, or a function to be executed
 * @param args The remaining arguments that will be passed to <code>f</code>
 **/
template <class F, class... Args>
inline Future<detail::AsyncResultOf<F, Args...>> async(F&& f, Args&&... args) {
  return ::dispenso::async(std::launch::deferred, std::forward<F>(f), std::forward<Args>(args)...);
}

/**
 * Make a <code>Future</code> in a ready state with the value passed into
 * <code>make_ready_future</code>
 *
 * @param t the value to use to create the returned future.
 **/
template <typename T>
inline Future<std::decay_t<T>> make_ready_future(T&& t) {
  return Future<std::decay_t<T>>(std::forward<T>(t), detail::ReadyTag());
}

/**
 * Make a <code>Future</code> in a ready state with the <code>reference_wrapper</code> passed into
 * <code>make_ready_future</code>
 *
 * @param x the wrapped reference to use to create the returned future.
 **/
template <typename X>
inline Future<X&> make_ready_future(std::reference_wrapper<X> x) {
  return Future<X&>(x, detail::ReadyTag());
}

/**
 * Make a <code>Future<void></code> in a ready state.
 *
 **/
inline Future<void> make_ready_future() {
  return Future<void>(detail::ReadyTag());
}

/**
 * Take a collection of futures, and return a future which will be ready when all input futures are
 * ready.
 *
 * @param first An iterator to the start of the future collection.
 * @param last An iterator to the end of the future collection.
 *
 * @return A Future containing a vector holding copies of the input Futures.  The returned Future
 * will be in ready state when all input Futures are ready.
 *
 **/
template <class InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>> when_all(
    InputIt first,
    InputIt last);

/**
 * Take a specific set of futures, and return a future which will be ready when all input futures
 *are ready.
 *
 * @param futures A parameter pack of futures.
 *
 * @return A Future containing a tuple holding copies of the input Futures.  The returned Future
 * will be in ready state when all input Futures are ready.
 *
 **/
template <class... Futures>
auto when_all(Futures&&... futures) -> Future<std::tuple<std::decay_t<Futures>...>>;

/**
 * Take a collection of futures, and return a future which will be ready when all input futures are
 * ready.
 *
 * @param tastSet A task set to register with such that after this call,
 * <code>taskSet::wait()</code> implies that the resultant future <code>is_ready()</code>
 * @param first An iterator to the start of the future collection.
 * @param last An iterator to the end of the future collection.
 *
 * @return A Future containing a vector holding copies of the input Futures.  The returned Future
 * will be in ready state when all input Futures are ready.
 *
 **/
template <class InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>>
when_all(TaskSet& taskSet, InputIt first, InputIt last);
template <class InputIt>
Future<std::vector<typename std::iterator_traits<InputIt>::value_type>>
when_all(ConcurrentTaskSet& taskSet, InputIt first, InputIt last);

/**
 * Take a specific set of futures, and return a future which will be ready when all input futures
 *are ready.
 *
 * @param tastSet A task set to register with such that after this call,
 * <code>taskSet::wait()</code> implies that the resultant future <code>is_ready()</code>
 * @param futures A parameter pack of futures.
 *
 * @return A Future containing a tuple holding copies of the input Futures.  The returned Future
 * will be in ready state when all input Futures are ready.
 *
 **/
template <class... Futures>
auto when_all(TaskSet& taskSet, Futures&&... futures)
    -> Future<std::tuple<std::decay_t<Futures>...>>;

template <class... Futures>
auto when_all(ConcurrentTaskSet& taskSet, Futures&&... futures)
    -> Future<std::tuple<std::decay_t<Futures>...>>;
} // namespace dispenso

#include <dispenso/detail/future_impl2.h>
