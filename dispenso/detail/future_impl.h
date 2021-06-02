// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <future>

#include <dispenso/completion_event.h>
#include <dispenso/once_function.h>
#include <dispenso/task_set.h>

namespace dispenso {
namespace detail {

template <typename Result>
class FutureImplResultMember {
 public:
  Result& result() const {
#if defined(__cpp_exceptions)
    if (exception_) {
      std::rethrow_exception(exception_);
    }
#endif // __cpp_exceptions
    return *reinterpret_cast<Result*>(resultBuf_);
  }

  ~FutureImplResultMember() {
#if defined(__cpp_exceptions)
    // We don't want to call the destructor if we never set the result.
    if (exception_) {
      return;
    }
#endif // __cpp_exceptions

    reinterpret_cast<Result*>(resultBuf_)->~Result();
  }

 protected:
  template <typename Func>
  void runToResult(Func&& f) {
#if defined(__cpp_exceptions)
    try {
      new (resultBuf_) Result(f());
    } catch (...) {
      exception_ = std::current_exception();
    }
#else
    new (resultBuf_) Result(f());
#endif // __cpp_exceptions
  }

  template <typename T>
  void setAsResult(T&& t) {
    new (resultBuf_) Result(std::forward<T>(t));
  }

  alignas(Result) mutable char resultBuf_[sizeof(Result)];

#if defined(__cpp_exceptions)
  std::exception_ptr exception_;
#endif // __cpp_exceptions
};

template <typename Result>
class FutureImplResultMember<Result&> {
 protected:
  template <typename Func>
  void runToResult(Func&& f) {
#if defined(__cpp_exceptions)
    try {
      result_ = &f();
    } catch (...) {
      exception_ = std::current_exception();
    }
#else
    result_ = &f();
#endif // __cpp_exceptions
  }

  Result& result() const {
#if defined(__cpp_exceptions)
    if (exception_) {
      std::rethrow_exception(exception_);
    }
#endif // __cpp_exceptions
    return *result_;
  }

  void setAsResult(Result* res) {
    result_ = res;
  }

  Result* result_;

#if defined(__cpp_exceptions)
  std::exception_ptr exception_;
#endif // __cpp_exceptions
};

template <>
class FutureImplResultMember<void> {
 protected:
  template <typename Func>
  void runToResult(Func&& f) {
#if defined(__cpp_exceptions)
    try {
      f();
    } catch (...) {
      exception_ = std::current_exception();
    }
#else
    f();
#endif // __cpp_exceptions
  }
  void result() const {
#if defined(__cpp_exceptions)
    if (exception_) {
      std::rethrow_exception(exception_);
    }
#endif // __cpp_exceptions
  }
  void setAsResult() const {}

#if defined(__cpp_exceptions)
  std::exception_ptr exception_;
#endif // __cpp_exceptions
};

template <typename Result>
class FutureBase;

template <typename Result>
class FutureImplBase : private FutureImplResultMember<Result>, public OnceCallable {
 public:
  enum Status { kNotStarted, kRunning, kReady };

  using FutureImplResultMember<Result>::result;
  using FutureImplResultMember<Result>::setAsResult;
  using FutureImplResultMember<Result>::runToResult;

  bool ready() {
    return status_.intrusiveStatus().load(std::memory_order_acquire) == kReady;
  }

  void run() override {
    (void)run(kNotStarted);
    decRefCountMaybeDestroy();
  }

  void wait() {
    if (waitCommon(true)) {
      return;
    }
    status_.wait(kReady);
  }

  template <class Rep, class Period>
  std::future_status waitFor(const std::chrono::duration<Rep, Period>& timeoutDuration) {
    if (waitCommon(allowInline_) || status_.waitFor(kReady, timeoutDuration)) {
      return std::future_status::ready;
    }
    return std::future_status::timeout;
  }

  template <class Clock, class Duration>
  std::future_status waitUntil(const std::chrono::time_point<Clock, Duration>& timeoutTime) {
    if (waitCommon(allowInline_) || status_.waitUntil(kReady, timeoutTime)) {
      return std::future_status::ready;
    }
    return std::future_status::timeout;
  }

  void incRefCount() {
    refCount_.fetch_add(1, std::memory_order_acquire);
  }

  void decRefCountMaybeDestroy() {
    DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(&refCount_);
    if (refCount_.fetch_sub(1, std::memory_order_release) == 1) {
      DISPENSO_TSAN_ANNOTATE_HAPPENS_AFTER(&refCount_);
      dealloc();
    }
  }

  void setReady() {
    status_.intrusiveStatus().store(kReady, std::memory_order_release);
    refCount_.store(1, std::memory_order_release);
  }

  void setAllowInline(bool allow) {
    allowInline_ = allow;
  }

  void setTaskSetCounter(std::atomic<ssize_t>* tsc) {
    taskSetCounter_ = tsc;
  }

 protected:
  bool run(int s) {
    while (s == kNotStarted) {
      if (status_.intrusiveStatus().compare_exchange_weak(s, kRunning, std::memory_order_acq_rel)) {
        runFunc();
        status_.notify(kReady);
        if (taskSetCounter_) {
          //  If we want TaskSet::wait to imply Future::is_ready(),
          //  we need to signal that *after* setting the Future status to ready.
          taskSetCounter_->fetch_sub(1, std::memory_order_release);
        }
        tryExecuteThenChain();
        return true;
      }
    }
    return false;
  }

  using ThenChainInvoke = void(void*, void*);

  struct ThenChain {
    ThenChain* next;
    void* impl;
    void* schedulable;
    ThenChainInvoke* invoke;

    ThenChain* scheduleDestroyAndGetNext() {
      invoke(impl, schedulable);
      constexpr size_t kImplSize = nextPowerOfTwo(sizeof(this));
      auto* ret = this->next;
      deallocSmallBuffer<kImplSize>(this);
      return ret;
    }
  };

  void tryExecuteThenChain() {
    ThenChain* head = thenChain_.load(std::memory_order_acquire);
    // While the chain contains anything, let's try to get it and dispatch the chain.
    while (head) {
      if (thenChain_.compare_exchange_weak(head, nullptr, std::memory_order_acq_rel)) {
        // Managed to exchange with head, value of thenChain_ now points to null chain.
        // Head points to the implicit list of items to be executed.
        while (head) {
          head = head->scheduleDestroyAndGetNext();
        }
        // At this point, the list is exhausted, so we exit the outer loop too.  It would be valid
        // to get head again and try all over again, but it is guaranteed that if another link was
        // added concurrently to our dispatch, that thread will attempt to dispatch it's own chain,
        // so we just exit the loop instead.
      }
    }
  }

  template <typename SomeFutureImpl, typename Schedulable>
  static void thenChainInvokeAsync(void* implv, void* schedulablev) {
    SomeFutureImpl* impl = reinterpret_cast<SomeFutureImpl*>(implv);
    Schedulable* schedulable = reinterpret_cast<Schedulable*>(schedulablev);
    schedulable->schedule(OnceFunction(impl, true), ForceQueuingTag());
  }

  template <typename SomeFutureImpl, typename Schedulable>
  static void thenChainInvoke(void* implv, void* schedulablev) {
    SomeFutureImpl* impl = reinterpret_cast<SomeFutureImpl*>(implv);
    Schedulable* schedulable = reinterpret_cast<Schedulable*>(schedulablev);
    schedulable->schedule(OnceFunction(impl, true));
  }

  template <typename SomeFutureImpl, typename Schedulable>
  void addToThenChainOrExecute(SomeFutureImpl* impl, Schedulable& sched, std::launch asyncPolicy) {
    if (status_.intrusiveStatus().load(std::memory_order_acquire) == kReady) {
      if ((asyncPolicy & std::launch::async) == std::launch::async) {
        sched.schedule(OnceFunction(impl, true), ForceQueuingTag());
      } else {
        sched.schedule(OnceFunction(impl, true));
      }
      return;
    }

    constexpr size_t kImplSize = nextPowerOfTwo(sizeof(ThenChain));
    auto* buffer = allocSmallBuffer<kImplSize>();
    ThenChain* link = reinterpret_cast<ThenChain*>(buffer);
    link->impl = impl;
    using NonConstSchedulable = std::remove_const_t<Schedulable>;
    NonConstSchedulable* nonConstSched = const_cast<NonConstSchedulable*>(&sched);
    link->schedulable = nonConstSched;
    if ((asyncPolicy & std::launch::async) == std::launch::async) {
      link->invoke = thenChainInvokeAsync<SomeFutureImpl, Schedulable>;
    } else {
      link->invoke = thenChainInvoke<SomeFutureImpl, Schedulable>;
    }
    link->next = thenChain_.load(std::memory_order_acquire);
    while (true) {
      if (thenChain_.compare_exchange_weak(link->next, link, std::memory_order_acq_rel)) {
        // Successfully swung func to head.
        break;
      }
    }

    // Okay, one last thing.  It is possible that we added to the thenChain just after
    // tryExecuteThenChain was called from run(). We still need to ensure that this work is kicked
    // off, so just double check here, and execute if that may have happened.
    if (status_.intrusiveStatus().load(std::memory_order_acquire) == kReady) {
      tryExecuteThenChain();
    }
  }

  inline bool waitCommon(bool allowInline) {
    int s = status_.intrusiveStatus().load(std::memory_order_acquire);
    return s == kReady || (allowInline && run(s));
  }

  virtual void runFunc() = 0;

  virtual void dealloc() = 0;

  virtual ~FutureImplBase() = default;

  bool allowInline_;

  CompletionEventImpl status_{kNotStarted};
  std::atomic<uint32_t> refCount_{2};
  std::atomic<ssize_t>* taskSetCounter_{nullptr};

  std::atomic<ThenChain*> thenChain_{nullptr};

  template <typename R, typename T>
  friend FutureImplBase<R&>* createValueFutureImplReady(std::reference_wrapper<T> t);
  template <typename R, typename T>
  friend FutureImplBase<R>* createRefFutureImplReady(T&& t);

  template <typename R>
  friend class FutureBase;
}; // namespace detail

template <size_t kBufferSize, typename F, typename Result>
class FutureImplSmall : public FutureImplBase<Result> {
 public:
  FutureImplSmall(F&& f) {
    new (func_) F(std::move(f));
  }

 protected:
  void runFunc() override {
    F* f = reinterpret_cast<F*>(func_);
    this->runToResult(*f);
    f->~F();
  }
  void dealloc() override {
    this->~FutureImplSmall();
    deallocSmallBuffer<kBufferSize>(this);
  }

  ~FutureImplSmall() override = default;

 private:
  alignas(F) char func_[sizeof(F)];
};

template <size_t kBufferSize, typename Result>
class FutureImplSmall<kBufferSize, void, Result> : public FutureImplBase<Result> {
 protected:
  void runFunc() override {}
  void dealloc() override {
    this->~FutureImplSmall();
    deallocSmallBuffer<kBufferSize>(this);
  }

  ~FutureImplSmall() override = default;
};

template <typename F, typename Result>
class FutureImplAlloc : public FutureImplBase<Result> {
 public:
  FutureImplAlloc(F&& f) {
    new (func_) F(std::move(f));
  }

 protected:
  void runFunc() override {
    F* f = reinterpret_cast<F*>(func_);
    this->runToResult(*f);
    f->~F();
  }
  void dealloc() override {
    this->~FutureImplAlloc();
    alignedFree(this);
  }

  ~FutureImplAlloc() override = default;

 private:
  alignas(F) char func_[sizeof(F)];
};

template <typename Result>
class FutureImplAlloc<void, Result> : public FutureImplBase<Result> {
 protected:
  void runFunc() override {}
  void dealloc() override {
    this->~FutureImplAlloc();
    alignedFree(this);
  }

  ~FutureImplAlloc() override = default;
};

template <typename Result, typename F>
inline FutureImplBase<Result>*
createFutureImpl(F&& f, bool allowInline, std::atomic<ssize_t>* taskSetCounter) {
  using FNoRef = typename std::remove_reference<F>::type;
  constexpr size_t kImplSize = nextPowerOfTwo(sizeof(FutureImplSmall<16, FNoRef, Result>));
  using SmallT = FutureImplSmall<kImplSize, FNoRef, Result>;
  using AllocT = FutureImplAlloc<FNoRef, Result>;

  FutureImplBase<Result>* ret;
  if (sizeof(SmallT) <= kMaxSmallBufferSize) {
    ret = new (allocSmallBuffer<kImplSize>()) SmallT(std::forward<F>(f));
  } else {
    ret = new (alignedMalloc(sizeof(AllocT))) AllocT(std::forward<F>(f));
  }
  ret->setAllowInline(allowInline);
  ret->setTaskSetCounter(taskSetCounter);
  return ret;
}

template <typename Result, typename T>
inline FutureImplBase<Result>* createValueFutureImplReady(T&& t) {
  constexpr size_t kImplSize = nextPowerOfTwo(sizeof(FutureImplSmall<16, void, Result>));
  using SmallT = FutureImplSmall<kImplSize, void, Result>;
  using AllocT = FutureImplAlloc<void, Result>;

  FutureImplBase<Result>* retval;

  if (sizeof(SmallT) <= kMaxSmallBufferSize) {
    retval = new (allocSmallBuffer<kImplSize>()) SmallT();
  } else {
    retval = new (alignedMalloc(sizeof(AllocT))) AllocT();
  }
  retval->setAsResult(std::forward<T>(t));
  retval->setReady();
  return retval;
}

template <typename X>
inline FutureImplBase<X&>* createRefFutureImplReady(std::reference_wrapper<X> x) {
  constexpr size_t kImplSize = nextPowerOfTwo(sizeof(FutureImplSmall<16, void, X&>));
  using SmallT = FutureImplSmall<kImplSize, void, X&>;
  FutureImplBase<X&>* retval = new (allocSmallBuffer<kImplSize>()) SmallT();
  retval->setAsResult(&x.get());
  retval->setReady();
  return retval;
}

inline FutureImplBase<void>* createVoidFutureImplReady() {
  constexpr size_t kImplSize = nextPowerOfTwo(sizeof(FutureImplSmall<16, void, void>));
  using SmallT = FutureImplSmall<kImplSize, void, void>;
  FutureImplBase<void>* retval = new (allocSmallBuffer<kImplSize>()) SmallT();
  retval->setReady();
  return retval;
}

template <typename TaskSetType>
struct TaskSetInterceptionInvoker {
  TaskSetInterceptionInvoker(TaskSetType& ts) : taskSet(ts) {}

  void schedule(OnceFunction f) {
    savedOffFn = std::move(f);
  }
  void schedule(OnceFunction f, ForceQueuingTag) {
    savedOffFn = std::move(f);
  }
  TaskSetType& taskSet;
  OnceFunction savedOffFn;
};

template <typename Result>
class FutureBase {
 protected:
  FutureBase() noexcept : impl_(nullptr) {}
  FutureBase(FutureBase&& f) noexcept : impl_(f.impl_) {
    f.impl_ = nullptr;
  }
  FutureBase(const FutureBase& f) noexcept {
    impl_ = f.impl_;
    if (impl_) {
      impl_->incRefCount();
    }
  }
  template <typename F, typename Schedulable>
  FutureBase(F&& f, Schedulable& schedulable, std::launch asyncPolicy, std::launch deferredPolicy)
      : impl_(createFutureImpl<Result>(
            std::forward<F>(f),
            (deferredPolicy & std::launch::deferred) == std::launch::deferred,
            nullptr)) {
    if ((asyncPolicy & std::launch::async) == std::launch::async) {
      schedulable.schedule(OnceFunction(impl_, true), ForceQueuingTag());
    } else {
      schedulable.schedule(OnceFunction(impl_, true));
    }
  }

  template <typename F>
  FutureBase(F&& f, TaskSet& taskSet, std::launch asyncPolicy, std::launch deferredPolicy)
      : impl_(createFutureImpl<Result>(
            std::forward<F>(f),
            (deferredPolicy & std::launch::deferred) == std::launch::deferred,
            &taskSet.outstandingTaskCount_)) {
    taskSet.outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    if ((asyncPolicy & std::launch::async) == std::launch::async) {
      taskSet.pool().schedule(OnceFunction(impl_, true), ForceQueuingTag());
    } else {
      taskSet.pool().schedule(OnceFunction(impl_, true));
    }
  }

  template <typename F>
  FutureBase(F&& f, ConcurrentTaskSet& taskSet, std::launch asyncPolicy, std::launch deferredPolicy)
      : impl_(createFutureImpl<Result>(
            std::forward<F>(f),
            (deferredPolicy & std::launch::deferred) == std::launch::deferred,
            &taskSet.outstandingTaskCount_)) {
    taskSet.outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    if ((asyncPolicy & std::launch::async) == std::launch::async) {
      taskSet.pool().schedule(OnceFunction(impl_, true), ForceQueuingTag());
    } else {
      taskSet.pool().schedule(OnceFunction(impl_, true));
    }
  }

  template <typename F, typename TaskSetType>
  FutureBase(
      F&& f,
      TaskSetInterceptionInvoker<TaskSetType>& invoker,
      std::launch asyncPolicy,
      std::launch deferredPolicy)
      : impl_(createFutureImpl<Result>(
            std::forward<F>(f),
            (deferredPolicy & std::launch::deferred) == std::launch::deferred,
            &invoker.taskSet.outstandingTaskCount_)) {
    invoker.taskSet.outstandingTaskCount_.fetch_add(1, std::memory_order_acquire);
    if ((asyncPolicy & std::launch::async) == std::launch::async) {
      invoker.schedule(OnceFunction(impl_, true), ForceQueuingTag());
    } else {
      invoker.schedule(OnceFunction(impl_, true));
    }
  }

  void move(FutureBase&& f) noexcept {
    if (impl_ == f.impl_) {
      return;
    } else if (impl_) {
      impl_->decRefCountMaybeDestroy();
    }
    impl_ = f.impl_;
    f.impl_ = nullptr;
  }
  void copy(const FutureBase& f) {
    if (impl_ != f.impl_) {
      if (impl_ != nullptr) {
        impl_->decRefCountMaybeDestroy();
      }
      impl_ = f.impl_;
      if (impl_) {
        impl_->incRefCount();
      }
    }
  }
  ~FutureBase() {
    if (impl_) {
      impl_->decRefCountMaybeDestroy();
    }
  }
  bool valid() const noexcept {
    return impl_;
  }
  bool is_ready() const {
    assertValid();
    return impl_->ready();
  }
  void wait() const {
    assertValid();
    impl_->wait();
  }
  template <class Rep, class Period>
  std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeoutDuration) const {
    assertValid();
    return impl_->waitFor(timeoutDuration);
  }
  template <class Clock, class Duration>
  std::future_status wait_until(const std::chrono::time_point<Clock, Duration>& timeoutTime) const {
    assertValid();
    return impl_->waitUntil(timeoutTime);
  }

  template <typename RetResult, typename F, typename Schedulable>
  FutureImplBase<RetResult>*
  thenImpl(F&& f, Schedulable& sched, std::launch asyncPolicy, std::launch deferredPolicy);

  template <typename RetResult, typename F>
  FutureImplBase<RetResult>*
  thenImpl(F&& f, TaskSet& sched, std::launch asyncPolicy, std::launch deferredPolicy);
  template <typename RetResult, typename F>
  FutureImplBase<RetResult>*
  thenImpl(F&& f, ConcurrentTaskSet& sched, std::launch asyncPolicy, std::launch deferredPolicy);

  template <typename RetResult, typename F, typename Schedulable>
  FutureImplBase<RetResult>*
  thenMoveImpl(F&& f, Schedulable& sched, std::launch asyncPolicy, std::launch deferredPolicy);

  template <typename RetResult, typename F>
  FutureImplBase<RetResult>*
  thenMoveImpl(F&& f, TaskSet& sched, std::launch asyncPolicy, std::launch deferredPolicy);
  template <typename RetResult, typename F>
  FutureImplBase<RetResult>* thenMoveImpl(
      F&& f,
      ConcurrentTaskSet& sched,
      std::launch asyncPolicy,
      std::launch deferredPolicy);

#if defined DISPENSO_DEBUG
  void assertValid() const {
    assert(valid());
  }
#else
  void assertValid() const {}
#endif // DISPENSO_DEBUG

  mutable FutureImplBase<Result>* impl_;
};

struct ReadyTag {};

} // namespace detail
} // namespace dispenso
