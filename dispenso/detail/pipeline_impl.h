// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#if __cplusplus >= 201703L
#include <optional>
#endif // C++17

#include <dispenso/detail/completion_event_impl.h>
#include <dispenso/detail/result_of.h>
#include <dispenso/task_set.h>

namespace dispenso {
namespace detail {

class LimitGatedScheduler {
 public:
  LimitGatedScheduler(ConcurrentTaskSet& tasks, size_t res)
      : impl_(std::make_unique<Impl>(tasks, res)) {}

  template <typename F>
  void schedule(F&& fPipe) {
    impl_->schedule(std::forward<F>(fPipe));
  }

  void wait() {
    impl_->wait();
  }

 private:
  // Put the guts within a unique_ptr to enable this type to be movable.
  class Impl {
   public:
    Impl(ConcurrentTaskSet& tasks, size_t res)
        : tasks_(tasks),
          resources_(std::min<size_t>(std::numeric_limits<ssize_t>::max(), res)),
          unlimited_(res >= std::numeric_limits<ssize_t>::max()) {}

    template <typename F>
    void schedule(F&& fPipe) {
      outstanding_.fetch_add(1, std::memory_order_acq_rel);

      if (unlimited_) {
        tasks_.schedule([this, fPipe = std::move(fPipe)]() mutable {
          fPipe([]() {});
          outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        });
        return;
      }

      queue_.enqueue([this, fPipe = std::move(fPipe)]() mutable {
        fPipe([this]() {
          OnceFunction func;
          if (queue_.try_dequeue(func)) {
            tasks_.schedule(std::move(func));
          } else {
            resources_.fetch_add(1, std::memory_order_acq_rel);
          }
        });
        outstanding_.fetch_sub(1, std::memory_order_acq_rel);
      });

      while (resources_.fetch_sub(1, std::memory_order_acq_rel) > 0) {
        OnceFunction func;
        if (queue_.try_dequeue(func)) {
          tasks_.schedule(std::move(func));
        } else {
          break;
        }
      }
      resources_.fetch_add(1, std::memory_order_acq_rel);
    }

    void wait() {
      if (!unlimited_) {
        OnceFunction func;
        while (queue_.try_dequeue(func)) {
          while (resources_.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
            std::this_thread::yield();
            resources_.fetch_add(1, std::memory_order_acq_rel);
          }
          tasks_.schedule(std::move(func));
        }
      }

      while (outstanding_.load(std::memory_order_acquire)) {
        if (!tasks_.tryExecuteNext()) {
          std::this_thread::yield();
        }
      }
    }

   private:
    ConcurrentTaskSet& tasks_;
    alignas(kCacheLineSize) std::atomic<ssize_t> resources_;
    alignas(kCacheLineSize) std::atomic<size_t> outstanding_{0};
    moodycamel::ConcurrentQueue<OnceFunction> queue_;
    // Note: In benchmarks, this doesn't seem to help very much (~1%), but using it should lower
    // resource requirements because the queue_ never needs to instantiate memory.
    const bool unlimited_;
  };
  std::unique_ptr<Impl> impl_;
};

template <typename T>
class OpResult {
 public:
  OpResult() : ptr_(nullptr) {}

  template <typename U>
  OpResult(U&& u) : ptr_(new (buf_) T(std::forward<U>(u))) {}

  OpResult(const OpResult<T>& oth) : ptr_(oth ? new (buf_) T(*oth.ptr_) : nullptr) {}

  OpResult(OpResult<T>&& oth) : ptr_(oth ? new (buf_) T(std::move(*oth.ptr_)) : nullptr) {
    oth.ptr_ = nullptr;
  }

  OpResult& operator=(const OpResult& oth) {
    if (&oth == this) {
      return *this;
    }
    if (ptr_) {
      ptr_->~T();
    }

    if (oth) {
      ptr_ = new (buf_) T(*oth.ptr_);
    } else {
      ptr_ = nullptr;
    }
    return *this;
  }

  OpResult& operator=(OpResult&& oth) {
    if (&oth == this) {
      return *this;
    }
    if (ptr_) {
      ptr_->~T();
    }

    if (oth) {
      ptr_ = new (buf_) T(std::move(*oth.ptr_));
      oth.ptr_ = nullptr;
    } else {
      ptr_ = nullptr;
    }

    return *this;
  }

  ~OpResult() {
    if (ptr_) {
      ptr_->~T();
    }
  }

  operator bool() const {
    return ptr_;
  }

  T& value() {
    return *ptr_;
  }

 private:
  alignas(T) char buf_[sizeof(T)];
  T* ptr_;
};

template <typename F>
struct Stage {
  Stage(F&& fIn, size_t limitIn) : f(std::move(fIn)), limit(limitIn) {}

  template <typename T>
  auto operator()(T&& t) {
    return f(std::forward<T>(t));
  }

  auto operator()() {
    return f();
  }

  F f;
  size_t limit;
};

template <typename T>
struct StageLimits {
  constexpr static size_t limit(const T& t) {
    return 1;
  }
};

template <typename T>
struct StageLimits<Stage<T>> {
  static size_t limit(const Stage<T>& t) {
    return std::max(size_t{1}, t.limit);
  }
};

enum class StageClass { kSingleStage, kGenerator, kOpTransform, kTransform, kSink };

template <typename T>
struct TransformTraits {
  static constexpr StageClass kStageClass = StageClass::kTransform;
};

#if __cplusplus >= 201703L
template <typename T>
struct TransformTraits<std::optional<T>> {
  static constexpr StageClass kStageClass = StageClass::kOpTransform;
};
#endif // C++17

template <typename T>
struct TransformTraits<OpResult<T>> {
  static constexpr StageClass kStageClass = StageClass::kOpTransform;
};

template <typename T>
struct OptionalStrippedTraits {
  using Type = T;
};

#if __cplusplus >= 201703L
template <typename T>
struct OptionalStrippedTraits<std::optional<T>> {
  using Type = T;
};
#endif // C++17

template <typename T>
struct OptionalStrippedTraits<OpResult<T>> {
  using Type = T;
};

struct SinkPipe {};

template <StageClass stageClass, typename CurStage, typename PipeNext>
class Pipe;

template <typename CurStage, typename PipeNext>
class TransformPipe {
 public:
  template <typename StageIn>
  TransformPipe(ConcurrentTaskSet& tasks, StageIn&& s, PipeNext&& n)
      : stage_(std::forward<StageIn>(s)),
        tasks_(tasks, StageLimits<CurStage>::limit(stage_)),
        pipeNext_(std::move(n)) {}

  void wait() {
    tasks_.wait();
    pipeNext_.wait();
  }

 protected:
  CurStage stage_;
  LimitGatedScheduler tasks_;
  PipeNext pipeNext_;
};

template <typename CurStage, typename PipeNext>
class Pipe<StageClass::kTransform, CurStage, PipeNext> : public TransformPipe<CurStage, PipeNext> {
 public:
  template <typename StageIn>
  Pipe(ConcurrentTaskSet& tasks, StageIn&& s, PipeNext&& n)
      : TransformPipe<CurStage, PipeNext>(tasks, std::forward<StageIn>(s), std::move(n)) {}

  template <typename Input>
  void execute(Input&& input) {
    this->tasks_.schedule([input = std::move(input), this](auto&& stageCompleteFunc) mutable {
      auto&& res = this->stage_(std::move(input));
      stageCompleteFunc();
      this->pipeNext_.execute(res);
    });
  }
};

template <typename CurStage, typename PipeNext>
class Pipe<StageClass::kOpTransform, CurStage, PipeNext>
    : public TransformPipe<CurStage, PipeNext> {
 public:
  template <typename StageIn>
  Pipe(ConcurrentTaskSet& tasks, StageIn&& s, PipeNext&& n)
      : TransformPipe<CurStage, PipeNext>(tasks, std::forward<StageIn>(s), std::move(n)) {}

  template <typename Input>
  void execute(Input&& input) {
    this->tasks_.schedule([input = std::move(input), this](auto&& stageCompleteFunc) mutable {
      auto op = this->stage_(std::move(input));
      stageCompleteFunc();
      if (op) {
        this->pipeNext_.execute(std::move(op.value()));
      }
    });
  }
};

template <typename CurStage, typename PipeNext>
class Pipe<StageClass::kGenerator, CurStage, PipeNext> {
 public:
  template <typename StageIn>
  Pipe(ConcurrentTaskSet& tasks, StageIn&& s, PipeNext&& n)
      : tasks_(tasks), stage_(std::forward<StageIn>(s)), pipeNext_(std::move(n)) {}

  void execute() {
    size_t numThreads = std::min(tasks_.numPoolThreads(), StageLimits<CurStage>::limit(stage_));
    completion_ = std::make_unique<CompletionEventImpl>(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
      tasks_.schedule([this]() {
        while (auto op = stage_()) {
          pipeNext_.execute(std::move(op.value()));
        }
        // fetch_sub returns the previous value, so if it was 1, that means no items are left.
        // notify wouldn't technically require the value to be set, since the underlying status
        // is already zero, but we just use the current notify interface, as this is unlikely to be
        // any kind of bottleneck.
        if (completion_->intrusiveStatus().fetch_sub(1, std::memory_order_acq_rel) == 1) {
          completion_->notify(0);
        }
      });
    }
  }

  void wait() {
    completion_->wait(0);
    pipeNext_.wait();
  }

 private:
  ConcurrentTaskSet& tasks_;
  std::unique_ptr<CompletionEventImpl> completion_;
  CurStage stage_;
  PipeNext pipeNext_;
};

template <typename CurStage>
class Pipe<StageClass::kSingleStage, CurStage, SinkPipe> {
 public:
  template <typename StageIn>
  Pipe(ConcurrentTaskSet& tasks, StageIn&& s) : tasks_(tasks), stage_(std::forward<StageIn>(s)) {}

  void execute() {
    size_t numThreads = std::min(tasks_.numPoolThreads(), StageLimits<CurStage>::limit(stage_));
    for (size_t i = 0; i < numThreads; ++i) {
      tasks_.schedule([this]() {
        while (stage_()) {
        }
      });
    }
  }

  void wait() {
    tasks_.wait();
  }

 private:
  ConcurrentTaskSet& tasks_;
  CurStage stage_;
};

template <typename CurStage>
class Pipe<StageClass::kSink, CurStage, SinkPipe> {
 public:
  template <typename StageIn>
  Pipe(ConcurrentTaskSet& tasks, StageIn&& s)
      : stage_(std::forward<StageIn>(s)), tasks_(tasks, StageLimits<CurStage>::limit(stage_)) {}

  template <typename Input>
  void execute(Input&& input) {
    tasks_.schedule([input = std::move(input), this](auto&& stageCompleteFunc) mutable {
      stage_(std::move(input));
      stageCompleteFunc();
    });
  }

  void wait() {
    tasks_.wait();
  }

 private:
  CurStage stage_;
  LimitGatedScheduler tasks_;
};

template <typename InputType, typename Stage0>
auto makePipesHelper(ConcurrentTaskSet& tasks, Stage0&& sCur) {
  return Pipe<StageClass::kSink, Stage0, SinkPipe>(tasks, std::forward<Stage0>(sCur));
}

template <typename InputType, typename Stage0, typename Stage1, typename... Stages>
auto makePipesHelper(
    ConcurrentTaskSet& tasks,
    Stage0&& sCur,
    Stage1&& sNext,
    Stages&&... sFollowing) {
  using Stage0Result = ResultOf<Stage0, typename OptionalStrippedTraits<InputType>::Type>;

  auto pipe = makePipesHelper<Stage0Result>(
      tasks, std::forward<Stage1>(sNext), std::forward<Stages>(sFollowing)...);

  constexpr StageClass kSc = TransformTraits<Stage0Result>::kStageClass;
  return Pipe<kSc, Stage0, decltype(pipe)>(tasks, std::forward<Stage0>(sCur), std::move(pipe));
}

template <typename Stage0, typename Stage1, typename... Stages>
auto makePipes(ConcurrentTaskSet& tasks, Stage0&& sCur, Stage1&& sNext, Stages&&... sFollowing) {
  auto pipe = makePipesHelper<ResultOf<Stage0>>(
      tasks, std::forward<Stage1>(sNext), std::forward<Stages>(sFollowing)...);
  return Pipe<StageClass::kGenerator, Stage0, decltype(pipe)>(
      tasks, std::forward<Stage0>(sCur), std::move(pipe));
}

template <typename Stage0>
auto makePipes(ConcurrentTaskSet& tasks, Stage0&& sCur) {
  return Pipe<StageClass::kSingleStage, Stage0, SinkPipe>(tasks, std::forward<Stage0>(sCur));
}

} // namespace detail
} // namespace dispenso
