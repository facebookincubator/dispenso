/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if __cplusplus >= 201703L
#include <optional>
#endif // C++17

#include <dispenso/detail/completion_event_impl.h>
#include <dispenso/detail/op_result.h>
#include <dispenso/detail/result_of.h>
#include <dispenso/task_set.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {
namespace detail {

// Maximum depth for inline pipeline continuation before forcing a schedule through
// the thread pool. Prevents unbounded stack growth when the pool inlines execution.
static constexpr int kMaxPipelineInlineDepth = 32;

class LimitGatedScheduler {
 public:
  LimitGatedScheduler(ConcurrentTaskSet& tasks, ssize_t res)
      : impl_(new (alignedMalloc(sizeof(Impl), alignof(Impl))) Impl(tasks, res)) {}

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
    Impl(ConcurrentTaskSet& tasks, ssize_t res)
        : tasks_(tasks),
          resources_(res),
          unlimited_(res == std::numeric_limits<ssize_t>::max()),
          serial_(res == 1) {}

    template <typename F>
    void schedule(F&& fPipe) {
      outstanding_.fetch_add(1, std::memory_order_acq_rel);

      // RAII guard ensures outstanding_ is decremented even if an exception propagates.
      // Without this, wait() would hang spinning on outstanding_ reaching zero.
      struct OutstandingGuard {
        DISPENSO_INLINE ~OutstandingGuard() {
          outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        }
        std::atomic<size_t>& outstanding_;
      };

      if (unlimited_) {
        tasks_.schedule([this, fPipe = std::move(fPipe)]() mutable {
          OutstandingGuard oGuard{outstanding_};
          if (!tasks_.hasException()) {
            fPipe([]() {});
          }
        });
        return;
      }

      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
      queue_.enqueue([this, fPipe = std::move(fPipe)]() mutable {
        OutstandingGuard oGuard{outstanding_};

        // RAII guard releases the resource slot if the completion callback is never
        // called (i.e. if the user's stage function throws before reaching it).
        // Disarmed on the normal path when the completion callback fires.
        struct ResourceGuard {
          DISPENSO_INLINE ~ResourceGuard() {
            if (armed_) {
              resources_.fetch_add(1, std::memory_order_acq_rel);
            }
          }
          DISPENSO_INLINE void disarm() {
            armed_ = false;
          }
          std::atomic<ssize_t>& resources_;
          bool armed_{true};
        };
        ResourceGuard rGuard{resources_};

#if defined(__cpp_exceptions)
        try {
#endif
          fPipe([this, &rGuard]() {
            rGuard.disarm();
            OnceFunction func;
            DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
            bool deqd = queue_.try_dequeue(func);
            DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
            if (deqd) {
              // For serial stages (limit=1), execute continuation inline to reduce
              // thread pool scheduling overhead. This is safe because we're already
              // on a worker thread and only one item can be in the stage at a time.
              // Depth-limit to prevent unbounded stack growth when the pool inlines.
              if (serial_) {
                static DISPENSO_THREAD_LOCAL int depth = 0;
                if (depth < kMaxPipelineInlineDepth) {
                  struct DepthGuard {
                    DISPENSO_INLINE DepthGuard(int& d) : d_(d) {
                      ++d_;
                    }
                    DISPENSO_INLINE ~DepthGuard() {
                      --d_;
                    }
                    int& d_;
                  };
                  DepthGuard dGuard(depth);
                  func();
                } else {
                  tasks_.schedule(std::move(func), ForceQueuingTag());
                }
              } else {
                tasks_.schedule(std::move(func));
              }
            } else {
              resources_.fetch_add(1, std::memory_order_acq_rel);
            }
          });
#if defined(__cpp_exceptions)
        } catch (...) {
          tasks_.trySetCurrentException();
        }
#endif
      });
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();

      while (resources_.fetch_sub(1, std::memory_order_acq_rel) > 0) {
        OnceFunction func;
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
        bool deqd = queue_.try_dequeue(func);
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
        if (deqd) {
          tasks_.schedule(std::move(func));
        } else {
          break;
        }
      }
      resources_.fetch_add(1, std::memory_order_acq_rel);
    }

    void wait() {
      if (!unlimited_) {
        // LIMITED path: drain the local queue into the ConcurrentTaskSet, then
        // return. We do NOT need to spin on outstanding_ here because the
        // generator's wait() calls ConcurrentTaskSet::wait() after all stage
        // wait() calls complete, which ensures every scheduled task finishes.
        OnceFunction func;
        while (true) {
          DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
          bool deqd = queue_.try_dequeue(func);
          DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
          if (!deqd) {
            break;
          }
          // When an exception has been captured, drain remaining queued items
          // without executing them. Decrement outstanding_ for each discarded
          // item, and use cleanupNotRun() since OnceFunction only frees its
          // resources when called (not on destruction).
          if (tasks_.hasException()) {
            outstanding_.fetch_sub(1, std::memory_order_acq_rel);
            func.cleanupNotRun();
            continue;
          }
          // Spin until a resource slot is available. Check for exceptions
          // each iteration to avoid deadlocking when all pool threads have
          // finished and no one will release a resource.
          while (resources_.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
            resources_.fetch_add(1, std::memory_order_acq_rel);
            if (tasks_.hasException()) {
              outstanding_.fetch_sub(1, std::memory_order_acq_rel);
              func.cleanupNotRun();
              goto next_item;
            }
            std::this_thread::yield();
          }
          tasks_.schedule(std::move(func));
        next_item:;
        }
        return;
      }

      // Unified drain + wait loop. We must keep checking the local queue
      // because items can arrive after an earlier drain pass — this happens
      // when a completion callback's try_dequeue races with a concurrent
      // enqueue: the callback sees an empty queue and releases the resource
      // instead of chaining, leaving the item orphaned in the queue.
      // Re-checking on every iteration ensures we eventually dispatch it.
      while (outstanding_.load(std::memory_order_acquire)) {
        // For the unlimited path, items are scheduled directly to CTS (not
        // queued locally). When an exception occurs, remaining items are
        // already in CTS's pool queue wrapped by packageTask — CTS::wait()
        // will drain them. Break out here to avoid spinning.
        if (tasks_.hasException()) {
          break;
        }
        OnceFunction func;
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
        bool deqd = queue_.try_dequeue(func);
        DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
        if (deqd) {
          // Wait for resource to become available
          while (resources_.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
            resources_.fetch_add(1, std::memory_order_acq_rel);
            if (!tasks_.tryExecuteNext()) {
              std::this_thread::yield();
            }
          }
          tasks_.schedule(std::move(func));
        } else if (!tasks_.tryExecuteNext()) {
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
    const bool serial_;
  };

  struct Deleter {
    void operator()(Impl* r) {
      r->~Impl();
      alignedFree(r);
    }
  };

  std::unique_ptr<Impl, Deleter> impl_;
};

template <typename F>
struct Stage {
  Stage(F&& fIn, ssize_t limitIn) : f(std::move(fIn)), limit(limitIn) {}

  template <typename T>
  auto operator()(T&& t) {
    return f(std::forward<T>(t));
  }

  auto operator()() {
    return f();
  }

  F f;
  ssize_t limit;
};

template <typename T>
struct StageLimits {
  constexpr static ssize_t limit(const T& /*t*/) {
    return 1;
  }
};

template <typename T>
struct StageLimits<Stage<T>> {
  static ssize_t limit(const Stage<T>& t) {
    return std::max(ssize_t{1}, t.limit);
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
    ssize_t numThreads = std::max<ssize_t>(
        1, std::min(tasks_.numPoolThreads(), StageLimits<CurStage>::limit(stage_)));
    completion_ = std::make_unique<CompletionEventImpl>(static_cast<int>(numThreads));
    for (ssize_t i = 0; i < numThreads; ++i) {
      tasks_.schedule([this]() {
        // RAII guard ensures the completion event is signaled even if an exception
        // propagates out of pipeNext_.execute() (e.g. when ConcurrentTaskSet runs a
        // downstream stage inline and it throws). Without this, wait() would hang on
        // completion_->wait(0) because the count is never decremented.
        struct CompletionGuard {
          DISPENSO_INLINE ~CompletionGuard() {
            if (completion->intrusiveStatus().fetch_sub(1, std::memory_order_acq_rel) == 1) {
              completion->notify(0);
            }
          }
          CompletionEventImpl* completion;
        };
        CompletionGuard cGuard{completion_.get()};

        while (!tasks_.hasException()) {
          auto op = stage_();
          if (!op) {
            break;
          }
          pipeNext_.execute(std::move(op.value()));
        }
      });
    }
  }

  void wait() {
    completion_->wait(0);
    pipeNext_.wait();
    tasks_.wait();
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
        while (!tasks_.hasException() && stage_()) {
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
