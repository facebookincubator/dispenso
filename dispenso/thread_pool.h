/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file thread_pool.h
 * @ingroup group_core
 * A file providing ThreadPool.  This is the heart of dispenso.  All other scheduling paradigms,
 * including TaskSets, Futures, pipelines, and parallel loops, are built on top of ThreadPool.
 **/

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <mutex>
#include <thread>

#include <moodycamel/concurrentqueue.h>

#include <dispenso/concurrent_object_arena.h>
#include <dispenso/cpu_set.h>
#include <dispenso/detail/math.h>
#include <dispenso/detail/per_thread_info.h>
#include <dispenso/mpmc_ring_buffer.h>
#include <dispenso/once_function.h>
#include <dispenso/platform.h>
#include <dispenso/thread_pool_wake.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {

namespace detail {
// Relaxed atomic load with TSAN happens-after annotation.
// Semantically equivalent to memory_order_consume (which compilers promote to
// acquire). On real hardware, address-dependent loads are naturally ordered —
// you can't dereference a pointer before loading it. The relaxed load avoids
// the acquire fence cost on weakly-ordered architectures (ARM: ldr vs ldar).
// The TSAN annotation establishes the happens-before edge that the C++ abstract
// machine requires but hardware provides for free via dependency ordering.
template <typename T>
T* consumeLoad(std::atomic<T*>& ptr) {
  T* p = ptr.load(std::memory_order_relaxed);
  DISPENSO_TSAN_ANNOTATE_HAPPENS_AFTER(&ptr);
  return p;
}
} // namespace detail

namespace detail {
template <typename Result>
class FutureBase;
template <typename Result>
class FutureImplBase;
} // namespace detail

#if !defined(DISPENSO_WAKEUP_ENABLE)
#if defined(_WIN32) || defined(__linux__) || defined(__MACH__)
#define DISPENSO_WAKEUP_ENABLE 1
#else
#define DISPENSO_WAKEUP_ENABLE 0
#endif // platform
#endif // DISPENSO_WAKEUP_ENABLE

#if !defined(DISPENSO_POLL_PERIOD_US)
#if defined(_WIN32)
#define DISPENSO_POLL_PERIOD_US 1000
#else
#if !(DISPENSO_WAKEUP_ENABLE)
#define DISPENSO_POLL_PERIOD_US 200
#else
#define DISPENSO_POLL_PERIOD_US (1 << 15) // Determined empirically good on dual Xeon Linux
#endif // DISPENSO_WAKEUP_ENABLE
#endif // PLATFORM
#endif // DISPENSO_POLL_PERIOD_US

constexpr uint32_t kDefaultSleepLenUs = DISPENSO_POLL_PERIOD_US;

constexpr bool kDefaultWakeupEnable = DISPENSO_WAKEUP_ENABLE;

/**
 * A simple tag specifier that can be fed to TaskSets and
 * ThreadPools to denote that the current thread should never immediately execute a functor, but
 * rather, the functor should always be placed in the ThreadPool's queue.
 **/
struct ForceQueuingTag {};

/**
 * The basic executor for dispenso.  It provides typical thread pool functionality, plus allows work
 * stealing by related types (e.g. TaskSet, Future, etc...), which prevents deadlock when waiting
 * for pool-recursive tasks.
 */
class DISPENSO_CACHELINE_ALIGNED ThreadPool {
 public:
  /**
   * Construct a thread pool.
   *
   * @param n The number of threads to spawn at construction.
   * @param poolLoadMultiplier A parameter that specifies how overloaded the pool should be before
   * allowing the current thread to self-steal work.
   **/
  DISPENSO_DLL_ACCESS ThreadPool(size_t n, size_t poolLoadMultiplier = 32);

  /**
   * Enable or disable signaling wake functionality.  If enabled, this will try to ensure that
   * threads are woken up proactively when work has not been available and it becomes available.
   * This function is blocking and potentially very slow.  Repeated use is discouraged.
   *
   * @param enable If set true, turns on signaling wake.  If false, turns it off.
   * @param sleepDuration If enable is true, this is the length of time a thread will wait for a
   * signal before waking up.  If enable is false, this is the length of time a thread will sleep
   * between polling.
   *
   * @note It is highly recommended to leave signaling wake enabled on Windows platforms, as
   * sleeping/polling tends to perform poorly for intermittent workloads.  For Mac/Linux platforms,
   * it is okay to enable signaling wake, particularly if you wish to set a longer expected duration
   * between work.  If signaling wake is disabled, ensure sleepDuration is small (e.g. 200us) for
   * best performance.  Most users will not need to call this function, as defaults are reasonable.
   *
   *
   **/
  template <class Rep, class Period>
  void setSignalingWake(
      bool enable,
      const std::chrono::duration<Rep, Period>& sleepDuration =
          std::chrono::microseconds(kDefaultSleepLenUs)) {
    setSignalingWake(
        enable,
        static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(sleepDuration).count()));
  }

  /**
   * Change the number of threads backing the thread pool.  This is a blocking and potentially
   * slow operation, and repeatedly resizing is discouraged.
   *
   * @param n The number of threads in use after call completion
   **/
  DISPENSO_DLL_ACCESS void resize(ssize_t n) {
    std::lock_guard<std::mutex> lk(threadsMutex_);
    resizeLocked(n);
  }

  /**
   * Get the number of threads backing the pool.  If called concurrently to <code>resize</code>, the
   * number returned may be stale.
   *
   * @return The current number of threads backing the pool.
   **/
  ssize_t numThreads() const {
    return numThreads_.load(std::memory_order_relaxed);
  }

  /**
   * Schedule a functor to be executed.  If the pool's load factor is high, execution may happen
   * inline by the calling thread.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f);

  /**
   * Schedule a functor to be executed.  The functor will always be queued and executed by pool
   * threads.
   *
   * @param f The functor to be executed.  <code>f</code>'s signature must match void().  Best
   * performance will come from passing lambdas, other concrete functors, or OnceFunction, but
   * std::function or similarly type-erased objects will also work.
   **/
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedule(F&& f, ForceQueuingTag);

  /**
   * Schedule multiple functors to be executed in bulk.  This is more efficient than calling
   * schedule() multiple times when you have many tasks to submit, as it reduces atomic contention
   * and allows for better thread wakeup behavior.
   *
   * @param count The number of functors to schedule.
   * @param gen A generator functor that takes an index and returns a functor to execute.
   *            gen(i) will be called for i in [0, count) to produce each task.
   *
   * @note Work is enqueued in chunks and interleaved with inline execution based on the pool's
   *       load factor, preventing both pool thread starvation and excessive queueing overhead.
   **/
  template <typename Generator>
  void scheduleBulk(size_t count, Generator&& gen);

  /**
   * Destruct the pool.  This destructor is blocking until all queued work is completed.  It is
   * illegal to call the destructor while any other thread makes calls to the pool (as is generally
   * the case with C++ classes).
   **/
  DISPENSO_DLL_ACCESS ~ThreadPool();

  /**
   * RAII handle that asks the pool's worker threads to stay awake (skip the
   * sleep transition at the end of their spin window) while at least one
   * AwakeRef is alive on the pool.
   *
   * Use this around bursts of fine-grained scheduling (e.g. parallel_for)
   * where workers may run out of immediate work mid-burst and would otherwise
   * sleep — paying ~3-5us futex wake latency just to be re-woken on the next
   * task. With AwakeRef held, workers keep spinning instead.
   *
   * Cost: one fetch_add at construction, one fetch_sub at destruction, one
   * relaxed load in the worker's sleep-decision path. Move-only (non-copyable)
   * to allow transferring ownership while keeping a unique-owner invariant.
   */
  class AwakeRef {
   public:
    AwakeRef() = default;
    explicit AwakeRef(ThreadPool* pool) : pool_(pool) {
      if (pool_) {
        pool_->keepAwakeCount_.fetch_add(1, std::memory_order_acq_rel);
      }
    }
    AwakeRef(const AwakeRef&) = delete;
    AwakeRef& operator=(const AwakeRef&) = delete;
    AwakeRef(AwakeRef&& other) noexcept : pool_(other.pool_) {
      other.pool_ = nullptr;
    }
    AwakeRef& operator=(AwakeRef&& other) noexcept {
      reset();
      pool_ = other.pool_;
      other.pool_ = nullptr;
      return *this;
    }
    ~AwakeRef() {
      reset();
    }
    void reset() {
      if (pool_) {
        pool_->keepAwakeCount_.fetch_sub(1, std::memory_order_release);
        pool_ = nullptr;
      }
    }

   private:
    ThreadPool* pool_ = nullptr;
  };

  /**
   * Acquire an AwakeRef that prevents worker threads from sleeping (skip the
   * sleep transition at the end of their spin window) while the returned
   * handle is alive. See AwakeRef for details.
   */
  AwakeRef keepAwake() {
    return AwakeRef(this);
  }

 private:
  class PerThreadData {
   public:
    void setThread(std::thread&& t);

    bool running();

    void stop();

    ~PerThreadData();

    alignas(kCacheLineSize) std::thread thread_;
    std::atomic<bool> running_{true};
  };

  DISPENSO_DLL_ACCESS uint32_t waitOnThread(int32_t threadIdx, uint32_t priorEpoch);

  void setSignalingWake(bool enable, uint32_t sleepDurationUs) {
    std::lock_guard<std::mutex> lk(threadsMutex_);
    ssize_t currentPoolSize = numThreads();
    resizeLocked(0);
    enableEpochWaiter_.store(enable, std::memory_order_release);
    sleepLengthUs_.store(sleepDurationUs, std::memory_order_release);
    resizeLocked(currentPoolSize);
  }

  DISPENSO_DLL_ACCESS void resizeLocked(ssize_t n);

  void executeNext(OnceFunction work);

  DISPENSO_DLL_ACCESS void threadLoopWake(PerThreadData& threadData, int32_t ringIndex);
  DISPENSO_DLL_ACCESS void threadLoopPoll(PerThreadData& threadData, int32_t ringIndex);

  void markWorkDone(bool& isWorking, double& spinTimeout, bool& wokeFromSleep);
  void markIdle(bool& isWorking);

  bool tryExecuteNext();
  bool tryExecuteNextFromProducerToken(moodycamel::ProducerToken& token);
  bool tryExecuteNextFromRings(size_t& startRing);

  // Core scheduling: central queue + bulk-like wake (default, throughput-oriented).
  DISPENSO_INLINE void scheduleImpl(OnceFunction task, moodycamel::ProducerToken* token);

  // Placed scheduling: proactive wake → steal ring → central queue.
  // Higher per-call cost but better latency for individual tasks (futures, pipelines).
  DISPENSO_INLINE void scheduleImplPlaced(OnceFunction task, moodycamel::ProducerToken* token);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f);

  template <typename F>
  void schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag);

  template <typename F>
  void schedulePlaced(moodycamel::ProducerToken& token, F&& f);

  template <typename F>
  void schedulePlaced(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag);

  // Placed scheduling: public-like API but private — only for internal dispenso callers.
  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedulePlaced(F&& f);

  template <typename F>
  DISPENSO_REQUIRES(OnceCallableFunc<F>)
  void schedulePlaced(F&& f, ForceQueuingTag);

  // Bulk placed scheduling: chunked submit through the steal-ring path. Internal only —
  // exposed via ConcurrentTaskSet's scheduleBulk under TaskCost::kHeavy routing.
  template <typename Generator>
  void scheduleBulkPlaced(size_t count, Generator&& gen);

  // Core bulk enqueue: unconditionally stage, enqueue, and wake for a chunk of tasks.
  // Caller is responsible for load factor checks. Count should be small (e.g. <= 2*numThreads).
  // When a producer token is provided, uses token-based enqueue for better throughput.
  template <typename Generator>
  void
  scheduleBulkEnqueue(size_t count, Generator&& gen, moodycamel::ProducerToken* token = nullptr);

  // Wake enough threads to handle pending work. Uses PoolWakeState's budget-
  // limited cascade for efficient parallel waking.
  // Missed wakes are benign: the EpochWaiter's sleep timeout provides a safety
  // net, so a missed wake only delays wakeup by up to that duration.
  void conditionallyWake() {
    auto* ws = detail::consumeLoad(wakeState_);
    if (enableEpochWaiter_.load(std::memory_order_acquire) && ws) {
      int32_t sleeping = ws->totalSleeping();
      if (sleeping > 0) {
        ssize_t pending = workRemaining_.load(std::memory_order_relaxed);
        ssize_t numT = numThreads_.load(std::memory_order_relaxed);
        ssize_t awake = numT - static_cast<ssize_t>(sleeping);
        if (pending > awake) {
          ws->claimAndWakeOne();
        }
      }
    }
  }

 public:
  // If we are not yet C++17, we provide aligned new/delete to avoid false sharing.
#if __cplusplus < 201703L
  static void* operator new(size_t sz) {
    return detail::alignedMalloc(sz);
  }
  static void operator delete(void* ptr) {
    return detail::alignedFree(ptr);
  }
#endif // __cplusplus

 private:
  // Per-thread ring buffer type for fork-join scheduling.
  // 16 slots matches kAuto's oversubscription factor and fits in one cache line group.
  using Ring = MpmcRingBuffer<OnceFunction, 16>;

  // Steal ring configuration.
  // Slots per thread: base capacity before sharing multiplier.
  static constexpr size_t kStealSlotsPerThread = 4;
  // Sharing factor: threads per steal ring (aligned with wake group size).
#if defined(DISPENSO_TUNE_STEAL_RING_SHARING)
  static constexpr size_t kStealRingSharing = DISPENSO_TUNE_STEAL_RING_SHARING;
#else
  // Matches the wake group-size default (8). See docs/design/wake_tuning.md.
  static constexpr size_t kStealRingSharing = 8;
#endif
  static constexpr size_t kStealRingCapacity = kStealSlotsPerThread * kStealRingSharing;
  using StealRing = MpmcRingBuffer<OnceFunction, kStealRingCapacity>;

  // Cross-ring steal gate: workers only probe other rings' has-work bitmask
  // after `failCount` consecutive empty pops on their own ring. Preserves
  // placed-scheduling locality during steady-state operation; the threshold
  // (~kSpinCheckInterval / 2) means we steal cross-ring only after sustained
  // local idle, by which point our own ring's locality is exhausted anyway.
#if defined(DISPENSO_TUNE_CROSS_RING_FAIL_THRESHOLD)
  static constexpr int kCrossRingFailThreshold = DISPENSO_TUNE_CROSS_RING_FAIL_THRESHOLD;
#else
  static constexpr int kCrossRingFailThreshold = 32;
#endif

  // Enqueue a task to the central concurrent queue with optional producer token.
  // Handles TSAN annotations. Used by scheduleBulkToRings for overflow tasks.
  // moodycamel::ConcurrentQueue::enqueue returns false only on allocation
  // failure; we propagate that as std::bad_alloc so callers don't silently
  // drop work (which would deadlock TaskSet::wait via inflated outstanding
  // counts). Out-of-memory recovery from individual small allocs is not
  // tractable in general; throwing lets the application unwind or terminate.
  DISPENSO_INLINE void enqueueToCentralQueue(OnceFunction task, moodycamel::ProducerToken* token) {
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
    bool enqueued;
    if (token) {
      enqueued = work_.enqueue(*token, std::move(task));
    } else {
      enqueued = work_.enqueue(std::move(task));
    }
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
    if (DISPENSO_EXPECT(!enqueued, false)) {
#if defined(__cpp_exceptions)
      throw std::bad_alloc();
#else
      std::abort();
#endif
    }
    // Mark queue as possibly-non-empty so spinning workers will try_dequeue.
    centralQueueNonEmpty_.store(true, std::memory_order_relaxed);
  }

  // Push task i to ring i (linear layout) for fork-join scheduling.
  // Tasks that don't fit in their ring go to central queue via fallbackToken.
  // Handles workRemaining_ accounting and waking (one batch wake at the end).
  // outstandingTaskCount_ must be managed by the caller.
  template <typename Generator>
  void scheduleBulkToRings(size_t count, Generator&& gen, moodycamel::ProducerToken* fallbackToken);

  // Shared work-finding logic for both loop variants. Checks own ring,
  // central queue, and steal ring as third tier.
  // Returns true if work was found and executed.
  // preferRing: sticky hint — true = try ring first, false = try central queue first.
  // failCount: consecutive failures finding work — used to gate cross-ring stealing
  //   so we preserve placed-scheduling locality during steady-state operation.
  DISPENSO_INLINE bool tryFindAndExecuteWork(
      Ring& myRing,
      StealRing& myStealRing,
      size_t myStealIdx,
      moodycamel::ConsumerToken& ctoken,
      bool& preferRing,
      int failCount,
      bool checkQueue = true);

  // Number of tasks each thread accumulates before flushing workRemaining_.
  // This batching reduces atomic contention in threadLoop, but inflates
  // workRemaining_ by up to kWorkBatchSize * numThreads, so poolLoadFactor_
  // must be reduced accordingly for accurate load-shedding in schedule().
  static constexpr int kWorkBatchSize = 8;

  // Minimum spinning threads before we skip waking a sleeper. If fewer than
  // this many threads are spinning, we wake a sleeper to ensure coverage.
  // Higher = more aggressive waking; lower = trust spinners more.
  static constexpr int32_t kSpinnerWakeThreshold = 2;

  mutable std::mutex threadsMutex_;
  std::deque<PerThreadData> threads_;
  size_t poolLoadMultiplier_;

  // These atomics are read frequently in the hot schedule() path, so they need
  // cache-line alignment to avoid false sharing with the mutex/deque above.
  alignas(kCacheLineSize) std::atomic<ssize_t> poolLoadFactor_;
  std::atomic<ssize_t> numThreads_;

  moodycamel::ConcurrentQueue<OnceFunction> work_;

  // Approximate flag indicating the central queue may have work. Set (store
  // true) after enqueue, cleared (store false) when try_dequeue finds the queue
  // empty. Used to gate try_dequeue in tryFindAndExecuteWork: a relaxed load
  // replaces the expensive try_dequeue CAS when the queue is known empty,
  // eliminating CAS contention from idle spinning threads. No atomic RMW —
  // only plain stores and loads — so no contention on the flag itself.
  // Brief false-negatives (flag cleared while an enqueue is in flight) are
  // bounded to one spin iteration and self-correcting.
  alignas(kCacheLineSize) std::atomic<bool> centralQueueNonEmpty_{false};

  // Refcount of outstanding AwakeRef handles. When > 0, worker threads skip
  // the sleep transition at the end of their spin window and continue
  // spinning. Bumped by AwakeRef ctor, decremented by dtor. Workers read with
  // relaxed ordering — a brief stale-true read just costs one extra spin
  // iteration; a brief stale-false read is bounded by the spin window and
  // self-corrects on the next iteration.
  alignas(kCacheLineSize) std::atomic<int32_t> keepAwakeCount_{0};

  alignas(kCacheLineSize) std::atomic<ssize_t> workRemaining_{0};

  alignas(kCacheLineSize) std::atomic<bool> enableEpochWaiter_{kDefaultWakeupEnable};
  std::atomic<uint32_t> sleepLengthUs_{kDefaultSleepLenUs};

  // Per-thread wake infrastructure: EpochWaiters, sleep masks, budget cascade.
  // Atomic pointer — schedule paths load with relaxed ordering.
  //
  // Retired PoolWakeState objects are kept alive for the lifetime of the pool
  // (grow-only graveyard); they are intentionally NOT freed at resize. schedule()
  // reads wakeState_ lock-free (no threadsMutex_) and dereferences it (e.g.
  // ws->totalSleeping()), so freeing a retired generation while a concurrent
  // schedule() may still hold that pointer is a use-after-free. resize() joins all
  // *pool* threads before swapping wakeState_, but external (non-pool) schedule()
  // callers race the swap, so a retired object must outlive any such in-flight
  // reader. Without a safe-reclamation protocol, never freeing during operation is
  // the only correct option (an earlier bounded variant that freed old
  // generations was reverted after TSAN caught exactly this free-vs-read race).
  //
  // Cost: one PoolWakeState (~O(numThreads)) is retained per resize() until the
  // pool is destroyed. resize() is expected to be rare, so this is bounded in
  // practice; only a process performing hundreds of thousands of resizes would
  // accumulate meaningful memory. See docs/design/roadmap.md ("Bounded
  // PoolWakeState reclamation") for the planned asymmetric-fence / hazard-pointer
  // scheme to bound this without taxing the schedule() hot path.
  std::atomic<PoolWakeState*> wakeState_{nullptr};
  std::vector<decltype(detail::makeAligned<PoolWakeState>(0))> wakeStateGraveyard_;

  // Per-thread rings for fork-join scheduling. ConcurrentObjectArena provides
  // stable pointers (grow-only, never freed), eliminating the need for a resize
  // lock on the schedule path. Threads check own ring first in the steal order.
  ConcurrentObjectArena<Ring> rings_;
  std::atomic<size_t> numRings_{0};

  // Steal rings for non-locality work distribution.
  // Populated by schedule() (both proactive wake and no-sleeper paths).
  // Consumed in tryFindAndExecuteWork (third tier) and outer thread loop.
  //
  // stealRingSharing_: threads per steal ring (default kStealRingSharing).
  //   Ring capacity = kStealSlotsPerThread * kStealRingSharing.
  ConcurrentObjectArena<StealRing> stealRings_;
  std::atomic<size_t> numStealRings_{0};
  size_t stealRingSharing_{kStealRingSharing};

  // Sparse hint for which steal rings have work. Bit i set means
  // stealRings_[i] may have work. Set on push (idempotent fetch_or); lazily
  // cleared by consumers that find a ring empty after popping or that
  // observe an empty ring at scan time. False positives are benign (just a
  // wasted try_pop); false negatives are not possible because every
  // successful push sets the bit before the work is observable.
  static constexpr size_t kMaxStealRings = 64;
  alignas(kCacheLineSize) std::atomic<uint64_t> stealRingsWithWork_{0};

  // Threads not currently in their inner work loop (spinning or sleeping).
  // Incremented when a thread exhausts work (exits inner loop with nothing found).
  // Decremented when a thread finds work (enters inner loop) or at thread exit.
  // Updated at burst boundaries (not per-task), so contention is low.
  // Used by schedule paths to skip wake calls when spinners exist.
  alignas(kCacheLineSize) std::atomic<int32_t> numNotWorking_{0};

  // Platform-tuned spin limit (0 = adaptive time-based spin, >0 = fixed count).
  // Linux/macOS: aggressive sleep (futex wake is cheap, ~3-5μs).
  // Windows: adaptive spin (WakeByAddressSingle is expensive).
  // Override via DISPENSO_TUNE_FIXED_SPIN_ITERS.
  int32_t spinLimit_{0};

#if defined DISPENSO_DEBUG
  alignas(kCacheLineSize) std::atomic<ssize_t> outstandingTaskSets_{0};
#endif // DISPENSO_DEBUG

  friend class ConcurrentTaskSet;
  friend class TaskSet;
  friend class TaskSetBase;

  template <typename Result>
  friend class detail::FutureBase;
  template <typename Result>
  friend class detail::FutureImplBase;
};

/**
 * Get access to the global thread pool.
 *
 * @return the global thread pool
 **/
DISPENSO_DLL_ACCESS ThreadPool& globalThreadPool();

/**
 * Change the number of threads backing the global thread pool.
 *
 * @param numThreads The number of threads to back the global thread pool.
 **/
DISPENSO_DLL_ACCESS void resizeGlobalThreadPool(size_t numThreads);

// ----------------------------- Implementation details -------------------------------------

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedule(F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedule(F&& f, ForceQueuingTag) {
  if (auto* token =
          static_cast<moodycamel::ProducerToken*>(detail::PerPoolPerThreadInfo::producer(this))) {
    schedule(*token, std::forward<F>(f), ForceQueuingTag());
    return;
  }

  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  scheduleImpl({std::forward<F>(f)}, nullptr);
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedule(token, std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
inline void ThreadPool::schedule(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag) {
  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  scheduleImpl({std::forward<F>(f)}, &token);
}

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedulePlaced(F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedulePlaced(std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
DISPENSO_REQUIRES(OnceCallableFunc<F>)
inline void ThreadPool::schedulePlaced(F&& f, ForceQueuingTag) {
  if (auto* token =
          static_cast<moodycamel::ProducerToken*>(detail::PerPoolPerThreadInfo::producer(this))) {
    schedulePlaced(*token, std::forward<F>(f), ForceQueuingTag());
    return;
  }

  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  scheduleImplPlaced({std::forward<F>(f)}, nullptr);
}

template <typename F>
inline void ThreadPool::schedulePlaced(moodycamel::ProducerToken& token, F&& f) {
  ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
  ssize_t quickLoadFactor = numThreads_.load(std::memory_order_relaxed);
  quickLoadFactor += quickLoadFactor / 2;
  if ((detail::PerPoolPerThreadInfo::isPoolRecursive(this) && curWork > quickLoadFactor) ||
      (curWork > poolLoadFactor_.load(std::memory_order_relaxed))) {
    f();
  } else {
    schedulePlaced(token, std::forward<F>(f), ForceQueuingTag());
  }
}

template <typename F>
inline void ThreadPool::schedulePlaced(moodycamel::ProducerToken& token, F&& f, ForceQueuingTag) {
  if (!numThreads_.load(std::memory_order_relaxed)) {
    f();
    return;
  }
  workRemaining_.fetch_add(1, std::memory_order_release);
  scheduleImplPlaced({std::forward<F>(f)}, &token);
}

DISPENSO_INLINE void ThreadPool::scheduleImpl(OnceFunction task, moodycamel::ProducerToken* token) {
  enqueueToCentralQueue(std::move(task), token);

  // Wake when pending work exceeds awake threads. Each schedule()
  // call wakes at most one sleeper — matching baseline's wake-per-task
  // cadence but using per-thread futexes. The one-at-a-time approach
  // avoids thundering herd on the central queue while still waking
  // threads proportionally to submitted work over a burst.
  auto* ws = detail::consumeLoad(wakeState_);
  if (enableEpochWaiter_.load(std::memory_order_acquire) && ws) {
    int32_t sleeping = ws->totalSleeping();
    if (sleeping > 0) {
      ssize_t pending = workRemaining_.load(std::memory_order_relaxed);
      ssize_t numT = numThreads_.load(std::memory_order_relaxed);
      ssize_t awake = numT - static_cast<ssize_t>(sleeping);
      if (pending > awake) {
        ws->claimAndWakeOne();
      }
    }
  }
}

DISPENSO_INLINE void ThreadPool::scheduleImplPlaced(
    OnceFunction task,
    moodycamel::ProducerToken* token) {
  // Proactive wake: claim a sleeping thread and push to its steal ring.
  auto* ws = detail::consumeLoad(wakeState_);
  if (enableEpochWaiter_.load(std::memory_order_acquire) && ws) {
    int32_t sleeping = ws->totalSleeping();
    if (sleeping > 0 &&
        numNotWorking_.load(std::memory_order_relaxed) - sleeping < kSpinnerWakeThreshold) {
      int32_t wokeThread = ws->claimAndWakeOne();
      if (wokeThread >= 0) {
        size_t stealIdx = static_cast<size_t>(wokeThread) / stealRingSharing_;
        if (stealIdx < numStealRings_.load(std::memory_order_relaxed) &&
            stealRings_[stealIdx].try_push(std::move(task))) {
          if (stealIdx < kMaxStealRings) {
            stealRingsWithWork_.fetch_or(uint64_t{1} << stealIdx, std::memory_order_release);
          }
          return;
        }
      }
    }
  }

  // Central queue fallback.
  enqueueToCentralQueue(std::move(task), token);

  conditionallyWake();
}

inline bool ThreadPool::tryExecuteNext() {
  OnceFunction next;
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  bool dequeued = work_.try_dequeue(next);
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  if (dequeued) {
    executeNext(std::move(next));
    return true;
  }
  return false;
}

inline bool ThreadPool::tryExecuteNextFromProducerToken(moodycamel::ProducerToken& token) {
  OnceFunction next;
  if (work_.try_dequeue_from_producer(token, next)) {
    executeNext(std::move(next));
    return true;
  }
  return false;
}

inline bool ThreadPool::tryExecuteNextFromRings(size_t& startRing) {
  OnceFunction task;
  // Acquire pairs with resizeLocked's numRings_.store(release), which is published
  // only after the new rings are fully constructed in the arena. A relaxed load
  // could observe the grown count without the rings' construction being visible,
  // letting us index a not-yet-constructed ring (UB; SIGILL on weak-memory targets
  // like arm64).
  size_t n = numRings_.load(std::memory_order_acquire);
  for (size_t i = 0; i < n; ++i) {
    size_t idx = (startRing + i) % n;
    if (rings_[idx].try_pop(task)) {
      startRing = idx;
      executeNext(std::move(task));
      return true;
    }
  }
  startRing = 0;
  return false;
}

inline void ThreadPool::executeNext(OnceFunction next) {
  next();
  workRemaining_.fetch_add(-1, std::memory_order_relaxed);
}

DISPENSO_INLINE bool ThreadPool::tryFindAndExecuteWork(
    Ring& myRing,
    StealRing& myStealRing,
    size_t myStealIdx,
    moodycamel::ConsumerToken& ctoken,
    bool& preferRing,
    int failCount,
    bool checkQueue) {
  OnceFunction task;

  if (preferRing) {
    bool fromRing = myRing.try_pop(task);
    if (fromRing) {
      task();
      return true;
    }
    if (checkQueue && centralQueueNonEmpty_.load(std::memory_order_relaxed)) {
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
      bool got = work_.try_dequeue(ctoken, task);
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
      if (got) {
        preferRing = false;
        task();
        return true;
      }
      // Empty on observation; clear flag (relaxed, plain store).
      centralQueueNonEmpty_.store(false, std::memory_order_relaxed);
    }
    if (!myStealRing.empty() && myStealRing.try_pop(task)) {
      task();
      return true;
    }
    if (failCount >= kCrossRingFailThreshold) {
      uint64_t mask = stealRingsWithWork_.load(std::memory_order_acquire);
      if (mask != 0) {
        if (myStealIdx < kMaxStealRings) {
          mask &= ~(uint64_t{1} << myStealIdx);
        }
        if (mask != 0) {
          int target = detail::countTrailingZeros(mask);
          if (stealRings_[static_cast<size_t>(target)].try_pop(task)) {
            task();
            return true;
          }
          stealRingsWithWork_.fetch_and(~(uint64_t{1} << target), std::memory_order_relaxed);
        }
      }
    }
  } else {
    if (checkQueue && centralQueueNonEmpty_.load(std::memory_order_relaxed)) {
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
      bool got = work_.try_dequeue(ctoken, task);
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
      if (got) {
        task();
        return true;
      }
      centralQueueNonEmpty_.store(false, std::memory_order_relaxed);
    }
    bool fromRing = myRing.try_pop(task);
    if (fromRing) {
      preferRing = true;
      task();
      return true;
    }
  }

  return false;
}

template <typename Generator>
void ThreadPool::scheduleBulkToRings(
    size_t count,
    Generator&& gen,
    moodycamel::ProducerToken* fallbackToken) {
  if (count == 0) {
    return;
  }
  assert(count <= numRings_.load(std::memory_order_relaxed));

  // Increment before enqueuing, consistent with schedule() and scheduleBulkEnqueue().
  // This ensures threads that wake and find work in a ring see a positive workRemaining_,
  // preventing premature re-sleep or incorrect load-balancing decisions.
  workRemaining_.fetch_add(static_cast<ssize_t>(count), std::memory_order_release);

  // No lock needed: ConcurrentObjectArena rings have stable pointers.

  // Linear layout: task i goes to ring i (1 task per ring, kStatic pattern).
  // Tasks that don't fit in their ring overflow to central queue.
  // Acquire: see tryExecuteNextFromRings. Pairs with the release store in
  // resizeLocked so we observe the freshly-constructed rings, not merely the
  // updated count.
  size_t ringCount = numRings_.load(std::memory_order_acquire);
  size_t tasksPerRing = (count + ringCount - 1) / ringCount;

  size_t taskIdx = 0;
  if (tasksPerRing <= 1) {
    // Fast path for kStatic (1 task per ring): no staging array needed.
    for (size_t ring = 0; ring < count && ring < ringCount; ++ring) {
      OnceFunction task = gen(ring);
      if (!rings_[ring].try_push(std::move(task))) {
        enqueueToCentralQueue(std::move(task), fallbackToken);
      }
    }
  } else {
    // Batched path for kAuto (multiple tasks per ring).
    constexpr size_t kMaxStage = Ring::capacity();
    for (size_t ring = 0; ring < ringCount && taskIdx < count; ++ring) {
      size_t blockEnd = std::min(taskIdx + tasksPerRing, count);
      size_t blockSize = blockEnd - taskIdx;

      // Stage tasks for this ring
      size_t toStage = std::min(blockSize, kMaxStage);
      OnceFunction staged[kMaxStage];
      for (size_t j = 0; j < toStage; ++j) {
        staged[j] = gen(taskIdx + j);
      }

      size_t pushed = rings_[ring].try_push_batch(staged, toStage);

      // Overflow: staged tasks that didn't fit
      for (size_t j = pushed; j < toStage; ++j) {
        enqueueToCentralQueue(std::move(staged[j]), fallbackToken);
      }

      // Tasks beyond ring capacity that weren't staged go straight to central queue
      for (size_t j = taskIdx + toStage; j < blockEnd; ++j) {
        enqueueToCentralQueue(gen(j), fallbackToken);
      }
      taskIdx += blockSize;
    }
  }

  // Wake only the threads whose rings received work (threads 0..count-1).
  // This avoids waking threads with empty rings, which would spin wastefully.
  auto* ws = detail::consumeLoad(wakeState_);
  if (enableEpochWaiter_.load(std::memory_order_acquire) && ws) {
    ws->wakeRange(static_cast<int32_t>(count));
  }
}

namespace detail {
// Generating iterator for scheduleBulkEnqueue. Produces OnceFunction objects
// on-the-fly during enqueue_bulk, avoiding the need for a staging buffer.
// moodycamel's enqueue_bulk uses single-pass input iterator semantics.
template <typename Generator>
struct BulkGenIter {
  using difference_type = std::ptrdiff_t;
  using value_type = OnceFunction;
  using pointer = OnceFunction*;
  using reference = OnceFunction&;
  using iterator_category = std::input_iterator_tag;

  Generator* gen;
  size_t index;
  OnceFunction operator*() {
    return (*gen)(index);
  }
  BulkGenIter& operator++() {
    ++index;
    return *this;
  }
  BulkGenIter operator++(int) {
    BulkGenIter tmp = *this;
    ++index;
    return tmp;
  }
};
} // namespace detail

template <typename Generator>
void ThreadPool::scheduleBulkEnqueue(
    size_t count,
    Generator&& gen,
    moodycamel::ProducerToken* token) {
  detail::BulkGenIter<typename std::remove_reference<Generator>::type> it{&gen, 0};

  // Single atomic update + bulk enqueue
  workRemaining_.fetch_add(static_cast<ssize_t>(count), std::memory_order_release);

  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
  bool enqueued;
  if (token) {
    enqueued = work_.enqueue_bulk(*token, it, count);
  } else {
    enqueued = work_.enqueue_bulk(it, count);
  }
  DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  if (DISPENSO_EXPECT(!enqueued, false)) {
    workRemaining_.fetch_sub(static_cast<ssize_t>(count), std::memory_order_relaxed);
#if defined(__cpp_exceptions)
    throw std::bad_alloc();
#else
    std::abort();
#endif
  }
  // Mark queue as possibly-non-empty so spinning workers will try_dequeue.
  centralQueueNonEmpty_.store(true, std::memory_order_relaxed);

  // Wake appropriate threads. Cap by actual sleeping count to avoid over-waking.
  // Spinning threads (numNotWorking - totalSleeping) will find enqueued work
  // naturally, so only wake enough sleepers to cover the deficit beyond the
  // spinner threshold.
  auto* ws = detail::consumeLoad(wakeState_);
  if (enableEpochWaiter_.load(std::memory_order_acquire) && ws) {
    int32_t sleeping = ws->totalSleeping();
    if (sleeping > 0) {
      int32_t notWorking = numNotWorking_.load(std::memory_order_relaxed);
      int32_t spinning = std::max(int32_t{0}, notWorking - sleeping);
      // Only count spinners beyond the threshold as "covering" tasks
      int32_t effectiveSpinners = std::max(int32_t{0}, spinning - kSpinnerWakeThreshold + 1);
      int32_t toWake = std::max(int32_t{0}, static_cast<int32_t>(count) - effectiveSpinners);
      toWake = std::min(toWake, sleeping);
      if (toWake <= ws->branchFactor()) {
        // Small N: direct claim+wake, no cascade overhead
        for (int32_t i = 0; i < toWake; ++i) {
          if (ws->claimAndWakeOne() < 0) {
            break;
          }
        }
      } else {
        // Large N: budget cascade for parallel fan-out
        ws->wakeN(toWake);
      }
    }
  }
}

template <typename Generator>
void ThreadPool::scheduleBulk(size_t count, Generator&& gen) {
  if (count == 0) {
    return;
  }

  ssize_t numPool = numThreads_.load(std::memory_order_relaxed);
  if (!numPool) {
    // No threads in pool - execute all inline
    for (size_t i = 0; i < count; ++i) {
      gen(i)();
    }
    return;
  }

  // Process in chunks, interleaving enqueue and inline execution based on load.
  size_t chunkSize = static_cast<size_t>(numPool) + static_cast<size_t>(numPool) / 2;
  size_t i = 0;
  while (i < count) {
    ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
    ssize_t loadFactor = poolLoadFactor_.load(std::memory_order_relaxed);
    if (curWork > loadFactor) {
      // Over load factor - execute one task inline, then re-check
      gen(i)();
      ++i;
    } else {
      // Under load factor - enqueue a chunk
      ssize_t room = loadFactor - curWork;
      size_t toEnqueue = std::min({count - i, chunkSize, static_cast<size_t>(room)});
      if (toEnqueue == 0) {
        toEnqueue = 1;
      }
      size_t base = i;
      scheduleBulkEnqueue(toEnqueue, [&gen, base](size_t j) { return gen(base + j); });
      i += toEnqueue;
    }
  }
}

template <typename Generator>
void ThreadPool::scheduleBulkPlaced(size_t count, Generator&& gen) {
  if (count == 0) {
    return;
  }

  ssize_t numPool = numThreads_.load(std::memory_order_relaxed);
  if (!numPool) {
    for (size_t i = 0; i < count; ++i) {
      gen(i)();
    }
    return;
  }

  // Process in chunks, interleaving placed-enqueue and inline execution based on load.
  size_t chunkSize = static_cast<size_t>(numPool) + static_cast<size_t>(numPool) / 2;
  size_t i = 0;
  while (i < count) {
    ssize_t curWork = workRemaining_.load(std::memory_order_relaxed);
    ssize_t loadFactor = poolLoadFactor_.load(std::memory_order_relaxed);
    if (curWork > loadFactor) {
      gen(i)();
      ++i;
    } else {
      ssize_t room = loadFactor - curWork;
      size_t toEnqueue = std::min({count - i, chunkSize, static_cast<size_t>(room)});
      if (toEnqueue == 0) {
        toEnqueue = 1;
      }
      // Batch the workRemaining bump for the whole chunk; placed path will
      // wake per-task as needed via scheduleImplPlaced's claimAndWakeOne.
      workRemaining_.fetch_add(static_cast<ssize_t>(toEnqueue), std::memory_order_release);
      for (size_t j = 0; j < toEnqueue; ++j) {
        scheduleImplPlaced({gen(i + j)}, nullptr);
      }
      i += toEnqueue;
    }
  }
}

} // namespace dispenso
