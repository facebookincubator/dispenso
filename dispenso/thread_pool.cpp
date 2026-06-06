/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/quanta.h>
#include <dispenso/thread_pool.h>
#include <dispenso/timing.h>

#if defined DISPENSO_DEBUG
#include <iostream>
#endif // DISPENSO_DEBUG

namespace dispenso {

namespace {
size_t getAdjustedThreadCount(size_t requested) {
  static const size_t maxThreads = []() {
    size_t maxT = std::numeric_limits<size_t>::max();

#if defined(_WIN32) && !defined(__MINGW32__)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

    char* envThreads = std::getenv("DISPENSO_MAX_THREADS_PER_POOL");

#if defined(_WIN32) && !defined(__MINGW32__)
#pragma warning(pop)
#endif

    if (envThreads) {
      char* end = nullptr;
      maxT = std::strtoul(envThreads, &end, 10);
#if defined DISPENSO_DEBUG
      std::cout << "DISPENSO_MAX_THREADS_PER_POOL = " << maxT << std::endl;
#endif // DISPENSO_DEBUG
    }
    return maxT;
  }();

  return std::min(requested, maxThreads);
}
} // namespace

// Compile-time override: forces all pools to use a fixed spin count.
// When not set, the platform default below is used.
#if defined(DISPENSO_TUNE_FIXED_SPIN_ITERS)
static constexpr int32_t kDefaultSpinLimit = DISPENSO_TUNE_FIXED_SPIN_ITERS;
#elif defined(_WIN32)
// Windows: spin_fixed_200 was ~6% faster (geomean across the tuning set)
// than the adaptive baseline on a 48-thread Xeon Platinum 8259CL Windows
// sweep. WakeByAddressSingle is expensive enough that even a small fixed
// spin is preferable to the adaptive time-based backoff, which tended to
// sleep too aggressively between parallel_for iterations.
static constexpr int32_t kDefaultSpinLimit = 200;
#else
// Linux/macOS: futex/__ulock wake is cheap (~3-5μs per group). Sleep
// aggressively to free SMT siblings for workers. Multiple pools may
// coexist and none has visibility into total system load, so
// conservative sleeping is the safe default.
static constexpr int32_t kDefaultSpinLimit = 400;
#endif

void ThreadPool::PerThreadData::setThread(std::thread&& t) {
  thread_ = std::move(t);
}

void ThreadPool::PerThreadData::stop() {
  running_.store(false, std::memory_order_release);
}

// Per-thread wait using PoolWakeState's per-thread EpochWaiter.
uint32_t ThreadPool::waitOnThread(int32_t threadIdx, uint32_t currentEpoch) {
  auto* ws = detail::consumeLoad(wakeState_);
  assert(ws && "wakeState_ null — threads must not outlive their PoolWakeState");
  if (sleepLengthUs_ > 0) {
    return ws->waiterFor(threadIdx).waitFor(
        currentEpoch, sleepLengthUs_.load(std::memory_order_acquire));
  } else {
    return ws->waiterFor(threadIdx).current();
  }
}

inline bool ThreadPool::PerThreadData::running() {
  return running_.load(std::memory_order_acquire);
}

ThreadPool::ThreadPool(size_t n, size_t poolLoadMultiplier)
    : poolLoadMultiplier_(poolLoadMultiplier),
      poolLoadFactor_(static_cast<ssize_t>(getAdjustedThreadCount(n) * poolLoadMultiplier)),
      numThreads_(static_cast<ssize_t>(getAdjustedThreadCount(n))),
      rings_(std::max(getAdjustedThreadCount(n), size_t{1}), getAdjustedThreadCount(n)),
      stealRings_(
          std::max(
              (getAdjustedThreadCount(n) + kStealRingSharing - 1) / kStealRingSharing,
              size_t{1}),
          (getAdjustedThreadCount(n) + kStealRingSharing - 1) / kStealRingSharing) {
  detail::registerFineSchedulerQuanta();
#if defined DISPENSO_DEBUG
  assert(poolLoadMultiplier > 0);
#endif // DISPENSO_DEBUG

  size_t adjustedN = static_cast<size_t>(numThreads_);
  // Set up per-thread ring counts and wake state.
  if (adjustedN > 0) {
    numRings_.store(adjustedN, std::memory_order_release);
    numStealRings_.store(
        (adjustedN + stealRingSharing_ - 1) / stealRingSharing_, std::memory_order_release);
    auto ws = detail::makeAligned<PoolWakeState>(static_cast<int32_t>(adjustedN));
    auto* rawWs = ws.get();
    wakeStateGraveyard_.push_back(std::move(ws));
    DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(&wakeState_);
    wakeState_.store(rawWs, std::memory_order_release);
  }

  // All threads start as "not working" — they haven't entered their work loops yet.
  numNotWorking_.store(static_cast<int32_t>(adjustedN), std::memory_order_relaxed);

  for (size_t i = 0; i < adjustedN; ++i) {
    threads_.emplace_back();
    int32_t ringIdx = static_cast<int32_t>(i);
    if (enableEpochWaiter_) {
      threads_.back().setThread(std::thread([this, &back = threads_.back(), ringIdx]() {
        threadLoopWake(back, ringIdx);
      }));
    } else {
      threads_.back().setThread(std::thread([this, &back = threads_.back(), ringIdx]() {
        threadLoopPoll(back, ringIdx);
      }));
    }
  }
}

ThreadPool::PerThreadData::~PerThreadData() {}

// Fixed-spin sleep strategy: workers iterate the work-finding loop up to
// kSpinLimit times before parking. kSpinCheckInterval bounds the "lean
// spin" warmup phase where threads only check their own ring (no central
// queue).
// Override via -DDISPENSO_TUNE_SPIN_CHECK_INTERVAL / -DDISPENSO_TUNE_FIXED_SPIN_ITERS.
#if defined(DISPENSO_TUNE_SPIN_CHECK_INTERVAL)
static constexpr int kSpinCheckInterval = DISPENSO_TUNE_SPIN_CHECK_INTERVAL;
#elif defined(__APPLE__)
// macOS: __ulock wake is faster than Linux futex, so threads can sleep/wake
// more quickly. Check more frequently to keep spin duration accurate.
static constexpr int kSpinCheckInterval = 32;
#elif defined(_WIN32)
// Windows: WaitOnAddress is expensive per-call. Spin longer between checks
// to reduce sleep/wake transitions.
static constexpr int kSpinCheckInterval = 256;
#else
static constexpr int kSpinCheckInterval = 64;
#endif

// How often idle threads check the central queue during the spin phase.
// Between checks, threads only inspect their ring (no CAS contention).
// Staggered by thread index so ~1/kQueueCheckInterval threads check
// per iteration, reducing try_dequeue contention from idle threads.
// Must be a power of 2 for fast bitmask check.
#if defined(DISPENSO_TUNE_QUEUE_CHECK_INTERVAL)
static constexpr int kQueueCheckInterval = DISPENSO_TUNE_QUEUE_CHECK_INTERVAL;
#else
static constexpr int kQueueCheckInterval = 8;
#endif

static_assert(
    (kSpinCheckInterval & (kSpinCheckInterval - 1)) == 0,
    "kSpinCheckInterval must be a power of 2");
static_assert(
    (kQueueCheckInterval & (kQueueCheckInterval - 1)) == 0,
    "kQueueCheckInterval must be a power of 2");

void ThreadPool::markWorkDone(bool& isWorking) {
  if (!isWorking) {
    numNotWorking_.fetch_sub(1, std::memory_order_relaxed);
    isWorking = true;
  }
}

void ThreadPool::markIdle(bool& isWorking) {
  if (isWorking) {
    numNotWorking_.fetch_add(1, std::memory_order_relaxed);
    isWorking = false;
  }
}

template <bool kUseWakeSleep>
void ThreadPool::threadLoopImpl(PerThreadData& data, int32_t ringIndex) {
  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  bool preferRing = false;

  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken, ringIndex);
  auto* ws = detail::consumeLoad(wakeState_);
  assert(ws && "wakeState_ null — threads must not outlive their PoolWakeState");
  uint32_t epoch = ws->waiterFor(ringIndex).current();
  size_t myRingIndex = static_cast<size_t>(ringIndex);
  Ring& myRing = rings_[myRingIndex];
  size_t myStealIdx = static_cast<size_t>(ringIndex) / stealRingSharing_;
  StealRing& myStealRing = stealRings_[myStealIdx];

  int failCount = 0;
  bool isWorking = false;

  while (data.running()) {
    int localWorkDone = 0;
    bool checkQueue = (failCount < kSpinCheckInterval) ||
        (((failCount + ringIndex) & (kQueueCheckInterval - 1)) == 0);
    while (tryFindAndExecuteWork(
        myRing, myStealRing, myStealIdx, ctoken, preferRing, failCount, checkQueue)) {
      ++localWorkDone;
      if (localWorkDone >= kWorkBatchSize) {
        workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
        localWorkDone = 0;
      }
      failCount = 0;
      checkQueue = true;
    }
    if (localWorkDone > 0) {
      markWorkDone(isWorking);
      workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
      failCount = 0;
      continue;
    }

    ++failCount;

    if (failCount < kSpinCheckInterval) {
      detail::cpuRelax();
      continue;
    }

    // Steal ring check (deferred from lean phase).
    {
      OnceFunction stealTask;
      if (myStealRing.try_pop(stealTask)) {
        markWorkDone(isWorking);
        executeNext(std::move(stealTask));
        failCount = 0;
        continue;
      }
    }

    // Intentional second increment: lean-phase failures (above) count once,
    // full-phase failures count twice. kDefaultSpinLimit is tuned assuming
    // this two-rate progression.
    ++failCount;

    detail::cpuRelax();

    if (failCount >= kDefaultSpinLimit) {
      if (keepAwakeCount_.load(std::memory_order_relaxed) > 0) {
        failCount = kDefaultSpinLimit;
        continue;
      }
      markIdle(isWorking);
      if (kUseWakeSleep) {
        ws->enterSleep(ringIndex);
        if (!data.running()) {
          ws->exitSleep(ringIndex);
          break;
        }
      }
      epoch = waitOnThread(ringIndex, epoch);
      if (kUseWakeSleep) {
        ws->exitSleep(ringIndex);
      }
      failCount = 0;
    }
  }

  markIdle(isWorking);
}

// Explicit instantiations so the linker finds them.
template void ThreadPool::threadLoopImpl<true>(PerThreadData&, int32_t);
template void ThreadPool::threadLoopImpl<false>(PerThreadData&, int32_t);

void ThreadPool::resizeLocked(ssize_t sn) {
  sn = getAdjustedThreadCount(sn);

  assert(sn >= 0);
  size_t n = static_cast<size_t>(sn);

  if (n == threads_.size()) {
    return;
  }

  // Stop ALL threads, drain work, rebuild wake state, restart.
  // Resize is a rare operation; correctness and simplicity take priority
  // over performance here. We must stop all threads before rebuilding
  // wakeState_ because sleeping threads hold references to EpochWaiters
  // inside the wake state — replacing it while threads sleep would
  // invalidate those futex addresses.
  for (auto& t : threads_) {
    t.stop();
  }
  {
    auto* ws = detail::consumeLoad(wakeState_);
    if (ws) {
      ws->wakeAll();
    }
  }

  // Drain central queue while threads are stopping
  while (tryExecuteNext()) {
  }

  for (auto& t : threads_) {
    t.thread_.join();
  }
  threads_.clear();

  // Drain all rings in the arena (including shadow entries from prior resize-up)
  for (size_t i = 0; i < rings_.size(); ++i) {
    OnceFunction task;
    while (rings_[i].try_pop(task)) {
      task();
    }
  }
  for (size_t i = 0; i < stealRings_.size(); ++i) {
    OnceFunction task;
    while (stealRings_[i].try_pop(task)) {
      task();
    }
  }

  // Rebuild infrastructure for new size — no lock needed.
  // Pool threads are stopped. External schedule() calls use atomic loads
  // and ConcurrentObjectArena (stable pointers) — safe to race.
  if (n > 0) {
    if (n > rings_.size()) {
      rings_.grow_by(n - rings_.size());
    }
    numRings_.store(n, std::memory_order_release);

    size_t newNumSteal = (n + stealRingSharing_ - 1) / stealRingSharing_;
    if (newNumSteal > stealRings_.size()) {
      stealRings_.grow_by(newNumSteal - stealRings_.size());
    }
    numStealRings_.store(newNumSteal, std::memory_order_release);

    auto newWake = detail::makeAligned<PoolWakeState>(static_cast<int32_t>(n));
    auto* rawNewWake = newWake.get();
    // Retained for the lifetime of the pool — never freed here. Freeing a retired
    // PoolWakeState would race a concurrent lock-free schedule() that still holds
    // the old wakeState_ pointer (use-after-free). See wakeStateGraveyard_ docs.
    wakeStateGraveyard_.push_back(std::move(newWake));
    DISPENSO_TSAN_ANNOTATE_HAPPENS_BEFORE(&wakeState_);
    wakeState_.store(rawNewWake, std::memory_order_release);
  } else {
    numStealRings_.store(0, std::memory_order_release);
    wakeState_.store(nullptr, std::memory_order_release);
  }

  poolLoadFactor_.store(static_cast<ssize_t>(n * poolLoadMultiplier_), std::memory_order_relaxed);
  numThreads_.store(sn, std::memory_order_relaxed);
  numNotWorking_.store(static_cast<int32_t>(n), std::memory_order_relaxed);

  // Start new threads (after counter setup so a thread that immediately finds
  // work can safely decrement numNotWorking_ without being overwritten).
  for (size_t i = 0; i < n; ++i) {
    threads_.emplace_back();
    int32_t ringIdx = static_cast<int32_t>(i);
    if (enableEpochWaiter_.load(std::memory_order_acquire)) {
      threads_.back().setThread(std::thread([this, &back = threads_.back(), ringIdx]() {
        threadLoopWake(back, ringIdx);
      }));
    } else {
      threads_.back().setThread(std::thread([this, &back = threads_.back(), ringIdx]() {
        threadLoopPoll(back, ringIdx);
      }));
    }
  }

  if (!sn) {
    // Pool will run future tasks inline since we have no threads, but we still need to empty
    // current set of tasks
    while (tryExecuteNext()) {
    }
  }
}

ThreadPool::~ThreadPool() {
#if defined DISPENSO_DEBUG
  assert(outstandingTaskSets_.load(std::memory_order_acquire) == 0);
#endif // DISPENSO_DEBUG

  // Strictly speaking, it is unnecessary to lock this in the destructor; however, it could be a
  // useful diagnostic to learn that the mutex is already locked when we reach this point.
  std::unique_lock<std::mutex> lk(threadsMutex_, std::try_to_lock);
  assert(lk.owns_lock());

  // Mark all threads as stopped first, then wake them all at once.
  // One-at-a-time stop+wake is fragile: a wake() can reach an already-awake
  // thread while another remains sleeping, forcing it to wait for the epoch
  // timeout to notice the stop flag.
  for (auto& t : threads_) {
    t.stop();
  }
  {
    auto* ws = detail::consumeLoad(wakeState_);
    if (ws) {
      ws->wakeAll();
    }
  }

  while (tryExecuteNext()) {
  }

  for (auto& t : threads_) {
    t.thread_.join();
  }
  threads_.clear();

  // Drain central queue
  while (tryExecuteNext()) {
  }

  // Drain all rings in the arena (including shadow entries)
  for (size_t i = 0; i < rings_.size(); ++i) {
    OnceFunction task;
    while (rings_[i].try_pop(task)) {
      task();
    }
  }
  for (size_t i = 0; i < stealRings_.size(); ++i) {
    OnceFunction task;
    while (stealRings_[i].try_pop(task)) {
      task();
    }
  }
  // wakeState_ graveyard freed by RAII (vector destructor)
}
ThreadPool& globalThreadPool() {
  // It should be illegal to access globalThreadPool after exiting main.
  // We default to hardware threads minus one because the calling thread usually is involved in
  // computation.
  static ThreadPool pool(std::thread::hardware_concurrency() - 1);
  return pool;
}

void resizeGlobalThreadPool(size_t numThreads) {
  globalThreadPool().resize(static_cast<ssize_t>(numThreads));
}

} // namespace dispenso
