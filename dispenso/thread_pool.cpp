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
// Windows: WakeByAddressSingle is expensive. Use adaptive time-based spin
// to avoid costly sleep/wake transitions between parallel_for iterations.
static constexpr int32_t kDefaultSpinLimit = 0; // adaptive mode
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

  spinLimit_ = kDefaultSpinLimit;

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

// Adaptive backoff: spin with cpuRelax, then periodically check elapsed time
// against a threshold to decide when to yield and when to sleep.
// kSpinCheckInterval: how many spin iterations between getTime() checks.
// kMaxSpinTimeoutSec: upper bound on spin time — used when work arrives regularly.
// kMinSpinTimeoutSec: lower bound — used after repeated idle sleeps.
// After each sleep, if we wake and find no work the spin timeout halves (down to
// kMinSpinTimeoutSec). Finding work resets the timeout to kMaxSpinTimeoutSec.
// This keeps threads hot during sustained parallel_for bursts but quickly backs
// off to near-zero spin when the pool is idle.
// Override via -DDISPENSO_TUNE_SPIN_CHECK_INTERVAL=N etc. for cross-platform tuning.
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

#if defined(DISPENSO_TUNE_MAX_SPIN_US)
static constexpr double kMaxSpinTimeoutSec = DISPENSO_TUNE_MAX_SPIN_US * 1e-6;
#elif defined(__APPLE__)
// macOS: __ulock wake is faster than Linux futex. 128us is a safe default
// across core counts — 64us showed slight edge on 12-core but regressed
// key benchmarks.
static constexpr double kMaxSpinTimeoutSec = 128e-6; // 128us
#elif defined(_WIN32)
// Windows: WaitOnAddress wake latency is high, so spin longer to avoid
// costly sleep/wake transitions between parallel_for iterations.
static constexpr double kMaxSpinTimeoutSec = 256e-6; // 256us
#else
static constexpr double kMaxSpinTimeoutSec = 128e-6; // 128us
#endif

#if defined(DISPENSO_TUNE_MIN_SPIN_US)
static constexpr double kMinSpinTimeoutSec = DISPENSO_TUNE_MIN_SPIN_US * 1e-6;
#else
static constexpr double kMinSpinTimeoutSec = 1e-6; // 1us
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

void ThreadPool::markWorkDone(bool& isWorking, double& spinTimeout, bool& wokeFromSleep) {
  if (!isWorking) {
    numNotWorking_.fetch_sub(1, std::memory_order_relaxed);
    isWorking = true;
  }
  if (!wokeFromSleep) {
    spinTimeout = std::min(kMaxSpinTimeoutSec, spinTimeout * 2.0);
  }
  wokeFromSleep = false;
}

void ThreadPool::markIdle(bool& isWorking) {
  if (isWorking) {
    numNotWorking_.fetch_add(1, std::memory_order_relaxed);
    isWorking = false;
  }
}

void ThreadPool::threadLoopWake(PerThreadData& data, int32_t ringIndex) {
  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  // Start preferring queue — schedule() enqueues there.
  bool preferRing = false;

  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken, ringIndex);
  auto* ws = detail::consumeLoad(wakeState_);
  uint32_t epoch = ws->waiterFor(ringIndex).current();
  size_t myRingIndex = static_cast<size_t>(ringIndex);
  Ring& myRing = rings_[myRingIndex];
  size_t myStealIdx = static_cast<size_t>(ringIndex) / stealRingSharing_;
  StealRing& myStealRing = stealRings_[myStealIdx];

  int failCount = 0;
  double idleStart = 0.0;
  double spinTimeout = kMaxSpinTimeoutSec;
  bool wokeFromSleep = false;
  bool isWorking = false; // starts not-working (counted in constructor init)

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
      markWorkDone(isWorking, spinTimeout, wokeFromSleep);
      workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
      failCount = 0;
      continue;
    }

    // --- No work found. Lean spin: only check the local ring to
    // minimize cross-socket cache traffic from idle threads. ---

    ++failCount;

    if (failCount < kSpinCheckInterval) {
      detail::cpuRelax();
      continue;
    }

    // --- Full machinery: steal ring, processBudget, adaptive backoff ---

    // Outer steal ring check (deferred from lean phase). No empty() guard:
    // try_pop already fast-paths the empty case with a relaxed head/tail compare
    // before any acquire load, so a separate empty() check would be redundant.
    {
      OnceFunction stealTask;
      if (myStealRing.try_pop(stealTask)) {
        markWorkDone(isWorking, spinTimeout, wokeFromSleep);
        executeNext(std::move(stealTask));
        failCount = 0;
        continue;
      }
    }

    // Intentional second increment: lean-phase failures (above) count once,
    // full-phase failures count twice. kDefaultSpinLimit is tuned assuming
    // this two-rate progression. In adaptive mode (spinLimit_ == 0) the
    // odd-aligned failCount values make the bitmask-aligned checks below
    // unreachable; parking is handled by the backstop timeout in
    // waitOnThread instead.
    ++failCount;

    if (spinLimit_ > 0) {
      // Fixed-spin mode: sleep after N total iterations, no time checks.
      ws->processBudget(ringIndex);
      detail::cpuRelax();

      if (failCount >= spinLimit_) {
        markIdle(isWorking);
        ws->enterSleep(ringIndex);
        if (!data.running()) {
          ws->exitSleep(ringIndex);
          break;
        }
        epoch = waitOnThread(ringIndex, epoch);
        ws->exitSleep(ringIndex);
        failCount = 0;
        wokeFromSleep = true;
      }
    } else {
      // Adaptive mode (default): time-based spin with exponential backoff.
      ws->processBudget(ringIndex);

      if (failCount == kSpinCheckInterval) {
        idleStart = getTime();
      }

      detail::cpuRelax();

      if ((failCount & (kSpinCheckInterval - 1)) == 0) {
        markIdle(isWorking);

        double elapsed = getTime() - idleStart;
        if (elapsed > spinTimeout) {
          ws->enterSleep(ringIndex);
          epoch = waitOnThread(ringIndex, epoch);
          ws->exitSleep(ringIndex);
          failCount = 0;
          wokeFromSleep = true;
          spinTimeout = std::max(kMinSpinTimeoutSec, spinTimeout * 0.5);
        } else if (elapsed > spinTimeout * 0.5) {
          std::this_thread::yield();
        }
      }
    }
  }

  // Clean up on thread exit (shutdown/resize): restore counter.
  markIdle(isWorking);
}

void ThreadPool::threadLoopPoll(PerThreadData& data, int32_t ringIndex) {
  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  bool preferRing = false;

  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken, ringIndex);
  auto* ws = detail::consumeLoad(wakeState_);
  uint32_t epoch = ws->waiterFor(ringIndex).current();
  size_t myRingIndex = static_cast<size_t>(ringIndex);
  Ring& myRing = rings_[myRingIndex];
  size_t myStealIdx = static_cast<size_t>(ringIndex) / stealRingSharing_;
  StealRing& myStealRing = stealRings_[myStealIdx];

  int failCount = 0;
  double idleStart = 0.0;
  double spinTimeout = kMaxSpinTimeoutSec;
  bool wokeFromSleep = false;
  bool isWorking = false;
  // Wake leaders: threads in the first subgroup of each group (positions
  // 0-3 in a group of 16). These threads share a futex and all wake from
  // the same bumpAndWakeAll call. All 4 participate in cascade fan-out.
  // Cascade-team threads (first subgroup of each group) check for wake
  // budgets via the inlined processBudget fast path.

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
      markWorkDone(isWorking, spinTimeout, wokeFromSleep);
      workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
      failCount = 0;
      continue;
    }

    // Lean spin for the first kSpinCheckInterval iterations.
    ++failCount;

    if (failCount < kSpinCheckInterval) {
      detail::cpuRelax();
      continue;
    }

    // Full machinery after lean phase.
    {
      OnceFunction stealTask;
      if (myStealRing.try_pop(stealTask)) {
        markWorkDone(isWorking, spinTimeout, wokeFromSleep);
        executeNext(std::move(stealTask));
        failCount = 0;
        continue;
      }
    }

    // Intentional second increment: lean-phase failures (above) count once,
    // full-phase failures count twice. kDefaultSpinLimit is tuned assuming
    // this two-rate progression. In adaptive mode (spinLimit_ == 0) the
    // odd-aligned failCount values make the bitmask-aligned checks below
    // unreachable; parking is handled by the backstop timeout in
    // waitOnThread instead.
    ++failCount;

    if (failCount == kSpinCheckInterval) {
      idleStart = getTime();
    }

    detail::cpuRelax();

    if (spinLimit_ > 0) {
      ws->processBudget(ringIndex);
      if (failCount >= spinLimit_) {
        markIdle(isWorking);
        epoch = waitOnThread(ringIndex, epoch);
        failCount = 0;
        wokeFromSleep = true;
      }
    } else if ((failCount & (kSpinCheckInterval - 1)) == 0) {
      markIdle(isWorking);

      ws->processBudget(ringIndex);

      if (failCount == kSpinCheckInterval) {
        idleStart = getTime();
      }

      double elapsed = getTime() - idleStart;
      if (elapsed > spinTimeout) {
        epoch = waitOnThread(ringIndex, epoch);
        failCount = 0;
        wokeFromSleep = true;
        spinTimeout = std::max(kMinSpinTimeoutSec, spinTimeout * 0.5);
      } else if (elapsed > spinTimeout * 0.5) {
        std::this_thread::yield();
      }
    }
  }

  markIdle(isWorking);
}

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
