/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/quanta.h>
#include <dispenso/thread_pool.h>

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

void ThreadPool::PerThreadData::setThread(std::thread&& t) {
  thread_ = std::move(t);
}

void ThreadPool::PerThreadData::stop() {
  running_.store(false, std::memory_order_release);
}

uint32_t ThreadPool::wait(uint32_t currentEpoch) {
  if (sleepLengthUs_ > 0) {
    return epochWaiter_.waitFor(currentEpoch, sleepLengthUs_.load(std::memory_order_acquire));
  } else {
    return epochWaiter_.current();
  }
}
void ThreadPool::wake() {
  epochWaiter_.bumpAndWake();
}

inline bool ThreadPool::PerThreadData::running() {
  return running_.load(std::memory_order_acquire);
}

ThreadPool::ThreadPool(size_t n, size_t poolLoadMultiplier)
    : poolLoadMultiplier_(poolLoadMultiplier),
      poolLoadFactor_(static_cast<ssize_t>(getAdjustedThreadCount(n) * poolLoadMultiplier)),
      numThreads_(static_cast<ssize_t>(getAdjustedThreadCount(n))) {
  detail::registerFineSchedulerQuanta();
#if defined DISPENSO_DEBUG
  assert(poolLoadMultiplier > 0);
#endif // DISPENSO_DEBUG

  size_t adjustedN = static_cast<size_t>(numThreads_);
  // Allocate per-thread rings
  if (adjustedN > 0) {
    rings_ = std::make_unique<Ring[]>(adjustedN);
    numRings_.store(adjustedN, std::memory_order_release);
  }

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

// On Windows, wake syscalls (WakeByAddressSingle) are expensive per-thread
// kernel transitions, so spin longer before sleeping to avoid costly
// sleep/wake cycles.
#if defined(_WIN32)
static constexpr int kBackoffYield = 100;
static constexpr int kBackoffSleep = kBackoffYield + 20;
#else
static constexpr int kBackoffYield = 50;
static constexpr int kBackoffSleep = kBackoffYield + 5;
#endif

void ThreadPool::threadLoopWake(PerThreadData& data, int32_t ringIndex) {
  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  int failCount = 0;
  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken, ringIndex);
  uint32_t epoch = epochWaiter_.current();
  size_t myRingIndex = static_cast<size_t>(ringIndex);

  while (data.running()) {
    int localWorkDone = 0;
    while (tryFindAndExecuteWork(myRingIndex, ctoken)) {
      ++localWorkDone;
      if (localWorkDone >= kWorkBatchSize) {
        workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
        localWorkDone = 0;
      }
      failCount = 0;
    }
    if (localWorkDone > 0) {
      workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
    }

    ++failCount;

    detail::cpuRelax();
    if (failCount > kBackoffSleep) {
      numSleeping_.fetch_add(1, std::memory_order_acq_rel);
      epoch = wait(epoch);
      numSleeping_.fetch_sub(1, std::memory_order_acq_rel);
      failCount = 0;
    } else if (failCount > kBackoffYield) {
      std::this_thread::yield();
    }
  }
}

void ThreadPool::threadLoopPoll(PerThreadData& data, int32_t ringIndex) {
  moodycamel::ConsumerToken ctoken(work_);
  moodycamel::ProducerToken ptoken(work_);

  int failCount = 0;
  detail::PerPoolPerThreadInfo::registerPool(this, &ptoken, ringIndex);
  uint32_t epoch = epochWaiter_.current();
  size_t myRingIndex = static_cast<size_t>(ringIndex);

  while (data.running()) {
    int localWorkDone = 0;
    while (tryFindAndExecuteWork(myRingIndex, ctoken)) {
      failCount = 0;
      ++localWorkDone;
      if (localWorkDone >= kWorkBatchSize) {
        workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
        localWorkDone = 0;
      }
    }
    if (localWorkDone > 0) {
      workRemaining_.fetch_sub(localWorkDone, std::memory_order_relaxed);
    }

    ++failCount;

    detail::cpuRelax();
    if (failCount > kBackoffSleep) {
      epoch = wait(epoch);
    } else if (failCount > kBackoffYield) {
      std::this_thread::yield();
    }
  }
}

void ThreadPool::resizeLocked(ssize_t sn) {
  sn = getAdjustedThreadCount(sn);

  assert(sn >= 0);
  size_t n = static_cast<size_t>(sn);

  if (n < threads_.size()) {
    // Mark all excess threads as stopped first, then wake them all at once.
    // See destructor comment for rationale (one-at-a-time wake is fragile).
    ssize_t excessCount = static_cast<ssize_t>(threads_.size()) - static_cast<ssize_t>(n);
    for (size_t i = n; i < threads_.size(); ++i) {
      threads_[i].stop();
    }
    if (excessCount > 0) {
      wakeN(excessCount);
    }
    // Join from back to front
    while (threads_.size() > n) {
      threads_.back().thread_.join();
      threads_.pop_back();
    }

    // Drain rings of stopped threads back to the central queue under exclusive
    // ringsLock_. Without this, tasks placed via scheduleBulkToRings into rings
    // [n, numRings_) would be stranded with no consumer, causing TaskSet::wait()
    // to hang on outstandingTaskCount_ never reaching zero.
    size_t oldRingCount = numRings_.load(std::memory_order_relaxed);
    if (n < oldRingCount) {
      ringsLock_.lock();
      for (size_t i = n; i < oldRingCount; ++i) {
        OnceFunction task;
        while (rings_[i].try_pop(task)) {
          work_.enqueue(std::move(task));
        }
      }
      numRings_.store(n, std::memory_order_release);
      ringsLock_.unlock();
    }

  } else if (n > threads_.size()) {
    // Grow rings array if needed — acquire exclusive lock to prevent concurrent
    // ring access from thread loops (tryFindAndExecuteWork) and scheduleBulkToRings.
    if (n > numRings_.load(std::memory_order_relaxed)) {
      ringsLock_.lock();
      // Drain old rings to central queue before freeing
      size_t oldCount = numRings_.load(std::memory_order_relaxed);
      for (size_t i = 0; i < oldCount; ++i) {
        OnceFunction task;
        while (rings_[i].try_pop(task)) {
          work_.enqueue(std::move(task));
        }
      }
      rings_ = std::make_unique<Ring[]>(n);
      numRings_.store(n, std::memory_order_release);
      ringsLock_.unlock();
    }
    for (size_t i = threads_.size(); i < n; ++i) {
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
  }
  poolLoadFactor_.store(static_cast<ssize_t>(n * poolLoadMultiplier_), std::memory_order_relaxed);
  numThreads_.store(sn, std::memory_order_relaxed);

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
  ssize_t numThreads = static_cast<ssize_t>(threads_.size());
  if (numThreads > 0) {
    wakeN(numThreads);
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

  // Drain any remaining ring work (threads already joined, no lock needed)
  size_t ringCount = numRings_.load(std::memory_order_relaxed);
  for (size_t i = 0; i < ringCount; ++i) {
    OnceFunction task;
    while (rings_[i].try_pop(task)) {
      task();
    }
  }
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

void ThreadPool::wakeN(ssize_t n) {
#if defined(_WIN32)
  // On Windows, always use WakeAll. WakeByAddressSingle has no batch wake
  // count, so multiple calls means O(N) kernel transitions. A single WakeAll
  // is one transition and keeps threads in their spin phase (kBackoffYield
  // iterations) where they can absorb follow-up work without re-waking.
  (void)n;
  epochWaiter_.bumpAndWakeAll();
#else
  ssize_t sleeping = numSleeping_.load(std::memory_order_relaxed);
  constexpr unsigned kWakeAllMultiplier = 2;
  if (static_cast<unsigned>(n) * kWakeAllMultiplier >= static_cast<unsigned>(sleeping)) {
    epochWaiter_.bumpAndWakeAll();
  } else {
    epochWaiter_.bumpAndWakeN(static_cast<int>(n), static_cast<int>(sleeping));
  }
#endif
}

} // namespace dispenso
