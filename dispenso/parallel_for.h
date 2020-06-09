// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

/**
 * @file functions for performing parallel for loops.
 **/

#pragma once

#include <cmath>
#include <limits>

#include <dispenso/detail/per_thread_info.h>
#include <dispenso/task_set.h>

namespace dispenso {

/**
 * A helper class for <code>parallel_for</code>.  It provides various configuration parameters to
 * describe how to break up work for parallel processing.
 **/
class ChunkedRange {
 public:
  struct Static {};
  struct Auto {};
  static constexpr size_t kStatic = std::numeric_limits<size_t>::max();

  /**
   * Create a ChunkedRange with specific chunk size
   *
   * @param s The start of the range.
   * @param e The end of the range.
   * @param c The chunk size.
   **/
  ChunkedRange(size_t s, size_t e, size_t c) : start_(s), end_(e), chunk_(c) {}
  /**
   * Create a ChunkedRange with chunk size equal to total items divided by number of threads.
   *
   * @param s The start of the range.
   * @param e The end of the range.
   **/
  ChunkedRange(size_t s, size_t e, Static) : ChunkedRange(s, e, kStatic) {}
  /**
   * Create a ChunkedRange with chunk size determined automatically to enable some dynamic load
   * balancing.
   *
   * @param s The start of the range.
   * @param e The end of the range.
   **/
  ChunkedRange(size_t s, size_t e, Auto) : ChunkedRange(s, e, 0) {}

 private:
  size_t calcChunkSize(size_t numLaunched, bool wait) const {
    size_t workingThreads = numLaunched + wait;
    if (workingThreads == 1) {
      return end_ - start_;
    }
    if (!chunk_) {
      // TODO(bbudge): play with different load balancing factors for auto.
      // size_t dynFactor = std::log2(1.0f + (end_ - start_) / (cyclesPerIndex_ * cyclesPerIndex_));
      constexpr size_t dynFactor = 16;
      return std::max<size_t>(
          1, (end_ - start_ + dynFactor * (workingThreads - 1)) / (dynFactor * workingThreads));
    } else if (chunk_ == kStatic) {
      return (end_ - start_ + workingThreads - 1) / workingThreads;
    }
    return chunk_;
  }

  size_t start_;
  size_t end_;
  size_t chunk_;
  template <typename TaskSetT, typename F>
  friend void parallel_for(TaskSetT&, const ChunkedRange&, F&&, bool wait);
  template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
  friend void
  parallel_for(TaskSetT&, StateContainer&, const StateGen&, const ChunkedRange&, F&&, bool wait);
};

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t begin, size_t end)</code>.
 * @param wait If true, this function blocks until all work is complete.  If false, the user should
 * wait on <code>taskSet</code> before assuming the loop has been completed.
 **/
template <typename TaskSetT, typename F>
void parallel_for(TaskSetT& taskSet, const ChunkedRange& range, F&& f, bool wait) {
  if (detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    f(range.start_, range.end_);
    return;
  }

  const size_t N = taskSet.numPoolThreads();
  const size_t numToLaunch = detail::getNumToLaunch(wait, N);
  const size_t chunk = range.calcChunkSize(numToLaunch, wait);

  if (wait) {
    alignas(kCacheLineSize) std::atomic<size_t> index(range.start_);
    auto worker = [end = range.end_, &index, f = std::move(f), chunk]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        size_t cur = index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(cur, std::min<size_t>(cur + chunk, end));
      }
    };

    using Function = decltype(worker);

    for (size_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule(Function(worker));
    }
    worker();
    taskSet.wait();
  } else {
    struct Atomic {
      Atomic(size_t i) : index(i) {}
      char buffer[kCacheLineSize];
      std::atomic<size_t> index;
      char buffer2[kCacheLineSize];
    };
    auto wrapper = std::make_shared<Atomic>(range.start_);
    auto worker = [end = range.end_, wrapper = std::move(wrapper), f, chunk]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        size_t cur = wrapper->index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(cur, std::min<size_t>(cur + chunk, end));
      }
    };

    using Function = decltype(worker);

    for (size_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule(Function(worker));
    }
  }
}

/**
 * Execute loop over the range in parallel on the global thread pool, and wait until complete.
 *
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t begin, size_t end)</code>.
 **/
template <typename F>
void parallel_for(const ChunkedRange& range, F&& f) {
  TaskSet taskSet(globalThreadPool());
  parallel_for(taskSet, range, std::forward<F>(f), true);
}

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param states A container of <code>State</code> (actual type of State TBD by user).  The
 * container will be resized to hold a <code>State</code> object per executing thread.  Container
 * must provide emplace_back() and must be forward-iterable.  Examples include std::vector,
 * std::deque, and std::list.  These are the states passed into <code>f</code>, and states must
 * remain a valid object until work is completed.
 * @param defaultState A functor with signature State().  It will be called to initialize the
 * objects for <code>states</code>.
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(State &s, size_t begin, size_t end)</code>.
 * @param wait If true, this function blocks until all work is complete.  If false, the user should
 * wait on <code>taskSet</code> before assuming the loop has been completed.
 **/
template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange& range,
    F&& f,
    bool wait) {
  if (detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    states.emplace_back(defaultState());
    f(*states.begin(), range.start_, range.end_);
    return;
  }

  const size_t N = taskSet.numPoolThreads();
  const size_t numToLaunch = detail::getNumToLaunch(wait, N);
  size_t chunk = range.calcChunkSize(numToLaunch, wait);

  for (size_t i = 0; i < numToLaunch + wait; ++i) {
    states.emplace_back(defaultState());
  }

  if (wait) {
    alignas(kCacheLineSize) std::atomic<size_t> index(range.start_);
    auto worker = [end = range.end_, &index, f, chunk](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();

      while (true) {
        size_t cur = index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(s, cur, std::min<size_t>(cur + chunk, end));
      }
    };

    auto it = states.begin();
    for (size_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule([& s = *it++, worker]() { worker(s); });
    }
    worker(*it);
    taskSet.wait();
  } else {
    struct Atomic {
      Atomic(size_t i) : index(i) {}
      char buffer[kCacheLineSize];
      std::atomic<size_t> index;
      char buffer2[kCacheLineSize];
    };
    auto wrapper = std::make_shared<Atomic>(range.start_);
    auto worker = [end = range.end_, wrapper = std::move(wrapper), f, chunk](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        size_t cur = wrapper->index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(s, cur, std::min<size_t>(cur + chunk, end));
      }
    };

    auto it = states.begin();
    for (size_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule([& s = *it++, worker]() { worker(s); });
    }
  }
}

/**
 * Execute loop over the range in parallel on the global thread pool and block until loop
 *completion.
 *
 * @param states A container of <code>State</code> (actual type of State TBD by user).  The
 * container will be resized to hold a <code>State</code> object per executing thread.  Container
 * must provide emplace_back() and must be forward-iterable.  Examples include std::vector,
 * std::deque, and std::list.  These are the states passed into <code>f</code>, and states must
 * remain a valid object until work is completed.
 * @param defaultState A functor with signature State().  It will be called to initialize the
 * objects for <code>states</code>.
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(State &s, size_t begin, size_t end)</code>.
 **/
template <typename F, typename StateContainer, typename StateGen>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange& range,
    F&& f) {
  TaskSet taskSet(globalThreadPool());
  parallel_for(taskSet, states, defaultState, range, std::forward<F>(f), true);
}

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t index)</code>.
 * @param wait If true, this function blocks until all work is complete.  If false, the user should
 * wait on <code>taskSet</code> before assuming the loop has been completed.
 **/
template <typename TaskSetT, typename F>
void parallel_for(TaskSetT& taskSet, size_t start, size_t end, F&& f, bool wait = true) {
  parallel_for(
      taskSet,
      ChunkedRange(start, end, ChunkedRange::Auto()),
      [f = std::move(f)](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) {
          f(i);
        }
      },
      wait);
}

/**
 * Execute loop over the range in parallel on the global thread pool and block on loop completion.
 *
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t index)</code>.
 **/
template <typename F>
void parallel_for(size_t start, size_t end, F&& f) {
  TaskSet taskSet(globalThreadPool());
  parallel_for(taskSet, start, end, std::forward<F>(f));
}

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param states A container of <code>State</code> (actual type of State TBD by user).  The
 * container will be resized to hold a <code>State</code> object per executing thread.  Container
 * must provide emplace_back() and must be forward-iterable.  Examples include std::vector,
 * std::deque, and std::list.  These are the states passed into <code>f</code>, and states must
 * remain a valid object until work is completed.
 * @param defaultState A functor with signature State().  It will be called to initialize the
 * objects for <code>states</code>.
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(State &s, size_t begin, size_t end)</code>.
 * @param wait If true, this function blocks until all work is complete.  If false, the user should
 * wait on <code>taskSet</code> before assuming the loop has been completed.
 **/
template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    size_t start,
    size_t end,
    F&& f,
    bool wait = true) {
  parallel_for(
      taskSet,
      states,
      defaultState,
      ChunkedRange(start, end, ChunkedRange::Auto()),
      [f = std::move(f)](auto& state, size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) {
          f(state, i);
        }
      },
      wait);
}

/**
 * Execute loop over the range in parallel on the global thread pool and block until loop
 *completion.
 *
 * @param states A container of <code>State</code> (actual type of State TBD by user).  The
 * container will be resized to hold a <code>State</code> object per executing thread.  Container
 * must provide emplace_back() and must be forward-iterable.  Examples include std::vector,
 * std::deque, and std::list.  These are the states passed into <code>f</code>, and states must
 * remain a valid object until work is completed.
 * @param defaultState A functor with signature State().  It will be called to initialize the
 * objects for <code>states</code>.
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(State &s, size_t begin, size_t end)</code>.
 **/
template <typename F, typename StateContainer, typename StateGen>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    size_t start,
    size_t end,
    F&& f) {
  TaskSet taskSet(globalThreadPool());
  return parallel_for(taskSet, states, defaultState, start, end, std::forward<F>(f));
}

} // namespace dispenso
