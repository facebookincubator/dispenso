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
 * A set of options to control parallel_for
 **/
struct ParForOptions {
  /**
   * The maximum number of threads to use.  This can be used to limit the number of threads below
   * the number associated with the TaskSet's thread pool.
   **/
  uint32_t maxThreads = std::numeric_limits<uint32_t>::max();
  /**
   * Specify whether the return of the parallel_for signifies the work is complete.  If the
   * parallel_for is initiated without providing a TaskSet, the parallel_for will always wait.
   *
   * @note If wait is true, the calling thread will always participate in computation.  If this is
   * not desired, pass wait as false, and wait manually outside of the parallel_for on the passed
   * TaskSet.
   **/
  bool wait = true;
};

/**
 * A helper class for <code>parallel_for</code>.  It provides various configuration parameters to
 * describe how to break up work for parallel processing.
 **/
struct ChunkedRange {
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
  ChunkedRange(size_t s, size_t e, size_t c) : start(s), end(e), chunk(c) {}
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

  size_t calcChunkSize(size_t numLaunched, bool oneOnCaller) const {
    size_t workingThreads = numLaunched + oneOnCaller;
    if (workingThreads == 1) {
      return end - start;
    }
    if (!chunk) {
      // TODO(bbudge): play with different load balancing factors for auto.
      // size_t dynFactor = std::log2(1.0f + (end_ - start_) / (cyclesPerIndex_ *
      // cyclesPerIndex_));
      constexpr size_t dynFactor = 16;
      return std::max<size_t>(
          1, (end - start + dynFactor * (workingThreads - 1)) / (dynFactor * workingThreads));
    } else if (chunk == kStatic) {
      return (end - start + workingThreads - 1) / workingThreads;
    }
    return chunk;
  }

  size_t start;
  size_t end;
  size_t chunk;
};

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t begin, size_t end)</code>.
 * @param options See ParForOptions for details.
 **/
template <typename TaskSetT, typename F>
void parallel_for(TaskSetT& taskSet, const ChunkedRange& range, F&& f, ParForOptions options = {}) {
  // TODO(bbudge): With options.maxThreads, we might want to allow a small fanout factor in
  // recursive case?
  if (detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    f(range.start, range.end);
    return;
  }

  const size_t N = taskSet.numPoolThreads();
  const bool useCallingThread = options.wait;
  const size_t numToLaunch = std::min<size_t>(options.maxThreads, N - useCallingThread);
  const size_t chunk = range.calcChunkSize(numToLaunch, useCallingThread);

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<size_t> index(range.start);
    auto worker = [end = range.end, &index, f = std::move(f), chunk]() {
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
    // TODO(bbudge): dispenso::make_shared?
    auto wrapper = std::make_shared<Atomic>(range.start);
    auto worker = [end = range.end, wrapper = std::move(wrapper), f, chunk]() {
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
      taskSet.schedule(Function(worker), ForceQueuingTag());
    }
  }
}

/**
 * Execute loop over the range in parallel on the global thread pool, and wait until complete.
 *
 * @param range The range defining the loop extents as well as chunking strategy.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t begin, size_t end)</code>.
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset to
 * true.
 **/
template <typename F>
void parallel_for(const ChunkedRange& range, F&& f, ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  parallel_for(taskSet, range, std::forward<F>(f), options);
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
 * @param options See ParForOptions for details.
 **/
template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange& range,
    F&& f,
    ParForOptions options = {}) {
  if (detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    states.emplace_back(defaultState());
    f(*states.begin(), range.start, range.end);
    return;
  }

  const size_t N = taskSet.numPoolThreads();
  const bool useCallingThread = options.wait;
  const size_t numToLaunch = std::min<size_t>(options.maxThreads, N - useCallingThread);
  const size_t chunk = range.calcChunkSize(numToLaunch, useCallingThread);

  for (size_t i = 0; i < numToLaunch + useCallingThread; ++i) {
    states.emplace_back(defaultState());
  }

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<size_t> index(range.start);
    auto worker = [end = range.end, &index, f, chunk](auto& s) {
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
    auto wrapper = std::make_shared<Atomic>(range.start);
    auto worker = [end = range.end, wrapper = std::move(wrapper), f, chunk](auto& s) {
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
      taskSet.schedule([& s = *it++, worker]() { worker(s); }, ForceQueuingTag());
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset to
 * true.
 **/
template <typename F, typename StateContainer, typename StateGen>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange& range,
    F&& f,
    ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  parallel_for(taskSet, states, defaultState, range, std::forward<F>(f), options);
}

/**
 * Execute loop over the range in parallel.
 *
 * @param taskSet The task set to schedule the loop on.
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t index)</code>.
 * @param options See ParForOptions for details.
 **/
template <typename TaskSetT, typename F>
void parallel_for(TaskSetT& taskSet, size_t start, size_t end, F&& f, ParForOptions options = {}) {
  parallel_for(
      taskSet,
      ChunkedRange(start, end, ChunkedRange::Auto()),
      [f = std::move(f)](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) {
          f(i);
        }
      },
      options);
}

/**
 * Execute loop over the range in parallel on the global thread pool and block on loop completion.
 *
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t index)</code>.
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset to
 * true.
 **/
template <typename F>
void parallel_for(size_t start, size_t end, F&& f, ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  parallel_for(taskSet, start, end, std::forward<F>(f), options);
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
 * @param options See ParForOptions for details.
 **/
template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    size_t start,
    size_t end,
    F&& f,
    ParForOptions options = {}) {
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
      options);
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset to
 * true.
 **/
template <typename F, typename StateContainer, typename StateGen>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    size_t start,
    size_t end,
    F&& f,
    ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  return parallel_for(taskSet, states, defaultState, start, end, std::forward<F>(f), options);
}

} // namespace dispenso
