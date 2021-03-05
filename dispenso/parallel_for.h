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
   * the number associated with the TaskSet's thread pool.  Setting maxThreads to zero  will result
   * in serial operation.
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
 * describe how to break up work for parallel processing.  ChunkedRanges can be created with Auto
 * chunking, Static chunking, or specific chunking.  Auto chunking makes large chunks for better
 * cache utilization, but tries to make enough chunks to provide some dynamic load balancing. Static
 * chunking makes N chunks given N threads to run the loop on.  User-specified chunking can be
 * useful for ensuring e.g. that at least a multiple of SIMD width is provided per chunk.
 * <code>parallel_for</code> calls that don't accept a ChunkedRange will create a ChunkedRange
 * internally using Auto chunking.
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

  bool isStatic() const {
    return chunk == kStatic;
  }

  size_t size() const {
    return end - start;
  }

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
      size_t chunks = dynFactor * workingThreads;
      return (end - start + chunks) / chunks;
    } else if (chunk == kStatic) {
      // This should never be called.  The static distribution versions of the parallel_for
      // functions should be invoked instead.
      std::abort();
    }
    return chunk;
  }

  size_t start;
  size_t end;
  size_t chunk;
};

namespace detail {

template <typename TaskSetT, typename F>
void parallel_for_staticImpl(
    TaskSetT& taskSet,
    const ChunkedRange& range,
    F&& f,
    ParForOptions options) {
  size_t numThreads = std::min<size_t>(taskSet.numPoolThreads(), options.maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<size_t>(numThreads, range.size());

  auto chunking = detail::staticChunkSize(range.size(), numThreads);
  size_t chunkSize = chunking.ceilChunkSize;

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  size_t firstLoopLen = chunking.transitionTaskIndex - perfectlyChunked;

  size_t start = range.start;
  size_t t;
  for (t = 0; t < firstLoopLen; ++t) {
    size_t next = start + chunkSize;
    taskSet.schedule([start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(start, next);
    });
    start = next;
  }

  // Reduce the remaining chunk sizes by 1.
  chunkSize -= !perfectlyChunked;
  // Finish submitting all but the last item.
  for (; t < numThreads - 1; ++t) {
    size_t next = start + chunkSize;
    taskSet.schedule([start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(start, next);
    });
    start = next;
  }

  if (options.wait) {
    f(start, range.end);
    taskSet.wait();
  } else {
    taskSet.schedule(
        [start, end = range.end, f]() {
          auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
          f(start, end);
        },
        ForceQueuingTag());
  }
}

template <typename TaskSetT, typename F, typename StateContainer, typename StateGen>
void parallel_for_staticImpl(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange& range,
    F&& f,
    ParForOptions options) {
  size_t numThreads = std::min<size_t>(taskSet.numPoolThreads(), options.maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<size_t>(numThreads, range.size());

  for (size_t i = 0; i < numThreads; ++i) {
    states.emplace_back(defaultState());
  }

  auto chunking = detail::staticChunkSize(range.size(), numThreads);
  size_t chunkSize = chunking.ceilChunkSize;

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  size_t firstLoopLen = chunking.transitionTaskIndex - perfectlyChunked;

  auto stateIt = states.begin();
  size_t start = range.start;
  size_t t;
  for (t = 0; t < firstLoopLen; ++t) {
    size_t next = start + chunkSize;
    taskSet.schedule([it = stateIt++, start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(*it, start, next);
    });
    start = next;
  }

  // Reduce the remaining chunk sizes by 1.
  chunkSize -= !perfectlyChunked;
  // Finish submitting all but the last item.
  for (; t < numThreads - 1; ++t) {
    size_t next = start + chunkSize;
    taskSet.schedule([it = stateIt++, start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(*it, start, next);
    });
    start = next;
  }

  if (options.wait) {
    f(*stateIt, start, range.end);
    taskSet.wait();
  } else {
    taskSet.schedule(
        [stateIt, start, end = range.end, f]() {
          auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
          f(*stateIt, start, end);
        },
        ForceQueuingTag());
  }
}

} // namespace detail

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
  if (!options.maxThreads || detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    f(range.start, range.end);
    return;
  }

  if (range.isStatic()) {
    detail::parallel_for_staticImpl(taskSet, range, std::forward<F>(f), options);
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset
 *to true.
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
  if (!options.maxThreads || detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    states.emplace_back(defaultState());
    f(*states.begin(), range.start, range.end);
    return;
  }

  if (range.isStatic()) {
    detail::parallel_for_staticImpl(
        taskSet, states, defaultState, range, std::forward<F>(f), options);
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
      taskSet.schedule([&s = *it++, worker]() { worker(s); });
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
      taskSet.schedule([&s = *it++, worker]() { worker(s); }, ForceQueuingTag());
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset
 *to true.
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset
 *to true.
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
 * @param options See ParForOptions for details.  <code>options.wait</code> will always be reset
 *to true.
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
