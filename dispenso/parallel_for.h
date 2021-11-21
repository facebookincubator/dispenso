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
 * Chunking strategy.  Typically if the cost of each loop iteration is roughly constant, kStatic
 * load balancing is preferred.  Additionally, when making a non-waiting parallel_for call in
 * conjunction with other parallel_for calls or with other task submissions to a TaskSet, some
 * dynamic load balancing is automatically introduced, and selecting kStatic load balancing here can
 * be better.  If the workload per iteration deviates a lot from constant, and some ranges may be
 * much cheaper than others, select kAuto.
 **/
enum class ParForChunking { kStatic, kAuto };

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

  /**
   * Specify whether default chunking should be static or auto (dynamic load balancing).  This is
   * used when invoking the version of parallel_for that takes index parameters (vs a ChunkedRange).
   **/
  ParForChunking defaultChunking = ParForChunking::kStatic;
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
template <typename IntegerT = ssize_t>
struct ChunkedRange {
  struct Static {};
  struct Auto {};
  static constexpr IntegerT kStatic = std::numeric_limits<IntegerT>::max();

  /**
   * Create a ChunkedRange with specific chunk size
   *
   * @param s The start of the range.
   * @param e The end of the range.
   * @param c The chunk size.
   **/
  ChunkedRange(IntegerT s, IntegerT e, IntegerT c) : start(s), end(e), chunk(c) {}
  /**
   * Create a ChunkedRange with chunk size equal to total items divided by number of threads.
   *
   * @param s The start of the range.
   * @param e The end of the range.
   **/
  ChunkedRange(IntegerT s, IntegerT e, Static) : ChunkedRange(s, e, kStatic) {}
  /**
   * Create a ChunkedRange with chunk size determined automatically to enable some dynamic load
   * balancing.
   *
   * @param s The start of the range.
   * @param e The end of the range.
   **/
  ChunkedRange(IntegerT s, IntegerT e, Auto) : ChunkedRange(s, e, 0) {}

  bool isStatic() const {
    return chunk == kStatic;
  }

  bool empty() const {
    return end <= start;
  }

  // This code returns int64 in order to avoid overflow, e.g. passing -2**30, 2**30 as int32 will
  // result in overflow unless we cast to 64-bit.
  int64_t size() const {
    return static_cast<int64_t>(end) - start;
  }

  template <typename OtherInt>
  IntegerT calcChunkSize(OtherInt numLaunched, bool oneOnCaller) const {
    ssize_t workingThreads = static_cast<ssize_t>(numLaunched) + ssize_t{oneOnCaller};
    assert(workingThreads > 1);

    if (!chunk) {
      // TODO(bbudge): play with different load balancing factors for auto.
      // IntegerT dynFactor = std::log2(1.0f + (end_ - start_) / (cyclesPerIndex_ *
      // cyclesPerIndex_));
      constexpr ssize_t dynFactor = 16;
      const ssize_t chunks = dynFactor * workingThreads;
      return static_cast<IntegerT>((size() + chunks) / chunks);
    } else if (chunk == kStatic) {
      // This should never be called.  The static distribution versions of the parallel_for
      // functions should be invoked instead.
      std::abort();
    }
    return chunk;
  }

  IntegerT start;
  IntegerT end;
  IntegerT chunk;
};

/**
 * Create a ChunkedRange with specified chunking strategy.
 *
 * @param start The start of the range.
 * @param end The end of the range.
 * @param chunking The strategy to use for chunking.
 **/
template <typename IntegerA, typename IntegerB>
inline ChunkedRange<std::common_type_t<IntegerA, IntegerB>>
makeChunkedRange(IntegerA start, IntegerB end, ParForChunking chunking = ParForChunking::kStatic) {
  using IntegerT = std::common_type_t<IntegerA, IntegerB>;
  return (chunking == ParForChunking::kStatic)
      ? ChunkedRange<IntegerT>(start, end, typename ChunkedRange<IntegerT>::Static())
      : ChunkedRange<IntegerT>(start, end, typename ChunkedRange<IntegerT>::Auto());
}

/**
 * Create a ChunkedRange with specific chunk size
 *
 * @param start The start of the range.
 * @param end The end of the range.
 * @param chunkSize The chunk size.
 **/
template <typename IntegerA, typename IntegerB, typename IntegerC>
inline ChunkedRange<std::common_type_t<IntegerA, IntegerB>>
makeChunkedRange(IntegerA start, IntegerB end, IntegerC chunkSize) {
  return ChunkedRange<std::common_type_t<IntegerA, IntegerB>>(start, end, chunkSize);
}

namespace detail {

template <typename TaskSetT, typename IntegerT, typename F>
void parallel_for_staticImpl(
    TaskSetT& taskSet,
    const ChunkedRange<IntegerT>& range,
    F&& f,
    ParForOptions options) {
  ssize_t numThreads = std::min<ssize_t>(taskSet.numPoolThreads(), options.maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<ssize_t>(numThreads, range.size());

  auto chunking = detail::staticChunkSize(range.size(), numThreads);
  IntegerT chunkSize = static_cast<IntegerT>(chunking.ceilChunkSize);

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  ssize_t firstLoopLen = static_cast<ssize_t>(chunking.transitionTaskIndex) - perfectlyChunked;

  IntegerT start = range.start;
  ssize_t t;
  for (t = 0; t < firstLoopLen; ++t) {
    IntegerT next = static_cast<IntegerT>(start + chunkSize);
    taskSet.schedule([start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(start, next);
    });
    start = next;
  }

  // Reduce the remaining chunk sizes by 1.
  chunkSize = static_cast<IntegerT>(chunkSize - !perfectlyChunked);
  // Finish submitting all but the last item.
  for (; t < numThreads - 1; ++t) {
    IntegerT next = static_cast<IntegerT>(start + chunkSize);
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

template <
    typename TaskSetT,
    typename IntegerT,
    typename F,
    typename StateContainer,
    typename StateGen>
void parallel_for_staticImpl(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange<IntegerT>& range,
    F&& f,
    ParForOptions options) {
  ssize_t numThreads = std::min<ssize_t>(taskSet.numPoolThreads(), options.maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<ssize_t>(numThreads, range.size());

  for (ssize_t i = 0; i < numThreads; ++i) {
    states.emplace_back(defaultState());
  }

  auto chunking = detail::staticChunkSize(range.size(), numThreads);
  IntegerT chunkSize = static_cast<IntegerT>(chunking.ceilChunkSize);

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  ssize_t firstLoopLen = chunking.transitionTaskIndex - perfectlyChunked;

  auto stateIt = states.begin();
  IntegerT start = range.start;
  ssize_t t;
  for (t = 0; t < firstLoopLen; ++t) {
    IntegerT next = static_cast<IntegerT>(start + chunkSize);
    taskSet.schedule([it = stateIt++, start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(*it, start, next);
    });
    start = next;
  }

  // Reduce the remaining chunk sizes by 1.
  chunkSize = static_cast<IntegerT>(chunkSize - !perfectlyChunked);
  // Finish submitting all but the last item.
  for (; t < numThreads - 1; ++t) {
    IntegerT next = static_cast<IntegerT>(start + chunkSize);
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
template <typename TaskSetT, typename IntegerT, typename F>
void parallel_for(
    TaskSetT& taskSet,
    const ChunkedRange<IntegerT>& range,
    F&& f,
    ParForOptions options = {}) {
  if (range.empty()) {
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }
  // TODO(bbudge): With options.maxThreads, we might want to allow a small fanout factor in
  // recursive case?
  if (!options.maxThreads || range.size() == 1 ||
      detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    f(range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  if (range.isStatic()) {
    detail::parallel_for_staticImpl(taskSet, range, std::forward<F>(f), options);
    return;
  }

  const ssize_t N = taskSet.numPoolThreads();
  const bool useCallingThread = options.wait;
  const ssize_t numToLaunch = std::min<ssize_t>(options.maxThreads, N - useCallingThread);

  if (numToLaunch == 1 && !useCallingThread) {
    taskSet.schedule([range, f = std::move(f)]() { f(range.start, range.end); });
    if (options.wait) {
      taskSet.wait();
    }
    return;
  } else if (numToLaunch == 0) {
    f(range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  const IntegerT chunk = range.calcChunkSize(numToLaunch, useCallingThread);

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<IntegerT> index(range.start);
    auto worker = [end = range.end, &index, f = std::move(f), chunk]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        IntegerT cur = index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(cur, std::min<IntegerT>(static_cast<IntegerT>(cur + chunk), end));
      }
    };

    using Function = decltype(worker);

    for (ssize_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule(Function(worker));
    }
    worker();
    taskSet.wait();
  } else {
    struct Atomic {
      Atomic(IntegerT i) : index(i) {}
      char buffer[kCacheLineSize];
      std::atomic<IntegerT> index;
      char buffer2[kCacheLineSize];
    };
    // TODO(bbudge): dispenso::make_shared?
    auto wrapper = std::make_shared<Atomic>(range.start);
    auto worker = [end = range.end, wrapper = std::move(wrapper), f, chunk]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        IntegerT cur = wrapper->index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(cur, std::min<IntegerT>(static_cast<IntegerT>(cur + chunk), end));
      }
    };

    using Function = decltype(worker);

    for (ssize_t i = 0; i < numToLaunch; ++i) {
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
template <typename IntegerT, typename F>
void parallel_for(const ChunkedRange<IntegerT>& range, F&& f, ParForOptions options = {}) {
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
template <
    typename TaskSetT,
    typename IntegerT,
    typename F,
    typename StateContainer,
    typename StateGen>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange<IntegerT>& range,
    F&& f,
    ParForOptions options = {}) {
  if (range.empty()) {
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }
  if (!options.maxThreads || range.size() == 1 ||
      detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    states.emplace_back(defaultState());
    f(*states.begin(), range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  if (range.isStatic()) {
    detail::parallel_for_staticImpl(
        taskSet, states, defaultState, range, std::forward<F>(f), options);
    return;
  }

  const ssize_t N = taskSet.numPoolThreads();
  const bool useCallingThread = options.wait;
  const ssize_t numToLaunch = std::min<ssize_t>(options.maxThreads, N - useCallingThread);

  for (ssize_t i = 0; i < numToLaunch + useCallingThread; ++i) {
    states.emplace_back(defaultState());
  }

  if (numToLaunch == 1 && !useCallingThread) {
    taskSet.schedule(
        [&s = states.front(), range, f = std::move(f)]() { f(s, range.start, range.end); });
    if (options.wait) {
      taskSet.wait();
    }
    return;
  } else if (numToLaunch == 0) {
    f(*states.begin(), range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  const IntegerT chunk = range.calcChunkSize(numToLaunch, useCallingThread);

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<IntegerT> index(range.start);
    auto worker = [end = range.end, &index, f, chunk](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();

      while (true) {
        IntegerT cur = index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(s, cur, std::min<IntegerT>(static_cast<IntegerT>(cur + chunk), end));
      }
    };

    auto it = states.begin();
    for (ssize_t i = 0; i < numToLaunch; ++i) {
      taskSet.schedule([&s = *it++, worker]() { worker(s); });
    }
    worker(*it);
    taskSet.wait();
  } else {
    struct Atomic {
      Atomic(IntegerT i) : index(i) {}
      char buffer[kCacheLineSize];
      std::atomic<IntegerT> index;
      char buffer2[kCacheLineSize];
    };
    auto wrapper = std::make_shared<Atomic>(range.start);
    auto worker = [end = range.end, wrapper = std::move(wrapper), f, chunk](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        IntegerT cur = wrapper->index.fetch_add(chunk, std::memory_order_relaxed);
        if (cur >= end) {
          break;
        }
        f(s, cur, std::min<IntegerT>(static_cast<IntegerT>(cur + chunk), end));
      }
    };

    auto it = states.begin();
    for (ssize_t i = 0; i < numToLaunch; ++i) {
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
template <typename F, typename IntegerT, typename StateContainer, typename StateGen>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    const ChunkedRange<IntegerT>& range,
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
template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
void parallel_for(
    TaskSetT& taskSet,
    IntegerA start,
    IntegerB end,
    F&& f,
    ParForOptions options = {}) {
  using IntegerT = std::common_type_t<IntegerA, IntegerB>;

  auto range = makeChunkedRange(start, end, options.defaultChunking);
  parallel_for(
      taskSet,
      range,
      [f = std::move(f)](IntegerT s, IntegerT e) {
        for (IntegerT i = s; i < e; ++i) {
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
template <
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
void parallel_for(IntegerA start, IntegerB end, F&& f, ParForOptions options = {}) {
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
template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    IntegerA start,
    IntegerB end,
    F&& f,
    ParForOptions options = {}) {
  using IntegerT = std::common_type_t<IntegerA, IntegerB>;
  auto range = makeChunkedRange(start, end, options.defaultChunking);
  parallel_for(
      taskSet,
      states,
      defaultState,
      range,
      [f = std::move(f)](auto& state, IntegerT s, IntegerT e) {
        for (IntegerT i = s; i < e; ++i) {
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
template <
    typename IntegerA,
    typename IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
void parallel_for(
    StateContainer& states,
    const StateGen& defaultState,
    IntegerA start,
    IntegerB end,
    F&& f,
    ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  parallel_for(taskSet, states, defaultState, start, end, std::forward<F>(f), options);
}

} // namespace dispenso
