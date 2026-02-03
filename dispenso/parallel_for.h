/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file parallel_for.h
 * @ingroup group_parallel
 * Functions for performing parallel for loops.
 **/

#pragma once

#include <cmath>
#include <limits>
#include <memory>

#include <dispenso/detail/can_invoke.h>
#include <dispenso/detail/per_thread_info.h>
#include <dispenso/task_set.h>

namespace dispenso {

#if DISPENSO_HAS_CONCEPTS
/**
 * @concept ParallelForRangeFunc
 * @brief A callable suitable for chunked parallel_for with (begin, end) signature.
 *
 * The callable must be invocable with two integer arguments representing the chunk range.
 **/
template <typename F, typename IntegerT>
concept ParallelForRangeFunc = std::invocable<F, IntegerT, IntegerT>;

/**
 * @concept ParallelForIndexFunc
 * @brief A callable suitable for element-wise parallel_for with single index signature.
 *
 * The callable must be invocable with a single integer argument representing the element index.
 **/
template <typename F, typename IntegerT>
concept ParallelForIndexFunc = std::invocable<F, IntegerT>;

/**
 * @concept ParallelForStateRangeFunc
 * @brief A callable suitable for stateful chunked parallel_for.
 *
 * The callable must be invocable with (State&, begin, end) arguments.
 **/
template <typename F, typename StateRef, typename IntegerT>
concept ParallelForStateRangeFunc = std::invocable<F, StateRef, IntegerT, IntegerT>;

/**
 * @concept ParallelForStateIndexFunc
 * @brief A callable suitable for stateful element-wise parallel_for.
 *
 * The callable must be invocable with (State&, index) arguments.
 **/
template <typename F, typename StateRef, typename IntegerT>
concept ParallelForStateIndexFunc = std::invocable<F, StateRef, IntegerT>;
#endif // DISPENSO_HAS_CONCEPTS

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
   * the number associated with the TaskSet's thread pool to control the degree of concurrency.
   * Setting maxThreads to zero or one will result in serial operation.
   **/
  uint32_t maxThreads = std::numeric_limits<int32_t>::max();
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

  /**
   * Specify a minimum number of items per chunk for static or auto dynamic load balancing.  Cheaper
   * workloads should have a higher number of minWorkItems.  Will be ignored if an explicit chunk
   * size is provided to ChunkedRange.
   **/
  uint32_t minItemsPerChunk = 1;

  /**
   * When set to false, and StateContainers are supplied to parallel_for, re-create container from
   * scratch each call to parallel_for.  When true, reuse existing state as much as possible (only
   * create new state if we require more than is already available in the container).
   **/
  bool reuseExistingState = false;
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
  // We need to utilize 64-bit integers to avoid overflow, e.g. passing -2**30, 2**30 as int32 will
  // result in overflow unless we cast to 64-bit.  Note that if we have a range of e.g. -2**63+1 to
  // 2**63-1, we cannot hold the result in an int64_t.  We could in a uint64_t, but it is quite
  // tricky to make this work.  However, I do not expect ranges larger than can be held in int64_t
  // since people want their computations to finish before the heat death of the sun (slight
  // exaggeration).
  using size_type = std::conditional_t<std::is_signed<IntegerT>::value, int64_t, uint64_t>;

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

  bool isAuto() const {
    return chunk == 0;
  }

  bool empty() const {
    return end <= start;
  }

  size_type size() const {
    return static_cast<size_type>(end) - start;
  }

  template <typename OtherInt>
  std::tuple<size_type, size_type>
  calcChunkSize(OtherInt numLaunched, bool oneOnCaller, size_type minChunkSize) const {
    size_type workingThreads = static_cast<size_type>(numLaunched) + size_type{oneOnCaller};
    assert(workingThreads > 0);

    if (!chunk) {
      size_type dynFactor = std::min<size_type>(16, size() / workingThreads);
      size_type chunkSize;
      do {
        size_type roughChunks = dynFactor * workingThreads;
        chunkSize = (size() + roughChunks - 1) / roughChunks;
        --dynFactor;
      } while (chunkSize < minChunkSize);
      return {chunkSize, (size() + chunkSize - 1) / chunkSize};
    } else if (chunk == kStatic) {
      // This should never be called.  The static distribution versions of the parallel_for
      // functions should be invoked instead.
      std::abort();
    }
    return {chunk, (size() + chunk - 1) / chunk};
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

struct NoOpIter {
  int& operator*() const {
    static int i = 0;
    return i;
  }
  NoOpIter& operator++() {
    return *this;
  }
  NoOpIter operator++(int) {
    return *this;
  }
};

struct NoOpContainer {
  size_t size() const {
    return 0;
  }

  bool empty() const {
    return true;
  }

  void clear() {}

  NoOpIter begin() {
    return {};
  }

  void emplace_back(int) {}

  int& front() {
    static int i;
    return i;
  }
};

struct NoOpStateGen {
  int operator()() const {
    return 0;
  }
};

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
    ssize_t maxThreads,
    bool wait,
    bool reuseExistingState) {
  using size_type = typename ChunkedRange<IntegerT>::size_type;

  size_type numThreads = std::min<size_type>(taskSet.numPoolThreads() + wait, maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min(numThreads, range.size());

  if (!reuseExistingState) {
    states.clear();
  }

  size_t numToEmplace = states.size() < static_cast<size_t>(numThreads)
      ? static_cast<size_t>(numThreads) - states.size()
      : 0;

  for (; numToEmplace--;) {
    states.emplace_back(defaultState());
  }

  auto chunking =
      detail::staticChunkSize(static_cast<ssize_t>(range.size()), static_cast<ssize_t>(numThreads));
  IntegerT chunkSize = static_cast<IntegerT>(chunking.ceilChunkSize);

  bool perfectlyChunked = static_cast<size_type>(chunking.transitionTaskIndex) == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  size_type firstLoopLen = chunking.transitionTaskIndex - perfectlyChunked;

  auto stateIt = states.begin();
  IntegerT start = range.start;
  size_type t;
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

  if (wait) {
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

  using size_type = typename ChunkedRange<IntegerT>::size_type;

  // Ensure minItemsPerChunk is sane
  uint32_t minItemsPerChunk = std::max<uint32_t>(1, options.minItemsPerChunk);

  // 0 indicates serial execution per API spec
  size_type maxThreads = std::max<int32_t>(options.maxThreads, 1);

  bool isStatic = range.isStatic();

  const size_type N = taskSet.numPoolThreads();
  if (N == 0 || !options.maxThreads || range.size() <= minItemsPerChunk ||
      detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    if (!options.reuseExistingState) {
      states.clear();
    }
    if (states.empty()) {
      states.emplace_back(defaultState());
    }
    f(*states.begin(), range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  // Adjust down workers if we would have too-small chunks
  if (minItemsPerChunk > 1) {
    size_type maxWorkers = range.size() / minItemsPerChunk;
    if (maxWorkers < maxThreads) {
      maxThreads = static_cast<uint32_t>(maxWorkers);
    }
    if (range.size() / (maxThreads + options.wait) < minItemsPerChunk && range.isAuto()) {
      isStatic = true;
    }
  } else if (range.size() <= N + options.wait) {
    if (range.isAuto()) {
      isStatic = true;
    } else if (!range.isStatic()) {
      maxThreads = range.size() - options.wait;
    }
  }

  if (isStatic) {
    detail::parallel_for_staticImpl(
        taskSet,
        states,
        defaultState,
        range,
        std::forward<F>(f),
        static_cast<ssize_t>(maxThreads),
        options.wait,
        options.reuseExistingState);
    return;
  }

  // wanting maxThreads workers (potentially including the calling thread), capped by N
  const size_type numToLaunch = std::min<size_type>(maxThreads - options.wait, N);

  if (!options.reuseExistingState) {
    states.clear();
  }

  size_t numToEmplace = static_cast<size_type>(states.size()) < (numToLaunch + options.wait)
      ? (static_cast<size_t>(numToLaunch) + options.wait) - states.size()
      : 0;
  for (; numToEmplace--;) {
    states.emplace_back(defaultState());
  }

  if (numToLaunch == 1 && !options.wait) {
    taskSet.schedule(
        [&s = states.front(), range, f = std::move(f)]() { f(s, range.start, range.end); });

    return;
  }

  auto chunkInfo = range.calcChunkSize(numToLaunch, options.wait, minItemsPerChunk);
  auto chunkSize = std::get<0>(chunkInfo);
  auto numChunks = std::get<1>(chunkInfo);

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<decltype(numChunks)> index(0);
    auto worker = [start = range.start, end = range.end, &index, f, chunkSize, numChunks](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();

      while (true) {
        auto cur = index.fetch_add(1, std::memory_order_relaxed);
        if (cur >= numChunks) {
          break;
        }
        auto sidx = static_cast<IntegerT>(start + cur * chunkSize);
        if (cur + 1 == numChunks) {
          f(s, sidx, end);
        } else {
          auto eidx = static_cast<IntegerT>(sidx + chunkSize);
          f(s, sidx, eidx);
        }
      }
    };

    auto it = states.begin();
    for (size_type i = 0; i < numToLaunch; ++i) {
      taskSet.schedule([&s = *it++, worker]() { worker(s); });
    }
    worker(*it);
    taskSet.wait();
  } else {
    struct Atomic {
      Atomic() : index(0) {}
      alignas(kCacheLineSize) std::atomic<decltype(numChunks)> index;
      char buffer[kCacheLineSize - sizeof(index)];
    };

    void* ptr = detail::alignedMalloc(sizeof(Atomic), alignof(Atomic));
    auto* atm = new (ptr) Atomic();

    std::shared_ptr<Atomic> wrapper(atm, detail::AlignedFreeDeleter<Atomic>());
    auto worker = [start = range.start,
                   end = range.end,
                   wrapper = std::move(wrapper),
                   f,
                   chunkSize,
                   numChunks](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        auto cur = wrapper->index.fetch_add(1, std::memory_order_relaxed);
        if (cur >= numChunks) {
          break;
        }
        auto sidx = static_cast<IntegerT>(start + cur * chunkSize);
        if (cur + 1 == numChunks) {
          f(s, sidx, end);
        } else {
          auto eidx = static_cast<IntegerT>(sidx + chunkSize);
          f(s, sidx, eidx);
        }
      }
    };

    auto it = states.begin();
    for (size_type i = 0; i < numToLaunch; ++i) {
      taskSet.schedule([&s = *it++, worker]() { worker(s); }, ForceQueuingTag());
    }
  }
}

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
DISPENSO_REQUIRES(ParallelForRangeFunc<F, IntegerT>)
void parallel_for(
    TaskSetT& taskSet,
    const ChunkedRange<IntegerT>& range,
    F&& f,
    ParForOptions options = {}) {
  detail::NoOpContainer container;
  parallel_for(
      taskSet,
      container,
      detail::NoOpStateGen(),
      range,
      [f = std::move(f)](int /*noop*/, auto i, auto j) { f(i, j); },
      options);
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
DISPENSO_REQUIRES(ParallelForRangeFunc<F, IntegerT>)
void parallel_for(const ChunkedRange<IntegerT>& range, F&& f, ParForOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  parallel_for(taskSet, range, std::forward<F>(f), options);
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
 * <code>void(size_t index)</code> or <code>void(size_t begin, size_t end)</code>.
 * @param options See ParForOptions for details.
 **/
template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<detail::CanInvoke<F(IntegerA)>::value, bool> = true>
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

template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<detail::CanInvoke<F(IntegerA, IntegerB)>::value, bool> = true>
void parallel_for(
    TaskSetT& taskSet,
    IntegerA start,
    IntegerB end,
    F&& f,
    ParForOptions options = {}) {
  auto range = makeChunkedRange(start, end, options.defaultChunking);
  parallel_for(taskSet, range, std::forward<F>(f), options);
}

/**
 * Execute loop over the range in parallel on the global thread pool and block on loop completion.
 *
 * @param start The start of the loop extents.
 * @param end The end of the loop extents.
 * @param f The functor to execute in parallel.  Must have a signature like
 * <code>void(size_t index)</code> or <code>void(size_t begin, size_t end)</code>.
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
 * <code>void(State &s, size_t index)</code> or
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
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<
        detail::CanInvoke<F(typename StateContainer::reference, IntegerA)>::value,
        bool> = true>
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

template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<
        detail::CanInvoke<F(typename StateContainer::reference, IntegerA, IntegerB)>::value,
        bool> = true>
void parallel_for(
    TaskSetT& taskSet,
    StateContainer& states,
    const StateGen& defaultState,
    IntegerA start,
    IntegerB end,
    F&& f,
    ParForOptions options = {}) {
  auto range = makeChunkedRange(start, end, options.defaultChunking);
  parallel_for(taskSet, states, defaultState, range, std::forward<F>(f), options);
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
 * <code>void(State &s, size_t index)</code> or
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
