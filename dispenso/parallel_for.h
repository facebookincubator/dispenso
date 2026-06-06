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

#include <dispenso/cpu_set.h>
#include <dispenso/detail/can_invoke.h>
#include <dispenso/detail/par_for_stripe.h>
#include <dispenso/detail/per_thread_info.h>
#include <dispenso/small_buffer_allocator.h>
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
 * much cheaper than others, select kAdaptive (or its alias kAuto).
 *
 * kAdaptive partitions the iteration space into P contiguous stripes (one per worker), each
 * consumed front-to-back by its owner via fetch_add on a per-stripe atomic cursor. When a worker's
 * own stripe is exhausted, it claims chunks from peers' stripes the same way, preferring same-L3
 * cache-group victims first to keep stolen data warm in the shared L3 (CCD on AMD, tile/SNC on
 * Intel). Chunk size is fixed per call (auto-derived from range size / worker count).
 *
 * kAuto is kept as a compatibility alias for kAdaptive. New code should prefer kAdaptive; kAuto
 * will be deprecated in a future release and removed in 2.0.
 **/
enum class ParForChunking {
  kStatic,
  kAdaptive,
  kAuto DISPENSO_DEPRECATED("Use ParForChunking::kAdaptive (kAuto will be removed in 2.0).") =
      kAdaptive,
};

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
   *
   * For kStatic, this is a hard floor on chunk size (only the last chunk may be smaller). For
   * kAdaptive, it is a SOFT HINT: the owner's pop step and the stealer's split point both respect
   * minItemsPerChunk, but after a steal the victim may be left with a sub-minItemsPerChunk
   * remainder. The remainder is still processed by the original owner, so up to <i>numWorkers</i>
   * sub-minItemsPerChunk chunks may appear in adversarial steal patterns. If you need a hard
   * granularity guarantee, use the <code>granularity</code> option or pass an explicit chunk size
   * to <code>makeChunkedRange</code>.
   **/
  uint32_t minItemsPerChunk = 1;

  /**
   * Specify a chunk-size granularity contract.  When > 1, every chunk passed to the user's lambda
   * by parallel work is guaranteed to have <code>(end - begin)</code> be a multiple of
   * <code>granularity</code>.  If the total range size is not a multiple of
   * <code>granularity</code>, the sub-granularity remainder (the "tail") will be executed serially
   * on the calling thread after the parallel portion completes; that single tail invocation is the
   * only call whose <code>(end - begin)</code> may not be a multiple of <code>granularity</code>.
   *
   * Useful for SIMD inner loops (e.g. <code>granularity = 8</code> for AVX-256 doubles), block
   * algorithms, or any inner loop with a fixed unroll factor where partial blocks add overhead.
   *
   * <code>granularity</code> is a contract on chunk boundaries, not on chunk size: chunks may still
   * vary widely (e.g. 8, 16, 24, 8000) — they just won't be 9 or 17 or 8001.
   *
   * Ignored (treated as 1) when an explicit chunk size is provided to ChunkedRange; in that case
   * the user is already specifying exact chunk granularity.
   **/
  uint32_t granularity = 1;

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
  std::tuple<size_type, size_type> calcChunkSize(
      OtherInt numLaunched,
      bool oneOnCaller,
      size_type minChunkSize,
      uint32_t granularity = 1,
      size_type maxDynFactor = 16) const {
    size_type workingThreads = static_cast<size_type>(numLaunched) + size_type{oneOnCaller};
    assert(workingThreads > 0);

    if (!chunk) {
      size_type dynFactor = std::min<size_type>(maxDynFactor, size() / workingThreads);
      size_type chunkSize;
      do {
        size_type roughChunks = dynFactor * workingThreads;
        chunkSize = (size() + roughChunks - 1) / roughChunks;
        if (granularity > 1) {
          // Round UP to a multiple of granularity (no smaller than granularity).
          chunkSize = ((chunkSize + granularity - 1) / granularity) * granularity;
        }
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
  using difference_type = std::ptrdiff_t;
  using value_type = int;
  using pointer = int*;
  using reference = int&;
  using iterator_category = std::random_access_iterator_tag;

  int& operator*() const {
    static DISPENSO_THREAD_LOCAL int dummy = 0;
    return dummy;
  }
  NoOpIter& operator++() {
    return *this;
  }
  NoOpIter operator++(int) {
    return *this;
  }
  NoOpIter& operator--() {
    return *this;
  }
  NoOpIter operator--(int) {
    return *this;
  }
  NoOpIter& operator+=(difference_type) {
    return *this;
  }
  NoOpIter& operator-=(difference_type) {
    return *this;
  }
  NoOpIter operator+(difference_type) const {
    return *this;
  }
  NoOpIter operator-(difference_type) const {
    return *this;
  }
  difference_type operator-(const NoOpIter&) const {
    return 0;
  }
  bool operator==(const NoOpIter&) const {
    return true;
  }
  bool operator!=(const NoOpIter&) const {
    return false;
  }
  bool operator<(const NoOpIter&) const {
    return false;
  }
  int& operator[](difference_type) const {
    static DISPENSO_THREAD_LOCAL int dummy = 0;
    return dummy;
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

// Round size DOWN to a multiple of granularity. Granularity must be >= 1.
template <typename IntegerT>
inline IntegerT roundDownToGranularity(IntegerT size, uint32_t granularity) {
  if (granularity <= 1) {
    return size;
  }
  using U = typename std::make_unsigned<IntegerT>::type;
  return static_cast<IntegerT>((static_cast<U>(size) / granularity) * granularity);
}

// Round size UP to a multiple of granularity. Granularity must be >= 1.
template <typename IntegerT>
inline IntegerT roundUpToGranularity(IntegerT size, uint32_t granularity) {
  if (granularity <= 1) {
    return size;
  }
  using U = typename std::make_unsigned<IntegerT>::type;
  return static_cast<IntegerT>(
      ((static_cast<U>(size) + granularity - 1) / granularity) * granularity);
}

/**
 * Initialize states container with enough entries for the given thread count.
 * Respects reuseExistingState: when true, only adds entries if the container
 * doesn't already have enough.
 */
template <typename StateContainer, typename StateGen>
void initStates(
    StateContainer& states,
    const StateGen& defaultState,
    size_t numNeeded,
    bool reuseExistingState) {
  if (!reuseExistingState) {
    states.clear();
  }
  for (size_t i = states.size(); i < numNeeded; ++i) {
    states.emplace_back(defaultState());
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
    ssize_t maxThreads,
    bool wait,
    bool reuseExistingState,
    uint32_t granularity = 1) {
  using size_type = typename ChunkedRange<IntegerT>::size_type;

  size_type numThreads = std::min<size_type>(taskSet.numPoolThreads() + 1, maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min(numThreads, range.size());
  // With granularity > 1 we cannot give a thread fewer than `granularity` items,
  // so reduce thread count to ensure each thread gets at least one granularity unit.
  if (granularity > 1) {
    size_type maxByGranularity = range.size() / static_cast<size_type>(granularity);
    if (maxByGranularity < numThreads) {
      numThreads = std::max<size_type>(1, maxByGranularity);
    }
  }

  detail::initStates(states, defaultState, static_cast<size_t>(numThreads), reuseExistingState);

  auto chunking = (granularity > 1)
      ? detail::staticChunkSizeGranular(
            static_cast<ssize_t>(range.size()), static_cast<ssize_t>(numThreads), granularity)
      : detail::staticChunkSize(
            static_cast<ssize_t>(range.size()), static_cast<ssize_t>(numThreads));
  IntegerT chunkSize = static_cast<IntegerT>(chunking.ceilChunkSize);

  bool perfectlyChunked = static_cast<size_type>(chunking.transitionTaskIndex) == numThreads;

  // Helper: compute chunk [start, end) for a given task index.
  auto chunkRange = [&](size_type idx) -> std::pair<IntegerT, IntegerT> {
    size_type transIdx = perfectlyChunked ? numThreads : chunking.transitionTaskIndex;
    // Without granularity, floor-chunks are `ceil - 1`. With granularity > 1,
    // floor-chunks are `ceil - granularity` (one fewer granularity unit).
    IntegerT chunkStep = granularity > 1 ? static_cast<IntegerT>(granularity) : IntegerT{1};
    IntegerT smallChunk =
        static_cast<IntegerT>(chunkSize - (perfectlyChunked ? IntegerT{0} : chunkStep));
    IntegerT start;
    if (idx < transIdx) {
      IntegerT i = static_cast<IntegerT>(idx);
      start = static_cast<IntegerT>(range.start + static_cast<IntegerT>(i * chunkSize));
    } else {
      IntegerT ti = static_cast<IntegerT>(transIdx);
      IntegerT ri = static_cast<IntegerT>(idx - transIdx);
      start = static_cast<IntegerT>(
          range.start + static_cast<IntegerT>(ti * chunkSize) +
          static_cast<IntegerT>(ri * smallChunk));
    }
    IntegerT end;
    if (idx + 1 == numThreads) {
      end = range.end;
    } else if (idx < transIdx) {
      end = static_cast<IntegerT>(start + chunkSize);
    } else {
      end = static_cast<IntegerT>(start + smallChunk);
    }
    return {start, end};
  };

  // Determine which chunk the calling thread should take for L2 locality.
  // If the caller is a pool thread with a ring, it takes the chunk matching
  // its ring index so that repeated parallel_for calls keep the same data
  // in the caller's L2 cache. Otherwise takes the last chunk.
  int32_t callerRing = wait ? detail::PerPoolPerThreadInfo::ringIndex(&taskSet.pool()) : -1;
  size_type callerChunk = numThreads - 1;
  if (callerRing >= 0 && static_cast<size_type>(callerRing) < numThreads) {
    callerChunk = static_cast<size_type>(callerRing);
  }

  // Schedule N-1 chunks to workers (skipping callerChunk when wait=true).
  // The generator remaps scheduler index to chunk index: indices below
  // callerChunk map 1:1, indices >= callerChunk shift up by 1.
  // This ensures ring i gets the data for chunk i (or chunk i+1 if i >= callerChunk),
  // maintaining L2 locality for each worker thread.
  size_type numToSchedule = wait ? numThreads - 1 : numThreads;

  if (numToSchedule > 0) {
    taskSet.scheduleBulk(static_cast<size_t>(numToSchedule), [&, callerChunk](size_t idx) {
      size_type chunkIdx = static_cast<size_type>(idx);
      if (wait && chunkIdx >= callerChunk) {
        ++chunkIdx;
      }

      auto chunkBounds = chunkRange(chunkIdx);
      IntegerT start = chunkBounds.first;
      IntegerT end = chunkBounds.second;

      auto stateIt = states.begin();
      std::advance(stateIt, static_cast<ptrdiff_t>(chunkIdx));

      return [it = stateIt, start, end, f]() {
        auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
        f(*it, start, end);
      };
    });
  }

  if (wait) {
    auto stateIt = states.begin();
    std::advance(stateIt, static_cast<ptrdiff_t>(callerChunk));
    auto callerBounds = chunkRange(callerChunk);
    {
      // Mark caller as inside parallel_for so nested parallel_for calls
      // skip the ring fast path (same as worker tasks which have this guard
      // in their lambda). Without this, the caller's nested parallel_for
      // would push to rings that workers can't drain while they're busy
      // with their own outer chunks.
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(*stateIt, callerBounds.first, callerBounds.second);
    }
    taskSet.wait();
  }
}

template <typename IntegerT>
struct ChunkSizingResult {
  typename ChunkedRange<IntegerT>::size_type maxThreads;
  bool isStatic;
};

/**
 * Adjust the thread count and static/dynamic scheduling decision based on work size.
 *
 * The goal is to avoid oversubscription and ensure each thread gets enough work:
 *
 * 1. Cap maxThreads to available pool threads (plus calling thread if waiting).
 * 2. If minItemsPerChunk > 1, reduce thread count so each thread gets at least
 *    that many items. If even after reduction the chunks are too small, fall back
 *    to static scheduling (which distributes items evenly without atomic contention).
 * 3. If minItemsPerChunk == 1 and the range is smaller than the thread count,
 *    fall back to static scheduling (auto mode) or cap threads (explicit dynamic).
 */
template <typename IntegerT>
ChunkSizingResult<IntegerT> adjustChunkSizing(
    const ChunkedRange<IntegerT>& range,
    typename ChunkedRange<IntegerT>::size_type maxThreads,
    bool isStatic,
    uint32_t minItemsPerChunk,
    typename ChunkedRange<IntegerT>::size_type poolThreads,
    bool wait) {
  using size_type = typename ChunkedRange<IntegerT>::size_type;

  // Step 1: never use more threads than could participate.
  // Always count the caller (+1) even when wait=false, because the caller
  // will eventually participate via TaskSet::wait()/destructor and can steal
  // work items. This prevents noWait parallel_for on small pools from
  // degenerating to serial inline execution.
  maxThreads = std::min<size_type>(maxThreads, poolThreads + 1);

  if (minItemsPerChunk > 1) {
    // Step 2a: reduce threads so each gets at least minItemsPerChunk items
    size_type maxWorkers = range.size() / minItemsPerChunk;
    if (maxWorkers < maxThreads) {
      maxThreads = maxWorkers;
    }
    // Step 2b: if dynamic chunks would still be too small, use static scheduling
    if (maxThreads > 0 && range.size() / (maxThreads + wait) < minItemsPerChunk && range.isAuto()) {
      isStatic = true;
    }
  } else if (range.size() <= poolThreads + wait) {
    // Step 3: fewer items than threads — static is better (no atomic overhead)
    if (range.isAuto()) {
      isStatic = true;
    } else if (!range.isStatic()) {
      maxThreads = range.size() - wait;
    }
  }

  return {maxThreads, isStatic};
}

// Per-group range for decoupled-atomic dynamic parallel_for. Each group of
// threads shares an L3-local atomic index over a contiguous sub-range of
// chunks, eliminating cross-CCD cache-line bouncing on many-core machines.
struct alignas(kCacheLineSize) GroupRange {
  std::atomic<size_t> index{0};
  size_t startChunk{0};
  size_t numGroupChunks{0};
};

/**
 * Dynamic (work-stealing) parallel_for implementation.
 *
 * Workers are partitioned into groups of ~16 threads, each with its own
 * atomic index over a contiguous sub-range of chunks. This keeps each
 * group's atomic in the local L3 cache on many-core machines with
 * multiple CCDs, eliminating cross-CCD cache-line contention on the
 * shared fetch_add. Intra-group load balancing (16 threads sharing one
 * L3-local atomic) is fast and preserved.
 *
 * When the total worker count is below 16, there is exactly one group
 * and the behavior is equivalent to the original single-atomic approach.
 *
 * The ExitAction callback is invoked when a worker finds no more chunks;
 * this allows the no-wait path to deallocate heap state when the last
 * worker exits, while the wait path passes a no-op.
 *
 * When wait is true, the calling thread participates as an additional worker
 * and blocks until all tasks complete.
 */
template <
    typename TaskSetT,
    typename IntegerT,
    typename F,
    typename StateContainer,
    typename IndexRef,
    typename ExitAction>
void parallel_for_dynamicImpl(
    TaskSetT& taskSet,
    StateContainer& states,
    IntegerT start,
    IntegerT end,
    F&& f,
    size_t numToLaunch,
    typename ChunkedRange<IntegerT>::size_type chunkSize,
    typename ChunkedRange<IntegerT>::size_type numChunks,
    IndexRef& index,
    ExitAction exitAction,
    bool wait) {
  size_t totalWorkers = numToLaunch + (wait ? 1 : 0);

  // Use the L3 cache group count (one per CCD on AMD, one per tile/SNC
  // cluster on Intel) when available, so the per-group atomic lands in each
  // CCD's L3. Falls back to a heuristic (~16 threads per group) on platforms
  // where sysfs/topology detection returns nothing (e.g. macOS).
  size_t l3Groups = CpuSet::l3CacheGroups().size();
  size_t effectiveGroups;
  if (l3Groups > 1 && totalWorkers > 16) {
    effectiveGroups = std::min(l3Groups, totalWorkers);
  } else {
    effectiveGroups = std::max<size_t>(1, (totalWorkers + 15) / 16);
  }

  // Single group: use the original shared-index path with no extra overhead.
  if (effectiveGroups <= 1) {
    auto worker = [start, end, &index, f, chunkSize, numChunks, exitAction](auto& s) {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      while (true) {
        auto cur = index.fetch_add(1, std::memory_order_relaxed);
        if (cur >= numChunks) {
          exitAction(cur);
          break;
        }
        auto sidx = static_cast<IntegerT>(start + cur * chunkSize);
        if (cur + 1 == numChunks) {
          f(s, sidx, end);
        } else {
          f(s, sidx, static_cast<IntegerT>(sidx + chunkSize));
        }
      }
    };

    {
      auto stateBegin = states.begin();
      // Use scheduleBulk so the per-worker tasks land directly in per-thread
      // rings (kStatic-style fast path), not the central moodycamel queue.
      // Each task is the worker loop itself; the shared atomic index inside
      // the worker handles dynamic chunk distribution.
      taskSet.scheduleBulk(static_cast<size_t>(numToLaunch), [stateBegin, worker](size_t i) {
        auto stateIt = stateBegin;
        std::advance(stateIt, static_cast<ptrdiff_t>(i));
        return [&s = *stateIt, worker]() { worker(s); };
      });
    }

    if (wait) {
      auto it = states.begin();
      std::advance(it, static_cast<ptrdiff_t>(numToLaunch));
      worker(*it);
      taskSet.wait();
    }
    return;
  }

  // Multi-group path: partition chunks among effectiveGroups groups, each with
  // its own cache-line-aligned atomic index.
  //
  // The GroupRange array and exit counter must outlive all workers. In the
  // wait path the caller blocks until completion, so stack ownership suffices.
  // In the no-wait path the function returns immediately, so we heap-allocate
  // and have the last worker free the memory alongside the caller's exitAction.

  // Header stored at the front of a single aligned allocation that holds the
  // exit counter followed by the GroupRange array.
  struct alignas(kCacheLineSize) GroupBlock {
    std::atomic<size_t> exitCounter{0};
    size_t numGroups;
    size_t totalWorkers;
    bool heapOwned;

    GroupRange* ranges() {
      auto* base = reinterpret_cast<char*>(this);
      uintptr_t off = detail::alignToCacheLine(sizeof(GroupBlock));
      return reinterpret_cast<GroupRange*>(base + off);
    }
  };

  size_t blockBytes =
      detail::alignToCacheLine(sizeof(GroupBlock)) + sizeof(GroupRange) * effectiveGroups;

  // Both the wait and no-wait paths heap-own the block; the last worker to
  // finish frees it. A thread-local buffer-reuse optimization for the wait
  // path was intentionally not used here: it is unsafe under nested multi-group
  // parallel_for on the same thread, because the caller runs its own chunk
  // (worker(*it) below) -- which may re-enter here -- BEFORE taskSet.wait()
  // returns, while this call's other-thread workers are still reading the
  // block. The per-call allocation is negligible relative to the >16-thread
  // work on this path; revisit only if it shows up in a profile.
  void* blockMem = detail::alignedMalloc(blockBytes, kCacheLineSize);
  bool heapOwned = true;

  auto* block = new (blockMem) GroupBlock{};
  block->numGroups = effectiveGroups;
  block->totalWorkers = totalWorkers;
  block->heapOwned = heapOwned;

  GroupRange* groupRanges = block->ranges();
  for (size_t g = 0; g < effectiveGroups; ++g) {
    new (&groupRanges[g]) GroupRange{};
  }

  size_t baseChunks = static_cast<size_t>(numChunks) / effectiveGroups;
  size_t extraChunks = static_cast<size_t>(numChunks) % effectiveGroups;
  size_t chunkOffset = 0;
  for (size_t g = 0; g < effectiveGroups; ++g) {
    size_t gc = baseChunks + (g < extraChunks ? 1 : 0);
    groupRanges[g].startChunk = chunkOffset;
    groupRanges[g].numGroupChunks = gc;
    chunkOffset += gc;
  }

  auto worker = [start, end, block, f, chunkSize, numChunks, exitAction](auto& s, size_t groupIdx) {
    auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
    auto& gr = block->ranges()[groupIdx];
    // Snapshot block-owned metadata BEFORE the release increment below. Once a
    // worker performs exitCounter.fetch_add, the worker that observes the final
    // count frees `block`, so `block` must not be dereferenced afterwards.
    // Reading block->totalWorkers in the `prev + 1 == ...` check after the
    // increment is a use-after-free race against that free (caught by TSAN).
    const size_t totalWorkersLocal = block->totalWorkers;
    const bool owned = block->heapOwned;
    while (true) {
      auto cur = gr.index.fetch_add(1, std::memory_order_relaxed);
      if (cur >= gr.numGroupChunks) {
        break;
      }
      auto globalChunk =
          static_cast<typename ChunkedRange<IntegerT>::size_type>(gr.startChunk + cur);
      auto sidx = static_cast<IntegerT>(start + globalChunk * chunkSize);
      if (globalChunk + 1 == numChunks) {
        f(s, sidx, end);
      } else {
        f(s, sidx, static_cast<IntegerT>(sidx + chunkSize));
      }
    }
    // The acq_rel increment orders every worker's prior block accesses (the gr
    // loop above) before the last worker's free. After it, only the last worker
    // touches block, and only to free it -- no peer reads block past this point.
    auto prev = block->exitCounter.fetch_add(1, std::memory_order_acq_rel);
    if (prev + 1 == totalWorkersLocal) {
      exitAction(numChunks + static_cast<decltype(numChunks)>(totalWorkersLocal) - 1);
      if (owned) {
        detail::alignedFree(block);
      }
    }
  };

  {
    auto stateBegin = states.begin();
    // Use scheduleBulk so the per-worker tasks land directly in per-thread
    // rings (kStatic-style fast path), not the central moodycamel queue.
    // The shared per-group atomic index inside the worker handles dynamic
    // chunk distribution across workers within a group.
    taskSet.scheduleBulk(
        numToLaunch, [stateBegin, worker, effectiveGroups, totalWorkers](size_t i) {
          auto stateIt = stateBegin;
          std::advance(stateIt, static_cast<ptrdiff_t>(i));
          size_t gIdx = (i * effectiveGroups) / totalWorkers;
          return [&s = *stateIt, worker, gIdx]() { worker(s, gIdx); };
        });
  }

  if (wait) {
    auto it = states.begin();
    std::advance(it, static_cast<ptrdiff_t>(numToLaunch));
    size_t gIdx = (numToLaunch * effectiveGroups) / totalWorkers;
    if (gIdx >= effectiveGroups) {
      gIdx = effectiveGroups - 1;
    }
    worker(*it, gIdx);
    taskSet.wait();
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
 * remain a valid object until work is completed.  When <code>options.wait</code> is false, "until
 * work is completed" extends beyond the return of this function — the caller must ensure
 * <code>states</code> outlives all scheduled work (e.g. by calling <code>taskSet.wait()</code>
 * before destroying it).
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

  // Granularity is ignored when the user provides an explicit chunk size — the user is
  // already specifying chunk granularity in that case.
  uint32_t granularity = (range.chunk == 0 || range.chunk == ChunkedRange<IntegerT>::kStatic)
      ? std::max<uint32_t>(1, options.granularity)
      : 1;

  // If granularity > 1, we may need to peel off a sub-granularity tail and run it serially
  // after the parallel portion. Compute the trimmed range and the tail boundary.
  IntegerT trimmedEnd = range.end;
  bool hasTail = false;
  if (granularity > 1) {
    size_type rem = range.size() % granularity;
    if (rem > 0) {
      trimmedEnd = static_cast<IntegerT>(range.end - static_cast<IntegerT>(rem));
      hasTail = true;
    }
  }

  // Helper to run the tail (if any) on the calling thread, after the parallel portion completes.
  auto runTail = [&]() {
    if (hasTail) {
      f(*states.begin(), trimmedEnd, range.end);
    }
  };

  uint32_t minItemsPerChunk = std::max<uint32_t>(1, options.minItemsPerChunk);
  size_type maxThreads = std::max<int32_t>(options.maxThreads, 1);
  bool isStatic = range.isStatic();

  // Build the trimmed range used by the parallel portion. Preserve chunking choice.
  ChunkedRange<IntegerT> parRange = range;
  parRange.end = trimmedEnd;

  const size_type N = taskSet.numPoolThreads();
  if (N == 0 || !options.maxThreads || parRange.size() <= minItemsPerChunk ||
      detail::PerPoolPerThreadInfo::isParForRecursive(&taskSet.pool())) {
    detail::initStates(states, defaultState, 1, options.reuseExistingState);
    // Inline single-call path: hand the full original range to the lambda. The
    // tail (if any) is absorbed into this single call — and that's fine because
    // there are no other parallel chunks to violate the granularity contract.
    f(*states.begin(), range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  auto chunkSizing =
      detail::adjustChunkSizing(parRange, maxThreads, isStatic, minItemsPerChunk, N, options.wait);
  maxThreads = chunkSizing.maxThreads;
  isStatic = chunkSizing.isStatic;

  // If adjustment reduced threads below 2, run inline — not worth parallelizing.
  if (maxThreads < 2) {
    detail::initStates(states, defaultState, 1, options.reuseExistingState);
    f(*states.begin(), range.start, range.end);
    if (options.wait) {
      taskSet.wait();
    }
    return;
  }

  if (isStatic) {
    detail::parallel_for_staticImpl(
        taskSet,
        states,
        defaultState,
        parRange,
        std::forward<F>(f),
        static_cast<ssize_t>(maxThreads),
        options.wait,
        options.reuseExistingState,
        granularity);
    runTail();
    return;
  }

  const size_type numToLaunch = std::min<size_type>(maxThreads - options.wait, N);

  detail::initStates(
      states,
      defaultState,
      static_cast<size_t>(numToLaunch + options.wait),
      options.reuseExistingState);

  // For kAdaptive (range.chunk == 0), use the per-stripe atomic-counter
  // implementation (Callisto-inspired). Each worker owns a contiguous stripe
  // and consumes chunkSize iterations at a time via fetch_add; when its own
  // stripe is exhausted, it claims chunks from peers' stripes the same way,
  // preferring same-L3 cache-group victims first.
  bool useAdaptive = range.chunk == 0;

  if (numToLaunch == 1 && !options.wait && !useAdaptive) {
    // No-wait path with a single non-adaptive worker: hand the trimmed range to
    // the worker and (if there's a tail) run the tail synchronously on the
    // caller before returning. kAdaptive falls through so the caller can later
    // participate via taskSet.wait().
    taskSet.schedule([&s = states.front(), parRange, f = std::move(f)]() {
      f(s, parRange.start, parRange.end);
    });
    runTail();
    return;
  }

  auto chunkInfo = parRange.calcChunkSize(numToLaunch, options.wait, minItemsPerChunk, granularity);
  auto chunkSize = std::get<0>(chunkInfo);
  auto numChunks = std::get<1>(chunkInfo);
  if (useAdaptive) {
    // Adaptive stripe path: aim for ~64 chunks per worker so peers always
    // have meaningful sub-ranges to steal even after an owner drains most
    // of its own stripe.
    auto adaptiveChunkInfo = parRange.calcChunkSize(
        numToLaunch, options.wait, minItemsPerChunk, granularity, /*maxDynFactor=*/64);
    auto adaptiveChunkSize = std::get<0>(adaptiveChunkInfo);
    if (options.wait) {
      // Wait path: caller participates → numWorkers = numToLaunch + 1.
      size_type numStripeWorkers = numToLaunch + 1;
      detail::StripeState<IntegerT> stripeState;
      detail::initStripeState(
          stripeState,
          parRange.start,
          parRange.end,
          static_cast<uint32_t>(numStripeWorkers),
          static_cast<IntegerT>(adaptiveChunkSize),
          granularity);
      auto stateBegin = states.begin();
      auto worker = [&stripeState, &f](auto& userState, uint32_t myIdx) {
        auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
        detail::runStripeWorker(stripeState, myIdx, userState, f);
      };
      if (numToLaunch > 0) {
        taskSet.scheduleBulk(static_cast<size_t>(numToLaunch), [stateBegin, worker](size_t idx) {
          auto stateIt = stateBegin;
          std::advance(stateIt, static_cast<ptrdiff_t>(idx));
          uint32_t myIdx = static_cast<uint32_t>(idx);
          return [&userState = *stateIt, myIdx, worker]() { worker(userState, myIdx); };
        });
      }
      auto callerIt = states.begin();
      std::advance(callerIt, static_cast<ptrdiff_t>(numToLaunch));
      worker(*callerIt, static_cast<uint32_t>(numToLaunch));
      taskSet.wait();
      runTail();
      return;
    }
    // No-wait adaptive path is not yet implemented for the stripe variant;
    // fall through to the dynamic-chunked path below. A heap-allocated
    // StripeState with shared_ptr lifetime would be the natural extension.
  }

  if (options.wait) {
    alignas(kCacheLineSize) std::atomic<decltype(numChunks)> index(0);
    detail::parallel_for_dynamicImpl(
        taskSet,
        states,
        parRange.start,
        parRange.end,
        std::forward<F>(f),
        static_cast<size_t>(numToLaunch),
        chunkSize,
        numChunks,
        index,
        [](auto) {},
        options.wait);
    runTail();
  } else {
    using SizeType = decltype(numChunks);
    struct ChunkIndex {
      std::atomic<SizeType> index;
    };
    static_assert(sizeof(ChunkIndex) <= kCacheLineSize, "ChunkIndex must fit in one cache line");
    char* mem = allocSmallBuffer<kCacheLineSize>();
    auto* ci = new (mem) ChunkIndex{{0}};
    SizeType lastExit = numChunks + static_cast<SizeType>(numToLaunch) - 1;
    // Capture the tail as part of the last-worker exit action so the contract holds
    // even when there is no external wait.
    IntegerT tailStart = trimmedEnd;
    IntegerT tailEnd = range.end;
    bool tailNeeded = hasTail;
    auto& tailState = *states.begin();
    // Copy f before std::forward may move it — the exitAction outlives this scope
    // when wait=false.
    auto tailFunc = f;
    detail::parallel_for_dynamicImpl(
        taskSet,
        states,
        parRange.start,
        parRange.end,
        std::forward<F>(f),
        static_cast<size_t>(numToLaunch),
        chunkSize,
        numChunks,
        ci->index,
        [ci, lastExit, tailFunc = std::move(tailFunc), &tailState, tailStart, tailEnd, tailNeeded](
            auto cur) {
          if (cur == lastExit) {
            if (tailNeeded) {
              tailFunc(tailState, tailStart, tailEnd);
            }
            deallocSmallBuffer<kCacheLineSize>(ci);
          }
        },
        options.wait);
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
#if DISPENSO_HAS_CONCEPTS
template <typename TaskSetT, std::integral IntegerA, std::integral IntegerB, typename F>
  requires std::invocable<F, IntegerA>
#else
template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<detail::CanInvoke<F(IntegerA)>::value, bool> = true>
#endif
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

/** @overload */
#if DISPENSO_HAS_CONCEPTS
template <typename TaskSetT, std::integral IntegerA, std::integral IntegerB, typename F>
  requires std::invocable<F, IntegerA, IntegerB>
#else
template <
    typename TaskSetT,
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true,
    std::enable_if_t<detail::CanInvoke<F(IntegerA, IntegerB)>::value, bool> = true>
#endif
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
#if DISPENSO_HAS_CONCEPTS
template <std::integral IntegerA, std::integral IntegerB, typename F>
#else
template <
    typename IntegerA,
    typename IntegerB,
    typename F,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
#endif
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
#if DISPENSO_HAS_CONCEPTS
template <
    typename TaskSetT,
    std::integral IntegerA,
    std::integral IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen>
  requires std::invocable<F, typename StateContainer::reference, IntegerA>
#else
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
#endif
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

/** @overload */
#if DISPENSO_HAS_CONCEPTS
template <
    typename TaskSetT,
    std::integral IntegerA,
    std::integral IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen>
  requires std::invocable<F, typename StateContainer::reference, IntegerA, IntegerB>
#else
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
#endif
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
#if DISPENSO_HAS_CONCEPTS
template <
    std::integral IntegerA,
    std::integral IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen>
#else
template <
    typename IntegerA,
    typename IntegerB,
    typename F,
    typename StateContainer,
    typename StateGen,
    std::enable_if_t<std::is_integral<IntegerA>::value, bool> = true,
    std::enable_if_t<std::is_integral<IntegerB>::value, bool> = true>
#endif
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
