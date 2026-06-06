/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Static-scheduling implementation for parallel_for.
//
// Not a standalone header — included from parallel_for.h after all
// required types (ChunkedRange, ParForOptions, etc.) are defined.

#pragma once

namespace dispenso {
namespace detail {

// Computes the [start, end) range for a given task index under static chunking.
// Encapsulates the two-size distribution (ceil chunks for indices < transIdx,
// smaller chunks for indices >= transIdx) so the calling function doesn't
// carry the branching complexity.
template <typename IntegerT>
struct StaticChunkMapper {
  using size_type = typename ChunkedRange<IntegerT>::size_type;

  size_type numThreads;
  IntegerT chunkSize;
  IntegerT smallChunk;
  size_type transIdx;
  IntegerT rangeStart;
  IntegerT rangeEnd;

  std::pair<IntegerT, IntegerT> operator()(size_type idx) const {
    IntegerT start;
    if (idx < transIdx) {
      IntegerT i = static_cast<IntegerT>(idx);
      start = static_cast<IntegerT>(rangeStart + static_cast<IntegerT>(i * chunkSize));
    } else {
      IntegerT ti = static_cast<IntegerT>(transIdx);
      IntegerT ri = static_cast<IntegerT>(idx - transIdx);
      start = static_cast<IntegerT>(
          rangeStart + static_cast<IntegerT>(ti * chunkSize) +
          static_cast<IntegerT>(ri * smallChunk));
    }
    IntegerT end;
    if (idx + 1 == numThreads) {
      end = rangeEnd;
    } else if (idx < transIdx) {
      end = static_cast<IntegerT>(start + chunkSize);
    } else {
      end = static_cast<IntegerT>(start + smallChunk);
    }
    return {start, end};
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
    bool reuseExistingState,
    uint32_t granularity = 1) {
  using size_type = typename ChunkedRange<IntegerT>::size_type;

  size_type numThreads = std::min<size_type>(taskSet.numPoolThreads() + 1, maxThreads);
  numThreads = std::min(numThreads, range.size());
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
  IntegerT chunkStep = granularity > 1 ? static_cast<IntegerT>(granularity) : IntegerT{1};
  IntegerT smallChunk =
      static_cast<IntegerT>(chunkSize - (perfectlyChunked ? IntegerT{0} : chunkStep));

  StaticChunkMapper<IntegerT> chunkRange{
      numThreads,
      chunkSize,
      smallChunk,
      perfectlyChunked ? numThreads : static_cast<size_type>(chunking.transitionTaskIndex),
      range.start,
      range.end};

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
      // skip the ring fast path.
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      f(*stateIt, callerBounds.first, callerBounds.second);
    }
    taskSet.wait();
  }
}

} // namespace detail
} // namespace dispenso
