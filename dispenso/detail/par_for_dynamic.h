/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Dynamic-scheduling (work-stealing) implementation for parallel_for.
//
// Not a standalone header — included from parallel_for.h after all
// required types (ChunkedRange, ParForOptions, etc.) are defined.

#pragma once

namespace dispenso {
namespace detail {

// Per-group range for decoupled-atomic dynamic parallel_for. Each group of
// threads shares an L3-local atomic index over a contiguous sub-range of
// chunks, eliminating cross-CCD cache-line bouncing on many-core machines.
struct alignas(kCacheLineSize) GroupRange {
  std::atomic<size_t> index{0};
  size_t startChunk{0};
  size_t numGroupChunks{0};
};

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

// Multi-group path for parallel_for_dynamicImpl. Partitions chunks among
// effectiveGroups groups, each with its own cache-line-aligned atomic index.
template <
    typename TaskSetT,
    typename IntegerT,
    typename F,
    typename StateContainer,
    typename ExitAction>
void parallel_for_dynamicMultiGroupImpl(
    TaskSetT& taskSet,
    StateContainer& states,
    IntegerT start,
    IntegerT end,
    F&& f,
    size_t numToLaunch,
    typename ChunkedRange<IntegerT>::size_type chunkSize,
    typename ChunkedRange<IntegerT>::size_type numChunks,
    ExitAction exitAction,
    bool wait,
    size_t effectiveGroups,
    size_t totalWorkers) {
  size_t blockBytes =
      detail::alignToCacheLine(sizeof(GroupBlock)) + sizeof(GroupRange) * effectiveGroups;

  // Both the wait and no-wait paths heap-own the block; the last worker to
  // finish frees it. A thread-local buffer-reuse optimization for the wait
  // path was intentionally not used here: it is unsafe under nested multi-group
  // parallel_for on the same thread.
  void* blockMem = detail::alignedMalloc(blockBytes, kCacheLineSize);

  auto* block = new (blockMem) GroupBlock{};
  block->numGroups = effectiveGroups;
  block->totalWorkers = totalWorkers;
  block->heapOwned = true;

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
    // Snapshot block-owned metadata BEFORE the release increment below.
    // Once a worker performs exitCounter.fetch_add, the worker that
    // observes the final count frees `block`, so `block` must not be
    // dereferenced afterwards.
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
    // The acq_rel increment orders every worker's prior block accesses
    // before the last worker's free.
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

// Dynamic (work-stealing) parallel_for implementation.
//
// Workers are partitioned into groups of ~16 threads, each with its own
// atomic index over a contiguous sub-range of chunks. When the total
// worker count is below 16, there is exactly one group and the behavior
// is equivalent to the original single-atomic approach.
//
// The ExitAction callback is invoked when a worker finds no more chunks;
// this allows the no-wait path to deallocate heap state when the last
// worker exits, while the wait path passes a no-op.
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
  // CCD's L3. Falls back to a heuristic (~16 threads per group).
  size_t l3Groups = CpuSet::l3CacheGroups().size();
  size_t effectiveGroups;
  if (l3Groups > 1 && totalWorkers > 16) {
    effectiveGroups = std::min(l3Groups, totalWorkers);
  } else {
    effectiveGroups = std::max<size_t>(1, (totalWorkers + 15) / 16);
  }

  if (effectiveGroups > 1) {
    parallel_for_dynamicMultiGroupImpl(
        taskSet,
        states,
        start,
        end,
        std::forward<F>(f),
        numToLaunch,
        chunkSize,
        numChunks,
        exitAction,
        wait,
        effectiveGroups,
        totalWorkers);
    return;
  }

  // Single group: use the original shared-index path with no extra overhead.
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
}

// No-wait dynamic dispatch for the top-level parallel_for. Handles the
// heap-allocated atomic index and the exit action that runs the
// granularity tail (if any) on the last worker to finish.
template <typename TaskSetT, typename IntegerT, typename F, typename StateContainer>
void parallel_for_dynamicNoWaitDispatch(
    TaskSetT& taskSet,
    StateContainer& states,
    const ChunkedRange<IntegerT>& parRange,
    F&& f,
    size_t numToLaunch,
    typename ChunkedRange<IntegerT>::size_type chunkSize,
    typename ChunkedRange<IntegerT>::size_type numChunks,
    IntegerT fullEnd,
    bool hasTail) {
  using SizeType = decltype(numChunks);
  struct ChunkIndex {
    std::atomic<SizeType> index;
  };
  static_assert(sizeof(ChunkIndex) <= kCacheLineSize, "ChunkIndex must fit in one cache line");
  char* mem = allocSmallBuffer<kCacheLineSize>();
  auto* ci = new (mem) ChunkIndex{{0}};
  SizeType lastExit = numChunks + static_cast<SizeType>(numToLaunch) - 1;
  IntegerT tailStart = parRange.end;
  IntegerT tailEnd = fullEnd;
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
      numToLaunch,
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
      false);
}

} // namespace detail
} // namespace dispenso
