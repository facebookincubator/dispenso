/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file par_for_stripe.h
 * Per-stripe atomic-counter work distribution for parallel_for.
 *
 * Inspired by Callisto-RTS (Harris/Kaestle, USENIX ATC 2015). The iteration
 * space [start, end) is partitioned into P contiguous stripes (one per
 * worker). Each stripe holds a single monotonic `next` cursor that advances
 * from begin toward end. Both the owner and any stealer claim chunkSize
 * iterations at a time via `next.fetch_add(chunkSize)`. A claim succeeds
 * iff the returned `prev` is < the stripe's `end` (immutable); otherwise the
 * stripe is exhausted.
 *
 * Using a single forward-moving cursor (shared between owner and stealers)
 * keeps the correctness argument trivial: each unique `prev` value claims a
 * unique disjoint sub-range. There is no possibility of overlap.
 *
 * Stealer victim selection uses has-work bitmasks (one bit per stripe,
 * packed into ceil(P/64) cache-line-isolated 64-bit words) so we never
 * probe an exhausted stripe. Each worker also has a precomputed per-L3
 * mask (intersection of "all workers" and "workers in my L3 group")
 * stored once at init. The same-L3 fast path is:
 *     hasWorkMasks[k].load() & l3MaskWords[myIdx][k]
 * then countTrailingZeros to pick a victim. The fallback (no same-L3
 * work) uses hasWorkMasks alone.
 *
 * Has-work bits are set to 1 at init for every non-empty stripe, and
 * cleared by the thread that wins the retire-CAS for a stripe — so the
 * mask becomes a point of write contention only at the very end of a
 * stripe's life, never on the hot claim path.
 *
 * Compared to the deque-based adaptive path:
 *  - Fixed chunk size (always exactly chunkSize iterations per f() call,
 *    except the last/boundary chunk). No recursive splitting decisions.
 *  - One atomic per claim (vs. the deque push + pop + has-work atomic of
 *    the adaptive path).
 *  - Stealer contention is on the victim stripe's `next` cursor only; no
 *    cross-thread CAS races on deque structures.
 *
 * The owner consuming front-to-back gets perfect prefetcher behavior on
 * its home stripe. Stealers competing for the same cursor add some
 * contention but each claim is still a contiguous chunkSize-sized run.
 *
 * Termination is by activeStripes == 0: every non-empty stripe registers
 * as active at construction; the worker that first observes a stripe is
 * exhausted decrements once. parallel_for completes when activeStripes
 * reaches 0.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include <dispenso/cpu_set.h>
#include <dispenso/detail/math.h>
#include <dispenso/platform.h>

namespace dispenso {
namespace detail {

// Per-stripe state. `end` is immutable after init; `next` is the shared
// claim cursor (owner and stealers both fetch_add). Cache-line-isolated so
// claim activity on one stripe doesn't false-share with adjacent stripes.
// Stored as the widened type so fetch_add(chunkSize) past `end` cannot
// overflow when IntegerT is a narrow type (e.g. int16_t over its full
// range). The claim succeeds only while next < end; if a claim races past
// end, we just fail-and-retire the stripe as usual.
template <typename IntegerT>
struct alignas(kCacheLineSize) StripeCursor {
  using WideT = typename std::conditional<std::is_signed<IntegerT>::value, int64_t, uint64_t>::type;
  WideT end;
  std::atomic<WideT> next;
  // True once any thread has observed `next >= end` and decremented
  // activeStripes. Prevents multiple decrements for the same stripe.
  std::atomic<bool> retired;
};

// One 64-bit has-work mask word, cache-line-isolated so stealers from
// different mask windows don't false-share.
struct alignas(kCacheLineSize) HasWorkWord {
  std::atomic<uint64_t> bits{0};
};

// Shared state across all workers for a single parallel_for invocation.
//
// Memory layout: a single placement-new'd buffer holds:
//   [stripes: numWorkers * StripeCursor]    (cache-line aligned)
//   [hasWorkMasks: numMaskWords * HasWorkWord] (cache-line aligned)
//   [l3MaskWords: numWorkers * numMaskWords * uint64_t]
// This avoids 3 separate heap allocations per parallel_for call.
template <typename IntegerT>
struct StripeState {
  StripeCursor<IntegerT>* stripes; // points into `buffer`
  HasWorkWord* hasWorkMasks; // points into `buffer`, numMaskWords entries
  // Per-worker immutable L3 mask. l3MaskWords[myIdx * numMaskWords + k] is
  // a bitmask over the workers in word k that share an L3 group with myIdx.
  // (l3MaskWords[myIdx*M + (myIdx>>6)] has the (myIdx & 63) bit CLEAR so
  // we don't self-probe.) Null if L3 topology is unavailable.
  uint64_t* l3MaskWords;
  // Backing buffer for the above (single allocation, alignedFree on destroy).
  void* buffer;

  uint32_t numWorkers;
  uint32_t numMaskWords; // ceil(numWorkers / 64)
  IntegerT chunkSize;
  uint32_t granularity;
  // Number of stripes that still contain unclaimed iterations. Init = count
  // of non-empty stripes; decremented exactly once per stripe by whichever
  // thread first observes the stripe is exhausted (via the `retired` CAS).
  alignas(kCacheLineSize) std::atomic<uint32_t> activeStripes{0};

  StripeState() = default;
  StripeState(const StripeState&) = delete;
  StripeState& operator=(const StripeState&) = delete;

  ~StripeState() {
    if (buffer) {
      // StripeCursor and HasWorkWord are trivial (atomic members destruct
      // trivially), so we only need to free the storage.
      detail::alignedFree(buffer);
    }
  }
};

// Get a cached mapping cpu_id -> L3-group-index, built once per process
// from CpuSet::l3CacheGroups(). Returns -1 for unknown CPUs.
inline const std::vector<int8_t>& cpuToL3Index() {
  static const std::vector<int8_t> table = []() {
    const auto& groups = ::dispenso::CpuSet::l3CacheGroups();
    int32_t maxCpu = -1;
    for (const auto& g : groups) {
      for (int32_t c : g.cpus) {
        if (c > maxCpu) {
          maxCpu = c;
        }
      }
    }
    std::vector<int8_t> v(static_cast<size_t>(maxCpu + 1), int8_t{-1});
    for (size_t gi = 0; gi < groups.size() && gi < 127; ++gi) {
      for (int32_t c : groups[gi].cpus) {
        if (c >= 0 && static_cast<size_t>(c) < v.size()) {
          v[static_cast<size_t>(c)] = static_cast<int8_t>(gi);
        }
      }
    }
    return v;
  }();
  return table;
}

// Per-worker stealer-local state. Lives on the worker's stack — no atomics.
struct StripeStealerLocal {
  uint32_t scanWord; // round-robin mask-word cursor
  int32_t lastVictim; // -1 == no warm victim
};

inline StripeStealerLocal makeStripeStealerLocal(uint32_t myIdx, uint32_t numMaskWords) {
  // Start scanning from our own mask word (so first probe hits a peer
  // likely in the same L3 group / nearby).
  return StripeStealerLocal{(myIdx >> 6) % std::max<uint32_t>(1, numMaskWords), -1};
}

// Round `value` DOWN (toward negative infinity) to a multiple of granularity.
template <typename IntegerT>
inline IntegerT alignDownStripe(IntegerT value, uint32_t granularity) {
  if (granularity <= 1) {
    return value;
  }
  IntegerT g = static_cast<IntegerT>(granularity);
  IntegerT d = value / g;
  // C++ truncates toward zero; correct to floor for signed negative remainders.
  if (std::is_signed<IntegerT>::value && d * g != value && value < IntegerT(0)) {
    --d;
  }
  return d * g;
}

// Try to claim a chunk from the given stripe. Returns true and sets
// [outBegin, outEnd) on success; false on exhaustion. The first caller to
// observe the stripe exhausted CAS-claims the `retired` flag, clears its
// has-work bit, and decrements activeStripes (so the stripe is counted
// down exactly once and no peer will probe it again via the mask scan).
template <typename IntegerT>
inline bool stripeClaim(
    StripeState<IntegerT>& state,
    uint32_t stripeIdx,
    IntegerT& outBegin,
    IntegerT& outEnd) {
  using Wide = typename StripeCursor<IntegerT>::WideT;
  auto& s = state.stripes[stripeIdx];
  const IntegerT chunkSize = state.chunkSize;
  Wide prev = s.next.fetch_add(static_cast<Wide>(chunkSize), std::memory_order_relaxed);
  if (prev >= s.end) {
    // Stripe exhausted before this claim. Try to be the one to retire it.
    bool expected = false;
    if (s.retired.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      // Memory ordering: the relaxed fetch_and (clearing the has-work bit)
      // is sequenced before the release fetch_sub (decrementing activeStripes).
      // Any thread that loads activeStripes with acquire and sees 0 is
      // guaranteed to also see the cleared bit, because release/acquire
      // on activeStripes publishes all prior writes — including the relaxed
      // mask clear. A stealer seeing the cleared bit while activeStripes > 0
      // is benign: it correctly skips an exhausted stripe.
      uint64_t bit = uint64_t{1} << (stripeIdx & 63);
      state.hasWorkMasks[stripeIdx >> 6].bits.fetch_and(~bit, std::memory_order_relaxed);
      state.activeStripes.fetch_sub(1, std::memory_order_release);
    }
    return false;
  }
  outBegin = static_cast<IntegerT>(prev);
  Wide endWide = prev + static_cast<Wide>(chunkSize);
  outEnd = static_cast<IntegerT>(endWide > s.end ? s.end : endWide);
  return true;
}

// Try to find a stripe with work, preferring same-L3 victims. Returns
// stripe index or -1 if none. Doesn't claim — caller must follow with
// stripeClaim (which may itself find the stripe exhausted in the race
// between mask-read and claim).
template <typename IntegerT>
inline int32_t pickStripeFromMasks(
    const StripeState<IntegerT>& state,
    uint32_t myIdx,
    const uint64_t* myL3Mask, // may be null
    uint32_t& scanWord) {
  const uint32_t numMaskWords = state.numMaskWords;
  if (numMaskWords == 0) {
    return -1;
  }
  // Pass A: same-L3 victims (intersection of hasWork and our L3 mask).
  if (myL3Mask) {
    for (uint32_t i = 0; i < numMaskWords; ++i) {
      uint32_t k = (scanWord + i) % numMaskWords;
      uint64_t bits = state.hasWorkMasks[k].bits.load(std::memory_order_relaxed) & myL3Mask[k];
      if (bits) {
        int bit = detail::countTrailingZeros(bits);
        scanWord = k;
        uint32_t v = (k << 6) + static_cast<uint32_t>(bit);
        if (v < state.numWorkers && v != myIdx) {
          return static_cast<int32_t>(v);
        }
        // Self bit shouldn't appear in our own L3 mask (cleared at init),
        // and v < numWorkers should always hold for a set has-work bit;
        // fall through defensively.
      }
    }
  }
  // Pass B: any victim with work.
  for (uint32_t i = 0; i < numMaskWords; ++i) {
    uint32_t k = (scanWord + i) % numMaskWords;
    uint64_t bits = state.hasWorkMasks[k].bits.load(std::memory_order_relaxed);
    if (myIdx >> 6 == k) {
      bits &= ~(uint64_t{1} << (myIdx & 63));
    }
    if (bits) {
      int bit = detail::countTrailingZeros(bits);
      scanWord = k;
      uint32_t v = (k << 6) + static_cast<uint32_t>(bit);
      if (v < state.numWorkers) {
        return static_cast<int32_t>(v);
      }
    }
  }
  return -1;
}

// Per-worker run loop.
template <typename IntegerT, typename F, typename UserState>
inline void
runStripeWorker(StripeState<IntegerT>& state, uint32_t myIdx, UserState& userState, F& f) {
  StripeStealerLocal sl = makeStripeStealerLocal(myIdx, state.numMaskWords);
  const uint64_t* myL3Mask = state.l3MaskWords
      ? state.l3MaskWords + static_cast<size_t>(myIdx) * state.numMaskWords
      : nullptr;

  IntegerT b, e;

  // Drain own stripe via repeated fetch_add. Contiguous forward consumption
  // gives the prefetcher a clean stride.
  while (stripeClaim(state, myIdx, b, e)) {
    f(userState, b, e);
  }

  // Own stripe done. Help peers via mask-driven steal scan with last-victim
  // retry (warms cursor cache line for the second claim in a row).
  while (true) {
    if (state.activeStripes.load(std::memory_order_acquire) == 0) {
      return;
    }

    // (1) Warm-cache retry of last successful victim.
    if (sl.lastVictim >= 0) {
      if (stripeClaim(state, static_cast<uint32_t>(sl.lastVictim), b, e)) {
        f(userState, b, e);
        continue;
      }
      sl.lastVictim = -1;
    }

    // (2) Pick a victim via mask scan (same-L3 preferred, then fallback).
    int32_t v = pickStripeFromMasks(state, myIdx, myL3Mask, sl.scanWord);
    if (v >= 0) {
      if (stripeClaim(state, static_cast<uint32_t>(v), b, e)) {
        f(userState, b, e);
        sl.lastVictim = v;
        continue;
      }
      // Stripe was exhausted between mask-read and claim — retire already
      // happened via stripeClaim. Loop and try again.
      continue;
    }

    // No work found. Either we're done or a peer is mid-claim on the last
    // stripe(s). Re-check active count and back off briefly.
    if (state.activeStripes.load(std::memory_order_acquire) == 0) {
      return;
    }
    for (int i = 0; i < 64; ++i) {
      ::dispenso::detail::cpuRelax();
    }
  }
}

// Initialize the stripe state: divide [start, end) into numWorkers
// contiguous stripes, granularity-aligned. The last stripe absorbs any
// remainder so totals match exactly. Single allocation for all per-call
// state (stripes + hasWorkMasks + l3MaskWords).
template <typename IntegerT>
inline void initStripeState(
    StripeState<IntegerT>& state,
    IntegerT start,
    IntegerT end,
    uint32_t numWorkers,
    IntegerT chunkSize,
    uint32_t granularity) {
  using Wide = typename std::conditional<std::is_signed<IntegerT>::value, int64_t, uint64_t>::type;
  state.numWorkers = numWorkers;
  state.numMaskWords = (numWorkers + 63u) / 64u;
  state.chunkSize = chunkSize;
  state.granularity = std::max<uint32_t>(1, granularity);

  // Decide whether to allocate L3 masks (skip if topology unavailable).
  const bool haveL3 = !::dispenso::CpuSet::l3CacheGroups().empty();

  // Compute the single buffer layout. Each section is cache-line aligned
  // for safety (stripes and hasWorkMasks are inherently cache-line types;
  // l3MaskWords is uint64_t and doesn't need it, but the start needs to
  // be at least 8-byte aligned, and cache-line is a no-op upgrade).
  const size_t stripesBytes = sizeof(StripeCursor<IntegerT>) * numWorkers;
  const size_t hasWorkBytes = sizeof(HasWorkWord) * state.numMaskWords;
  const size_t l3MaskBytes =
      haveL3 ? sizeof(uint64_t) * static_cast<size_t>(numWorkers) * state.numMaskWords : 0;
  // Pad each section to cache-line so the next section starts aligned.
  auto roundUp = [](size_t n, size_t align) { return (n + align - 1) & ~(align - 1); };
  const size_t off0 = 0;
  const size_t off1 = roundUp(off0 + stripesBytes, kCacheLineSize);
  const size_t off2 = roundUp(off1 + hasWorkBytes, kCacheLineSize);
  const size_t totalBytes = off2 + l3MaskBytes;

  state.buffer = detail::alignedMalloc(totalBytes, kCacheLineSize);
  auto* base = static_cast<char*>(state.buffer);
  state.stripes = reinterpret_cast<StripeCursor<IntegerT>*>(base + off0);
  state.hasWorkMasks = reinterpret_cast<HasWorkWord*>(base + off1);
  state.l3MaskWords = haveL3 ? reinterpret_cast<uint64_t*>(base + off2) : nullptr;

  // Placement-new the StripeCursor and HasWorkWord (initializes atomics).
  for (uint32_t i = 0; i < numWorkers; ++i) {
    new (state.stripes + i) StripeCursor<IntegerT>();
  }
  for (uint32_t k = 0; k < state.numMaskWords; ++k) {
    new (state.hasWorkMasks + k) HasWorkWord();
  }

  // Build per-worker L3 masks. For each worker w, l3MaskWords[w*M + k] is
  // a bit-per-worker mask of which workers in word k share an L3 group
  // with w. We need each worker's L3 group id at init time — sample it
  // here using the dispatching thread's view. (Workers haven't run yet,
  // so we can't ask them; instead we walk the CpuSet groups and assign
  // each worker an L3 by interleaving workers across groups. This is a
  // heuristic — actual placement depends on the OS scheduler — but on a
  // pool with reasonably-sticky threads it's close enough to be useful.)
  //
  // Simpler heuristic: each worker is assigned to L3 group (w % numL3Groups).
  // Stable across calls; doesn't reflect actual OS thread placement but
  // gives a deterministic locality grouping that should help.
  if (haveL3) {
    std::memset(state.l3MaskWords, 0, l3MaskBytes);
    const auto& groups = ::dispenso::CpuSet::l3CacheGroups();
    const uint32_t numL3 = static_cast<uint32_t>(std::max<size_t>(1, groups.size()));
    for (uint32_t w = 0; w < numWorkers; ++w) {
      uint32_t myGroup = w % numL3;
      uint64_t* row = state.l3MaskWords + static_cast<size_t>(w) * state.numMaskWords;
      for (uint32_t v = 0; v < numWorkers; ++v) {
        if (v == w) {
          continue; // never self-probe
        }
        if ((v % numL3) == myGroup) {
          row[v >> 6] |= uint64_t{1} << (v & 63);
        }
      }
    }
  }

  Wide totalRange = static_cast<Wide>(end) - static_cast<Wide>(start);
  uint32_t activeCount = 0;
  IntegerT cursor = start;
  for (uint32_t i = 0; i < numWorkers; ++i) {
    auto& s = state.stripes[i];
    IntegerT stripeEnd;
    if (i + 1 == numWorkers) {
      stripeEnd = end;
    } else {
      Wide perStripe = totalRange / static_cast<Wide>(numWorkers);
      Wide endWide = static_cast<Wide>(start) + static_cast<Wide>(i + 1) * perStripe;
      stripeEnd = alignDownStripe(static_cast<IntegerT>(endWide), state.granularity);
      if (stripeEnd <= cursor) {
        stripeEnd = cursor;
      }
      if (stripeEnd >= end) {
        stripeEnd = end;
      }
    }
    s.end = stripeEnd;
    s.next.store(cursor, std::memory_order_relaxed);
    if (stripeEnd > cursor) {
      ++activeCount;
      s.retired.store(false, std::memory_order_relaxed);
      // Mark this stripe as having work in the has-work mask.
      state.hasWorkMasks[i >> 6].bits.fetch_or(uint64_t{1} << (i & 63), std::memory_order_relaxed);
    } else {
      // Empty stripe — pre-retire so peers don't try to claim from it.
      s.retired.store(true, std::memory_order_relaxed);
    }
    cursor = stripeEnd;
  }
  state.activeStripes.store(activeCount, std::memory_order_release);
}

} // namespace detail
} // namespace dispenso
