/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Future benchmarks targeting workloads where futures are the natural
// abstraction (rather than raw scheduling).  Compute-bound fork-join code
// belongs in tree_benchmark.cpp; this file is for patterns where futures
// genuinely earn their keep.
//
// Two workloads:
//
//   1. Speculative search (when_any) — launch K candidates, take the first to
//      complete.  Models "first usable answer wins" patterns: ML inference
//      ensembles, parallel search from multiple seeds, parallel hash lookups
//      across shards.
//
//   2. Memoization fan-in (shared future) — M concurrent consumers request
//      results for K cached keys.  First request initializes the cached
//      future; subsequent requests for the same key wait on it.  Models
//      lazy asset/model loaders with concurrent clients.

#include <array>
#include <atomic>
#include <future>
#include <mutex>
#include <optional>
#include <vector>

#include <dispenso/future.h>
#include <dispenso/task_set.h>
#include <dispenso/thread_pool.h>

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

#include "thread_benchmark_common.h"

namespace {

// Same workload kernel used in tree_benchmark.cpp.  CPU-bound, no allocations.
uint32_t busyWork(uint32_t seed, size_t iterations) {
  uint32_t h = seed;
  for (size_t i = 0; i < iterations; ++i) {
    h *= 2654435761u;
    h ^= h >> 16;
  }
  return h;
}

// ============================================================================
// Benchmark 1: Speculative search via when_any.
//
// Hash-cracking style workload: each candidate scans iterations of a 32-bit
// hash, looking for one whose low N bits are zero (a poor-man's proof-of-
// work).  Different seeds find a match at different iteration counts;
// distribution is roughly geometric with mean 1/p where p = 2^-N.  Some
// seeds get lucky and find quickly; others slog all the way to maxIters
// without finding.  The loop EARLY-EXITS on success — work is genuinely
// search-shaped, not fixed-cost.
//
// Speculative pattern: launch K candidates with different seeds, take the
// first one to finish.  Speedup over picking one randomly comes from
// running K independent geometric variables and taking their min.
//
// The serial baseline runs ONE candidate (no oracle pre-pick) and reports
// the per-find cost.  Comparing speculative-K against serial shows the
// parallel speedup of taking the min of K geometric variables.
// ============================================================================

constexpr size_t kSpecMaxIters = 5'000'000;
constexpr uint32_t kSpecMask = 0xffff; // 16 zero bits → ~65k iters per find on avg

// Returns the iteration index where (h & mask) == 0 was satisfied, or maxIters
// if no match was found within the budget.
size_t crackHash(uint32_t seed, size_t maxIters, uint32_t mask) {
  uint32_t h = seed;
  for (size_t i = 0; i < maxIters; ++i) {
    h *= 2654435761u;
    h ^= h >> 16;
    if ((h & mask) == 0) {
      return i + 1;
    }
  }
  return maxIters;
}

// Honest serial baseline: pick one seed (no oracle) and run.  Across many
// iterations this samples the geometric find-time distribution.
void BM_speculative_serial(benchmark::State& state) {
  uint32_t iter = 0;
  for (auto _ : state) {
    size_t found = crackHash(0xc0ffee + iter, kSpecMaxIters, kSpecMask);
    benchmark::DoNotOptimize(found);
    ++iter;
  }
}

// Speculative search: measures TIME-TO-FIRST-RESULT (the search cost only).
// Losers are drained between iterations using state.PauseTiming() so the
// drain cost is excluded from the measured time and the pool starts each
// iteration clean.  Without this, a tight benchmark loop saturates the pool
// with leftover loser work and the measurement degenerates into
// throughput-bounded cycle time rather than search latency.
//
// TODO: When dispenso gains a Future-level cancellation token, when_any can
// auto-trigger cancellation on losers.  Cancellation won't interrupt running
// tasks (compute is opaque to the library), but it will (a) skip
// not-yet-started losers when the pool is saturated and (b) let user kernels
// cooperatively short-circuit on the token.  The PauseTiming/drain dance
// becomes unnecessary at that point.
template <size_t K>
void BM_speculative_dispenso(benchmark::State& state) {
  auto& pool = dispenso::globalThreadPool();
  uint32_t iter = 0;
  for (auto _ : state) {
    std::vector<dispenso::Future<size_t>> candidates;
    candidates.reserve(K);
    // Distinct seeds per candidate (and per iter) so each runs an independent
    // geometric search.
    for (size_t k = 0; k < K; ++k) {
      const uint32_t seed = 0xc0ffee + iter * static_cast<uint32_t>(K) + static_cast<uint32_t>(k);
      candidates.emplace_back([seed]() { return crackHash(seed, kSpecMaxIters, kSpecMask); }, pool);
    }
    size_t winner = dispenso::when_any(candidates.begin(), candidates.end()).get();
    benchmark::DoNotOptimize(winner);
    state.PauseTiming();
    for (auto& c : candidates) {
      c.wait();
    }
    state.ResumeTiming();
    ++iter;
  }
}

#if !defined(BENCHMARK_WITHOUT_FOLLY)
template <size_t K>
void BM_speculative_folly(benchmark::State& state) {
  static folly::CPUThreadPoolExecutor exec(std::thread::hardware_concurrency());
  uint32_t iter = 0;
  for (auto _ : state) {
    // collectAny moves the input futures in; losers' handles are lost.  Track
    // completion via a shared atomic counter so we can drain in unmeasured
    // time after when_any returns, mirroring the dispenso variant.
    auto remaining = std::make_shared<std::atomic<size_t>>(K);
    std::vector<folly::SemiFuture<size_t>> candidates;
    candidates.reserve(K);
    for (size_t k = 0; k < K; ++k) {
      const uint32_t seed = 0xc0ffee + iter * static_cast<uint32_t>(K) + static_cast<uint32_t>(k);
      candidates.emplace_back(folly::via(&exec, [seed, remaining]() {
        size_t found = crackHash(seed, kSpecMaxIters, kSpecMask);
        remaining->fetch_sub(1, std::memory_order_release);
        return found;
      }));
    }
    auto result = folly::collectAny(std::move(candidates)).get();
    benchmark::DoNotOptimize(result.first);
    state.PauseTiming();
    while (remaining->load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
    state.ResumeTiming();
    ++iter;
  }
}
#endif

BENCHMARK(BM_speculative_serial)->UseRealTime();
BENCHMARK_TEMPLATE(BM_speculative_dispenso, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_speculative_dispenso, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_speculative_dispenso, 16)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK_TEMPLATE(BM_speculative_folly, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_speculative_folly, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_speculative_folly, 16)->UseRealTime();
#endif

// ============================================================================
// Benchmark 2: KV cache memoization.
//
// Models a key-value cache where many concurrent requesters look up keys with
// a skewed access pattern (a small fraction of slots get most of the
// requests).  First request to a slot atomically claims it, kicks off the
// expensive compute, and stores the resulting Future.  Subsequent requests
// for that same slot just observe the already-claimed Future and (a) wait if
// the compute is still in flight or (b) return immediately if it's done.
//
// The win pattern requires more concurrent demand than pool threads can
// satisfy in parallel — otherwise every request runs its own compute in
// parallel and there's nothing for memoization to dedupe.  We achieve this
// by making the request count >> pool size.
//
// Cache slot uses double-checked atomic-bool + mutex:
//   - Hot read path: one acquire load + branch (no mutex).
//   - Cold init path: mutex serializes; one winner constructs the Future.
//
// Why mutex on init (instead of CAS-publishing a heap-allocated Future): the
// Future ctor has a side effect (queues a task on the pool).  Two racing
// initializers would each queue a wasted compute the loser can't recall.
// Mutex serializes init so only the winner ever constructs.
// ============================================================================

constexpr size_t kKvNumSlots = 1024;
constexpr size_t kKvNumRequests = 4096; // ~25x oversubscription on 165-thread pool
constexpr size_t kKvComputeWork = 500000; // ~830µs per compute (5 cycles/iter at ~3GHz)

// Deterministic skewed access: 50% of requests hit the top 10% of slots.
size_t pickKey(size_t i) {
  uint64_t h = (i * 8121u + 28411u) & 0xffffffffu;
  if ((h & 0x1u) == 0) {
    return (h >> 1) % (kKvNumSlots / 10); // hot
  }
  return (h >> 1) % kKvNumSlots; // cold
}

struct DispensoCacheSlot {
  std::atomic<bool> initialized{false};
  std::mutex mu;
  std::optional<dispenso::Future<uint32_t>> future;
};

struct StdCacheSlot {
  std::atomic<bool> initialized{false};
  std::mutex mu;
  std::shared_future<uint32_t> future;
};

#if !defined(BENCHMARK_WITHOUT_FOLLY)
struct FollyCacheSlot {
  std::atomic<bool> initialized{false};
  std::mutex mu;
  std::unique_ptr<folly::SharedPromise<uint32_t>> promise;
};
#endif

// No-cache baseline: every request computes independently. Upper bound on
// what memoization can save you when compute is scarce.
void BM_kv_no_cache(benchmark::State& state) {
  auto& pool = dispenso::globalThreadPool();
  for (auto _ : state) {
    dispenso::ConcurrentTaskSet tasks(pool);
    for (size_t i = 0; i < kKvNumRequests; ++i) {
      tasks.schedule([i]() {
        size_t key = pickKey(i);
        uint32_t v = busyWork(0xc0ffee + static_cast<uint32_t>(key), kKvComputeWork);
        benchmark::DoNotOptimize(v);
      });
    }
    tasks.wait();
  }
}

// Idiomatic dispenso shape:
//   - TaskSet (non-concurrent) for the 4096 consumer fan-out, since this is
//     single-producer.  scheduleBulk amortizes per-task atomic overhead and
//     uses the central-queue's bulk-enqueue path.  TaskSet always uses the
//     central queue, which is right for these tiny consumer tasks.
//   - Future(lambda, pool) for the cached compute, which routes through
//     pool.schedulePlaced (placed/locality-aware path).  Heavy ~830µs
//     computes benefit from steal-ring distribution.
void BM_kv_dispenso(benchmark::State& state) {
  auto& pool = dispenso::globalThreadPool();
  for (auto _ : state) {
    state.PauseTiming();
    auto cache = std::make_unique<std::array<DispensoCacheSlot, kKvNumSlots>>();
    state.ResumeTiming();

    auto& cacheRef = *cache;
    dispenso::TaskSet tasks(pool);
    tasks.scheduleBulk(kKvNumRequests, [&cacheRef, &pool](size_t i) {
      return [i, &cacheRef, &pool]() {
        size_t key = pickKey(i);
        DispensoCacheSlot& slot = cacheRef[key];
        if (!slot.initialized.load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lk(slot.mu);
          if (!slot.future) {
            slot.future.emplace(
                [key]() { return busyWork(0xc0ffee + static_cast<uint32_t>(key), kKvComputeWork); },
                pool);
            slot.initialized.store(true, std::memory_order_release);
          }
        }
        uint32_t v = slot.future->get();
        benchmark::DoNotOptimize(v);
      };
    });
    tasks.wait();

    state.PauseTiming();
    cache.reset();
    state.ResumeTiming();
  }
}

void BM_kv_std_shared_future(benchmark::State& state) {
  auto& pool = dispenso::globalThreadPool();
  for (auto _ : state) {
    state.PauseTiming();
    auto cache = std::make_unique<std::array<StdCacheSlot, kKvNumSlots>>();
    state.ResumeTiming();

    auto& cacheRef = *cache;
    dispenso::ConcurrentTaskSet tasks(pool);
    for (size_t i = 0; i < kKvNumRequests; ++i) {
      tasks.schedule([i, &cacheRef]() {
        size_t key = pickKey(i);
        StdCacheSlot& slot = cacheRef[key];
        if (!slot.initialized.load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lk(slot.mu);
          if (!slot.future.valid()) {
            slot.future = std::async(std::launch::async, [key]() {
                            return busyWork(0xc0ffee + static_cast<uint32_t>(key), kKvComputeWork);
                          }).share();
            slot.initialized.store(true, std::memory_order_release);
          }
        }
        uint32_t v = slot.future.get();
        benchmark::DoNotOptimize(v);
      });
    }
    tasks.wait();

    state.PauseTiming();
    cache.reset();
    state.ResumeTiming();
  }
}

#if !defined(BENCHMARK_WITHOUT_FOLLY)
void BM_kv_folly(benchmark::State& state) {
  static folly::CPUThreadPoolExecutor exec(std::thread::hardware_concurrency());
  auto& pool = dispenso::globalThreadPool();
  for (auto _ : state) {
    state.PauseTiming();
    auto cache = std::make_unique<std::array<FollyCacheSlot, kKvNumSlots>>();
    state.ResumeTiming();

    auto& cacheRef = *cache;
    dispenso::ConcurrentTaskSet tasks(pool);
    for (size_t i = 0; i < kKvNumRequests; ++i) {
      tasks.schedule([i, &cacheRef]() {
        size_t key = pickKey(i);
        FollyCacheSlot& slot = cacheRef[key];
        if (!slot.initialized.load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lk(slot.mu);
          if (!slot.promise) {
            slot.promise = std::make_unique<folly::SharedPromise<uint32_t>>();
            auto* p = slot.promise.get();
            // Submit the compute as a bare task; it fulfills the SharedPromise
            // so all consumers (current and future) can wait on it.
            exec.add([key, p]() {
              p->setValue(busyWork(0xc0ffee + static_cast<uint32_t>(key), kKvComputeWork));
            });
            slot.initialized.store(true, std::memory_order_release);
          }
        }
        uint32_t v = slot.promise->getSemiFuture().via(&exec).get();
        benchmark::DoNotOptimize(v);
      });
    }
    tasks.wait();

    state.PauseTiming();
    cache.reset();
    state.ResumeTiming();
  }
}
#endif

BENCHMARK(BM_kv_no_cache)->UseRealTime();
BENCHMARK(BM_kv_dispenso)->UseRealTime();
BENCHMARK(BM_kv_std_shared_future)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK(BM_kv_folly)->UseRealTime();
#endif

} // namespace

BENCHMARK_MAIN();
