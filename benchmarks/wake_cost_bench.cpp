/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Wake-cost microbench: measures the per-platform syscall costs that
// determine the optimal `bumpAndWakeN` threshold (when to switch from
// "loop wake-one" to "single wake-all").
//
// Built on dispenso's EpochWaiter wrappers so it ports to Linux / macOS /
// Windows automatically. Output is platform-uniform: feed the numbers
// into the formula at the bottom of the report to set
// kWakeAllThreshold for the platform.
//
// Three measurements:
//   1. wake-1 producer syscall cost (constant in N)
//   2. wake-all producer syscall cost (linear in N)
//   3. spurious-wake round-trip (worker wakes, finds nothing, re-sleeps)

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include <dispenso/detail/epoch_waiter.h>

using clk = std::chrono::steady_clock;
using namespace dispenso::detail;

// ===================================================================
// Test 1+2: producer cost of wake-1 and wake-all, varying N.
// All N waiters share one EpochWaiter. We park them, then time the
// producer-side wake. For wake-1 we issue exactly 1 wake; for wake-all
// we issue 1 wake-all and let the kernel walk N waiters.
// ===================================================================

struct WaiterPool {
  EpochWaiter waiter;
  std::atomic<int> ready_count{0};
  std::atomic<int> exit{0};
};

static void waiter_thread(WaiterPool* p) {
  uint32_t epoch = p->waiter.current();
  p->ready_count.fetch_add(1, std::memory_order_release);
  while (p->exit.load(std::memory_order_acquire) == 0) {
    epoch = p->waiter.wait(epoch);
  }
}

static void wait_until_parked(WaiterPool* p, int N) {
  // Wait until all N waiters have called wait() at least once. This is
  // approximate (they may not be in the syscall yet), so we add a sleep.
  while (p->ready_count.load(std::memory_order_acquire) < N) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

struct WakeMeasurements {
  int64_t wake_1_ns;
  int64_t wake_all_ns;
};

static WakeMeasurements measure_wake_costs(int N) {
  WaiterPool pool;
  std::vector<std::thread> threads;
  threads.reserve(N);
  for (int i = 0; i < N; ++i) {
    threads.emplace_back(waiter_thread, &pool);
  }
  wait_until_parked(&pool, N);

  // Measure wake-1: issue many wake-1 calls, average the producer cost.
  // We must let the kernel re-park the woken thread between calls;
  // simplest is to also call bumpAndWake which advances the epoch — the
  // woken thread re-loops and re-parks.
  constexpr int kIters = 200;

  // Warmup
  for (int i = 0; i < 50; ++i) {
    pool.waiter.bumpAndWake();
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  int64_t wake_1_total = 0;
  for (int i = 0; i < kIters; ++i) {
    auto t0 = clk::now();
    pool.waiter.bumpAndWake();
    auto t1 = clk::now();
    wake_1_total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  // Measure wake-all: each call wakes all parked waiters, so we need to
  // sleep between calls long enough for them to re-park. 200µs is
  // generous on all platforms.
  for (int i = 0; i < 20; ++i) {
    pool.waiter.bumpAndWakeAll();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  int64_t wake_all_total = 0;
  for (int i = 0; i < kIters; ++i) {
    auto t0 = clk::now();
    pool.waiter.bumpAndWakeAll();
    auto t1 = clk::now();
    wake_all_total += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  pool.exit.store(1, std::memory_order_release);
  pool.waiter.bumpAndWakeAll();
  for (auto& t : threads)
    t.join();

  return {wake_1_total / kIters, wake_all_total / kIters};
}

// ===================================================================
// Test 3: spurious-wake round-trip — worker wakes, finds no useful work,
// re-sleeps. Producer measures total wall time of the cycle.
// ===================================================================

struct SpuriousPool {
  EpochWaiter waiter;
  std::atomic<int> wake_round{0}; // bumped on each wake
  std::atomic<int> sleep_round{0}; // worker sets when it re-sleeps
  std::atomic<int> exit{0};
};

static void spurious_thread(SpuriousPool* p) {
  uint32_t epoch = p->waiter.current();
  int last_seen = 0;
  while (p->exit.load(std::memory_order_acquire) == 0) {
    int wr = p->wake_round.load(std::memory_order_acquire);
    if (wr != last_seen) {
      // "Found nothing useful, re-sleep."
      last_seen = wr;
      p->sleep_round.store(wr, std::memory_order_release);
      continue;
    }
    epoch = p->waiter.wait(epoch);
  }
}

static int64_t measure_spurious_round_trip() {
  SpuriousPool pool;
  std::thread t(spurious_thread, &pool);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  constexpr int kIters = 200;
  int64_t total_ns = 0;
  for (int trial = 0; trial < kIters; ++trial) {
    int target = trial + 1;
    auto t0 = clk::now();
    pool.wake_round.fetch_add(1, std::memory_order_release);
    pool.waiter.bumpAndWake();
    while (pool.sleep_round.load(std::memory_order_acquire) != target) {
    }
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
    total_ns += ns;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  pool.exit.store(1, std::memory_order_release);
  pool.wake_round.fetch_add(1);
  pool.waiter.bumpAndWakeAll();
  t.join();

  return total_ns / kIters;
}

// ===================================================================
// Main: report measurements + recommended threshold.
//
// Threshold formula: kWakeAllThreshold is the smallest K where
//   K * wake_1_cost > wake_all_cost(groupSize)
// for the typical group size on this platform (16 by default).
// ===================================================================

int main() {
#ifdef __linux__
  printf("Platform: Linux (futex_wake supports exact K → kWakeAllThreshold = INT_MAX)\n");
#elif defined(__APPLE__)
  printf("Platform: macOS\n");
#elif defined(_WIN32)
  printf("Platform: Windows\n");
#else
  printf("Platform: unknown\n");
#endif
  printf("\n");
  printf("Spurious-wake round-trip: ");
  fflush(stdout);
  int64_t spurious_ns = measure_spurious_round_trip();
  printf("%lld ns\n\n", static_cast<long long>(spurious_ns));

  printf("Producer-side wake costs (avg over 200 trials):\n");
  printf(
      "%5s | %10s | %10s | %s\n",
      "N",
      "wake-1",
      "wake-all",
      "K threshold (wake-all wins for K >=)");
  printf("------+------------+------------+------------------------------------\n");

  for (int N : {2, 3, 4, 6, 8, 10, 12, 16, 32, 64, 96}) {
    if (N > static_cast<int>(std::thread::hardware_concurrency()) - 1)
      continue;
    auto m = measure_wake_costs(N);
    // Compute threshold: smallest K where K * wake_1 > wake_all (group of N).
    // I.e., K > wake_all / wake_1.
    int64_t threshold = m.wake_1_ns > 0 ? (m.wake_all_ns / m.wake_1_ns + 1) : 0;
    // Use %lld + long long cast for portability — int64_t is `long long` on
    // Windows MSVC, not `long`, so %ld would print garbage there.
    printf(
        "%5d | %8lld ns | %8lld ns | %lld\n",
        N,
        static_cast<long long>(m.wake_1_ns),
        static_cast<long long>(m.wake_all_ns),
        static_cast<long long>(threshold));
    fflush(stdout);
  }

  printf("\nGuidance:\n");
  printf("- For each group size G in production, look up the row N=G above.\n");
  printf("- The 'K threshold' column tells you: for wake-K calls with K below\n");
  printf("  this value, the loop wake-one path is cheaper. For K >= threshold,\n");
  printf("  the wake-all path is cheaper (ignoring spurious-wake CPU cost,\n");
  printf("  which is bounded by spurious-round-trip * (G - K) and runs in\n");
  printf("  parallel on idle cores).\n");
  printf("- Set DISPENSO_TUNE_WAKE_ALL_THRESHOLD to the threshold value for\n");
  printf("  your default group size on this platform.\n");
  return 0;
}
