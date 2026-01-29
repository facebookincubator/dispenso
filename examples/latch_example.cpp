/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example latch_example.cpp
 * Demonstrates thread synchronization with dispenso::Latch.
 */

#include <dispenso/latch.h>
#include <dispenso/parallel_for.h>
#include <dispenso/thread_pool.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  // Example 1: Basic latch synchronization
  std::cout << "Example 1: Basic latch synchronization\n";
  {
    constexpr int kNumThreads = 4;
    dispenso::Latch latch(kNumThreads);
    std::atomic<int> readyCount(0);

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    for (int i = 0; i < kNumThreads; ++i) {
      taskSet.schedule([&latch, &readyCount, i]() {
        // Simulate some initialization work
        readyCount.fetch_add(1, std::memory_order_relaxed);
        std::cout << "  Thread " << i << " ready\n";

        // Signal completion and wait for others
        latch.arrive_and_wait();

        // All threads are synchronized here
      });
    }

    taskSet.wait();

    std::cout << "  All " << readyCount.load() << " threads synchronized\n";
  }

  // Example 2: Using count_down separately from wait
  std::cout << "\nExample 2: Separate count_down and wait\n";
  {
    constexpr int kNumWorkers = 3;
    dispenso::Latch workComplete(kNumWorkers);
    std::vector<int> results(kNumWorkers, 0);

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    // Launch workers
    for (int i = 0; i < kNumWorkers; ++i) {
      taskSet.schedule([&workComplete, &results, i]() {
        // Do work
        results[static_cast<size_t>(i)] = (i + 1) * 10;

        // Signal work is done (non-blocking)
        workComplete.count_down();
      });
    }

    // Main thread waits for all workers
    workComplete.wait();

    std::cout << "  Results: ";
    for (int r : results) {
      std::cout << r << " ";
    }
    std::cout << "\n";
  }

  // Example 3: try_wait for non-blocking check
  std::cout << "\nExample 3: Non-blocking try_wait check\n";
  {
    dispenso::Latch latch(1);

    std::cout << "  Before count_down, try_wait() = " << (latch.try_wait() ? "true" : "false")
              << "\n";

    latch.count_down();

    std::cout << "  After count_down, try_wait() = " << (latch.try_wait() ? "true" : "false")
              << "\n";
  }

  // Example 4: Latch for phased computation
  std::cout << "\nExample 4: Phased computation with multiple latches\n";
  {
    constexpr int kNumThreads = 4;
    dispenso::Latch phase1Complete(kNumThreads);
    dispenso::Latch phase2Complete(kNumThreads);

    std::atomic<int> phase1Sum(0);
    std::atomic<int> phase2Sum(0);

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    for (int i = 0; i < kNumThreads; ++i) {
      taskSet.schedule([&, i]() {
        // Phase 1: Each thread contributes its value
        phase1Sum.fetch_add(i + 1, std::memory_order_relaxed);
        phase1Complete.arrive_and_wait();

        // Phase 2: Each thread reads the phase 1 result
        int localResult = phase1Sum.load() * (i + 1);
        phase2Sum.fetch_add(localResult, std::memory_order_relaxed);
        phase2Complete.count_down();
      });
    }

    phase2Complete.wait();

    std::cout << "  Phase 1 sum: " << phase1Sum.load() << " (1+2+3+4 = 10)\n";
    std::cout << "  Phase 2 sum: " << phase2Sum.load() << " (10*1 + 10*2 + 10*3 + 10*4 = 100)\n";
  }

  // Example 5: Latch as a one-shot gate
  std::cout << "\nExample 5: Latch as a one-shot start gate\n";
  {
    dispenso::Latch startGate(1);
    std::atomic<int> workersStarted(0);

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    // Schedule workers that wait for the start signal
    for (int i = 0; i < 5; ++i) {
      taskSet.schedule([&startGate, &workersStarted]() {
        // Wait for start signal
        startGate.wait();
        workersStarted.fetch_add(1, std::memory_order_relaxed);
      });
    }

    // Let workers queue up
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "  Workers before gate opens: " << workersStarted.load() << "\n";

    // Open the gate
    startGate.count_down();

    taskSet.wait();
    std::cout << "  Workers after gate opens: " << workersStarted.load() << "\n";
  }

  std::cout << "\nAll Latch examples completed successfully!\n";
  return 0;
}
