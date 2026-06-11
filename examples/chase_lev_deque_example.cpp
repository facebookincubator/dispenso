/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example chase_lev_deque_example.cpp
 * Demonstrates the SPMC work-stealing deque: one producer pushes, multiple consumers steal.
 */

#include <dispenso/chase_lev_deque.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  // Example 1: Single-thread push and pop (LIFO order)
  std::cout << "Example 1: Owner push/pop (LIFO)\n";
  {
    dispenso::ChaseLevDeque<int, 64> deque;

    deque.try_push(10);
    deque.try_push(20);
    deque.try_push(30);
    std::cout << "  Size after 3 pushes: " << deque.size() << "\n";

    int val = 0;
    deque.try_pop(val);
    std::cout << "  Popped (LIFO): " << val << "\n"; // 30

    deque.try_pop(val);
    std::cout << "  Popped (LIFO): " << val << "\n"; // 20
  }

  // Example 2: Steal from another thread (FIFO order)
  std::cout << "\nExample 2: Steal from another thread (FIFO)\n";
  {
    dispenso::ChaseLevDeque<int, 64> deque;

    // Push several items
    for (int i = 1; i <= 5; ++i) {
      deque.try_push(i);
    }

    // Steal from a different thread
    int stolen = 0;
    std::thread thief([&deque, &stolen]() {
      // try_steal returns the oldest element (FIFO)
      deque.try_steal(stolen);
    });
    thief.join();

    std::cout << "  Stolen (FIFO, oldest): " << stolen << "\n"; // 1
    std::cout << "  Remaining size: " << deque.size() << "\n";
  }

  // Example 3: One producer, multiple consumers (work-stealing pattern)
  std::cout << "\nExample 3: SPMC work-stealing pattern\n";
  {
    dispenso::ChaseLevDeque<int, 256> deque;
    std::atomic<int> totalStolen(0);
    std::atomic<int> stealCount(0);
    std::atomic<bool> done(false);

    constexpr int kNumItems = 200;
    constexpr int kNumStealers = 3;

    // Launch stealer threads
    std::vector<std::thread> stealers;
    for (int t = 0; t < kNumStealers; ++t) {
      stealers.emplace_back([&]() {
        int val = 0;
        while (!done.load(std::memory_order_acquire) || !deque.empty()) {
          if (deque.try_steal(val)) {
            totalStolen.fetch_add(val, std::memory_order_relaxed);
            stealCount.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }

    // Producer: push work items
    for (int i = 0; i < kNumItems; ++i) {
      while (!deque.try_push(i)) {
        // Deque full; pop one ourselves to make room
        int val = 0;
        if (deque.try_pop(val)) {
          totalStolen.fetch_add(val, std::memory_order_relaxed);
          stealCount.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    done.store(true, std::memory_order_release);

    // Drain remaining items
    int val = 0;
    while (deque.try_pop(val)) {
      totalStolen.fetch_add(val, std::memory_order_relaxed);
      stealCount.fetch_add(1, std::memory_order_relaxed);
    }

    for (auto& t : stealers) {
      t.join();
    }

    int expectedSum = kNumItems * (kNumItems - 1) / 2;
    std::cout << "  Items processed: " << stealCount.load() << " (expected: " << kNumItems << ")\n";
    std::cout << "  Sum: " << totalStolen.load() << " (expected: " << expectedSum << ")\n";
  }

  // Example 4: Capacity and overflow handling
  std::cout << "\nExample 4: Bounded capacity\n";
  {
    dispenso::ChaseLevDeque<int, 4> deque; // Very small capacity

    std::cout << "  Capacity: " << deque.capacity() << "\n";

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
      ok = deque.try_push(i);
    }
    std::cout << "  Pushed 4 items: " << (ok ? "success" : "failed") << "\n";

    // 5th push should fail (deque is full)
    ok = deque.try_push(99);
    std::cout << "  5th push: " << (ok ? "success" : "full (expected)") << "\n";
  }

  std::cout << "\nAll ChaseLevDeque examples completed successfully!\n";
  return 0;
}
