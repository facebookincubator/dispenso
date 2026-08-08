/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example mpmc_ring_buffer_example.cpp
 * Demonstrates the bounded multi-producer multi-consumer ring buffer:
 * fail-fast try_push/try_pop, batch push, and concurrent producers/consumers.
 */

#include <dispenso/mpmc_ring_buffer.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  // Example 1: Single-threaded push and pop (FIFO order).
  std::cout << "Example 1: Single-threaded push/pop (FIFO)\n";
  {
    dispenso::MpmcRingBuffer<int, 8> ring;

    for (int i = 0; i < 3; ++i) {
      ring.try_push(i * 10);
    }
    std::cout << "  Size after 3 pushes: " << ring.size() << "\n";

    int value = 0;
    while (ring.try_pop(value)) {
      std::cout << "  Popped: " << value << "\n";
    }
  }

  // Example 2: Fail-fast semantics -- push fails when full, pop fails when empty.
  std::cout << "\nExample 2: try_push fails when full\n";
  {
    dispenso::MpmcRingBuffer<int, 2> ring;
    std::cout << "  push 1: " << ring.try_push(1) << "\n"; // 1 (success)
    std::cout << "  push 2: " << ring.try_push(2) << "\n"; // 1 (success)
    std::cout << "  push 3 (full): " << ring.try_push(3) << "\n"; // 0 (fail)
  }

  // Example 3: Batch push of several items with a single reservation.
  std::cout << "\nExample 3: Batch push\n";
  {
    dispenso::MpmcRingBuffer<int, 16> ring;
    std::vector<int> items = {1, 2, 3, 4, 5};
    const size_t pushed = ring.try_push_batch(items.data(), items.size());
    std::cout << "  Pushed " << pushed << " of " << items.size() << " items\n";
  }

  // Example 4: Multiple producers and consumers concurrently.
  std::cout << "\nExample 4: Concurrent producers/consumers\n";
  {
    constexpr int kProducers = 4;
    constexpr int kPerProducer = 1000;
    dispenso::MpmcRingBuffer<int, 1024> ring;

    std::atomic<int> consumedSum{0};
    std::atomic<int> consumedCount{0};
    std::atomic<bool> done{false};

    std::vector<std::thread> consumers;
    for (int c = 0; c < 2; ++c) {
      consumers.emplace_back([&]() {
        int value = 0;
        while (!done.load(std::memory_order_acquire) ||
               consumedCount.load(std::memory_order_acquire) < kProducers * kPerProducer) {
          if (ring.try_pop(value)) {
            consumedSum.fetch_add(value, std::memory_order_relaxed);
            consumedCount.fetch_add(1, std::memory_order_acq_rel);
          }
        }
      });
    }

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
      producers.emplace_back([&]() {
        for (int i = 0; i < kPerProducer; ++i) {
          while (!ring.try_push(1)) {
            // Buffer momentarily full; retry until a consumer drains a slot.
          }
        }
      });
    }

    for (auto& t : producers) {
      t.join();
    }
    done.store(true, std::memory_order_release);
    for (auto& t : consumers) {
      t.join();
    }

    std::cout << "  Consumed " << consumedCount.load() << " items, sum = " << consumedSum.load()
              << " (expected " << kProducers * kPerProducer << ")\n";
  }

  std::cout << "\nAll MpmcRingBuffer examples completed successfully!\n";
  return 0;
}
