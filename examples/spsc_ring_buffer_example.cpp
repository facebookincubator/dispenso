/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example spsc_ring_buffer_example.cpp
 * Demonstrates the single-producer single-consumer ring buffer: the fast,
 * lock-free choice when exactly one thread pushes and one thread pops.
 */

#include <dispenso/spsc_ring_buffer.h>

#include <iostream>
#include <thread>
#include <vector>

int main() {
  // Example 1: Single-threaded push and pop (FIFO order).
  std::cout << "Example 1: Single-threaded push/pop (FIFO)\n";
  {
    dispenso::SPSCRingBuffer<int, 8> ring;

    for (int i = 0; i < 3; ++i) {
      ring.try_push(i + 1);
    }

    int value = 0;
    while (ring.try_pop(value)) {
      std::cout << "  Popped: " << value << "\n";
    }
  }

  // Example 2: One producer thread streams to one consumer thread.
  std::cout << "\nExample 2: Producer/consumer handoff\n";
  {
    constexpr int kCount = 100000;
    dispenso::SPSCRingBuffer<int, 1024> ring;

    std::thread producer([&]() {
      for (int i = 0; i < kCount; ++i) {
        while (!ring.try_push(i)) {
          // Buffer full; wait for the consumer to make room.
        }
      }
    });

    long long sum = 0;
    int received = 0;
    int value = 0;
    while (received < kCount) {
      if (ring.try_pop(value)) {
        sum += value;
        ++received;
      }
    }

    producer.join();
    // Sum of 0..kCount-1 = kCount*(kCount-1)/2.
    const long long expected = static_cast<long long>(kCount) * (kCount - 1) / 2;
    std::cout << "  Received " << received << " items, sum = " << sum << " (expected " << expected
              << ")\n";
  }

  std::cout << "\nAll SPSCRingBuffer examples completed successfully!\n";
  return 0;
}
