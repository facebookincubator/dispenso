/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example concurrent_vector_example.cpp
 * Demonstrates thread-safe vector operations with dispenso::ConcurrentVector.
 */

#include <dispenso/concurrent_vector.h>
#include <dispenso/parallel_for.h>

#include <algorithm>
#include <iostream>
#include <numeric>

int main() {
  // Example 1: Basic concurrent push_back
  std::cout << "Example 1: Concurrent push_back from multiple threads\n";
  {
    dispenso::ConcurrentVector<int> vec;

    // Push elements concurrently
    dispenso::parallel_for(0, 1000, [&vec](size_t i) { vec.push_back(static_cast<int>(i)); });

    std::cout << "  Vector size after concurrent pushes: " << vec.size() << "\n";
    std::cout << "  (Note: order may vary due to concurrent access)\n";
  }

  // Example 2: grow_by for batch insertion
  std::cout << "\nExample 2: grow_by for efficient batch insertion\n";
  {
    dispenso::ConcurrentVector<double> vec;

    // Each thread grows the vector by a batch
    dispenso::parallel_for(0, 10, [&vec](size_t threadId) {
      auto it = vec.grow_by(100, static_cast<double>(threadId));
      (void)it; // Iterator points to start of grown range
    });

    std::cout << "  Vector size: " << vec.size() << " (expected: 1000)\n";

    // Count occurrences
    std::array<int, 10> counts = {0};
    for (double val : vec) {
      counts[static_cast<size_t>(val)]++;
    }
    std::cout << "  Each thread's value appears 100 times: ";
    std::cout << (std::all_of(counts.begin(), counts.end(), [](int c) { return c == 100; }) ? "yes"
                                                                                            : "no")
              << "\n";
  }

  // Example 3: Using grow_by_generator
  std::cout << "\nExample 3: grow_by_generator with custom initialization\n";
  {
    dispenso::ConcurrentVector<int> vec;

    // Grow with a generator function
    int startValue = 0;
    vec.grow_by_generator(10, [&startValue]() { return startValue++; });

    std::cout << "  Generated values: ";
    for (int val : vec) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  // Example 4: grow_to_at_least
  std::cout << "\nExample 4: grow_to_at_least for minimum size guarantee\n";
  {
    dispenso::ConcurrentVector<int> vec;

    // Multiple threads try to ensure minimum size
    dispenso::parallel_for(0, 10, [&vec](size_t i) {
      size_t requiredSize = (i + 1) * 100;
      vec.grow_to_at_least(requiredSize, -1);
    });

    std::cout << "  Vector size: " << vec.size() << " (at least 1000)\n";
  }

  // Example 5: Reserving capacity with ReserveTag
  std::cout << "\nExample 5: Reserving capacity for better performance\n";
  {
    // Reserve space upfront for better memory allocation
    dispenso::ConcurrentVector<int> vec(10000, dispenso::ReserveTag);

    std::cout << "  Capacity after reserve: " << vec.capacity() << "\n";
    std::cout << "  Size (still empty): " << vec.size() << "\n";

    // Now concurrent operations will be faster
    dispenso::parallel_for(0, 5000, [&vec](size_t i) { vec.push_back(static_cast<int>(i)); });

    std::cout << "  Size after pushes: " << vec.size() << "\n";
  }

  // Example 6: Iterator stability
  std::cout << "\nExample 6: Iterator stability during concurrent modification\n";
  {
    dispenso::ConcurrentVector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    // Get iterators to existing elements
    auto it = vec.begin();
    int& firstElement = *it;

    // Push more elements concurrently
    dispenso::parallel_for(0, 100, [&vec](size_t i) { vec.push_back(static_cast<int>(i + 100)); });

    // Original iterator and reference are still valid
    std::cout << "  First element via original iterator: " << *it << "\n";
    std::cout << "  First element via original reference: " << firstElement << "\n";
    std::cout << "  Vector size now: " << vec.size() << "\n";
  }

  // Example 7: Standard container operations
  std::cout << "\nExample 7: Standard container operations\n";
  {
    dispenso::ConcurrentVector<int> vec = {5, 3, 1, 4, 2};

    // Sorting (not concurrent - use single thread)
    std::sort(vec.begin(), vec.end());

    std::cout << "  Sorted: ";
    for (int val : vec) {
      std::cout << val << " ";
    }
    std::cout << "\n";

    // Access by index
    std::cout << "  vec[2] = " << vec[2] << "\n";

    // Front and back
    std::cout << "  front() = " << vec.front() << ", back() = " << vec.back() << "\n";
  }

  // Example 8: Copy and move semantics
  std::cout << "\nExample 8: Copy and move semantics\n";
  {
    dispenso::ConcurrentVector<int> original = {1, 2, 3, 4, 5};

    // Copy
    dispenso::ConcurrentVector<int> copied = original;
    std::cout << "  Original size: " << original.size() << ", Copy size: " << copied.size() << "\n";

    // Move
    dispenso::ConcurrentVector<int> moved = std::move(original);
    std::cout << "  After move - Original size: " << original.size()
              << ", Moved size: " << moved.size() << "\n";
  }

  std::cout << "\nAll ConcurrentVector examples completed successfully!\n";
  return 0;
}
