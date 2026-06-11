/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example small_vector_example.cpp
 * Demonstrates inline storage and heap fallback with dispenso::SmallVector.
 */

#include <dispenso/small_vector.h>

#include <algorithm>
#include <iostream>

int main() {
  // Example 1: Inline storage (no heap allocation for small sizes)
  std::cout << "Example 1: Inline storage (N=4)\n";
  {
    dispenso::SmallVector<int, 4> vec;

    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    std::cout << "  Size: " << vec.size() << ", Capacity: " << vec.capacity() << "\n";
    std::cout << "  Elements: ";
    for (int v : vec) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }

  // Example 2: Growing beyond inline capacity (spills to heap)
  std::cout << "\nExample 2: Growing beyond inline capacity\n";
  {
    dispenso::SmallVector<int, 4> vec;

    for (int i = 0; i < 10; ++i) {
      vec.push_back(i * 10);
    }

    std::cout << "  Size: " << vec.size() << ", Capacity: " << vec.capacity() << "\n";
    std::cout << "  Elements: ";
    for (int v : vec) {
      std::cout << v << " ";
    }
    std::cout << "\n";
    std::cout << "  front()=" << vec.front() << ", back()=" << vec.back() << "\n";
  }

  // Example 3: Initializer list and standard algorithms
  std::cout << "\nExample 3: Initializer list and sorting\n";
  {
    dispenso::SmallVector<int, 8> vec = {5, 2, 8, 1, 9, 3};

    std::sort(vec.begin(), vec.end());

    std::cout << "  Sorted: ";
    for (int v : vec) {
      std::cout << v << " ";
    }
    std::cout << "\n";

    // Index-based access
    std::cout << "  vec[0]=" << vec[0] << ", vec[5]=" << vec[5] << "\n";
  }

  // Example 4: Reserve and emplace_back
  std::cout << "\nExample 4: Reserve and emplace_back\n";
  {
    dispenso::SmallVector<std::pair<int, double>, 2> vec;

    // Reserve beyond inline capacity
    vec.reserve(8);
    std::cout << "  After reserve(8): capacity=" << vec.capacity() << ", size=" << vec.size()
              << "\n";

    vec.emplace_back(1, 3.14);
    vec.emplace_back(2, 2.72);
    vec.emplace_back(3, 1.41);

    std::cout << "  Pairs: ";
    for (const auto& p : vec) {
      std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << "\n";
  }

  // Example 5: Pop, erase, and clear
  std::cout << "\nExample 5: Pop, erase, and clear\n";
  {
    dispenso::SmallVector<int, 4> vec = {1, 2, 3, 4};

    vec.pop_back();
    std::cout << "  After pop_back: size=" << vec.size() << "\n";

    // Erase element at index 1
    vec.erase(vec.begin() + 1);
    std::cout << "  After erase(begin+1): ";
    for (int v : vec) {
      std::cout << v << " ";
    }
    std::cout << "\n";

    vec.clear();
    std::cout << "  After clear: size=" << vec.size() << ", empty=" << (vec.empty() ? "yes" : "no")
              << "\n";
  }

  // Example 6: Copy and move semantics
  std::cout << "\nExample 6: Copy and move semantics\n";
  {
    dispenso::SmallVector<int, 4> original = {10, 20, 30};

    // Copy
    dispenso::SmallVector<int, 4> copied = original;
    std::cout << "  Original size: " << original.size() << ", Copy size: " << copied.size() << "\n";

    // Move
    dispenso::SmallVector<int, 4> moved = std::move(original);
    std::cout << "  After move - Original size: " << original.size()
              << ", Moved size: " << moved.size() << "\n";
  }

  std::cout << "\nAll SmallVector examples completed successfully!\n";
  return 0;
}
