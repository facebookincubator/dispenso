/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example parallel_invoke_example.cpp
 * Demonstrates launching heterogeneous tasks in parallel with dispenso::parallel_invoke.
 */

#include <dispenso/parallel_invoke.h>
#include <dispenso/thread_pool.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  // Example 1: Run two independent computations in parallel
  std::cout << "Example 1: Two heterogeneous tasks in parallel\n";
  {
    double sqrtResult = 0.0;
    double logResult = 0.0;

    dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
    dispenso::parallel_invoke(
        tasks,
        [&sqrtResult]() {
          // Heavy computation 1
          for (int i = 1; i <= 100000; ++i) {
            sqrtResult += std::sqrt(static_cast<double>(i));
          }
        },
        [&logResult]() {
          // Heavy computation 2
          for (int i = 1; i <= 100000; ++i) {
            logResult += std::log(static_cast<double>(i));
          }
        });
    tasks.wait();

    std::cout << "  sqrt sum: " << sqrtResult << "\n";
    std::cout << "  log sum:  " << logResult << "\n";
  }

  // Example 2: Three-way parallel fork (different work per branch)
  std::cout << "\nExample 2: Three-way parallel fork\n";
  {
    std::vector<int> a(1000), b(1000), c(1000);

    dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
    dispenso::parallel_invoke(
        tasks,
        [&a]() { std::iota(a.begin(), a.end(), 0); },
        [&b]() { std::iota(b.begin(), b.end(), 1000); },
        [&c]() { std::iota(c.begin(), c.end(), 2000); });
    tasks.wait();

    std::cout << "  a[0]=" << a[0] << " a[999]=" << a[999] << "\n";
    std::cout << "  b[0]=" << b[0] << " b[999]=" << b[999] << "\n";
    std::cout << "  c[0]=" << c[0] << " c[999]=" << c[999] << "\n";
  }

  // Example 3: Recursive divide-and-conquer (parallel merge sort sketch)
  std::cout << "\nExample 3: Recursive divide-and-conquer with parallel_invoke\n";
  {
    std::vector<int> data = {8, 3, 1, 7, 0, 10, 2, 9, 5, 4, 6, 11};
    dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());

    // Sort two halves in parallel, then merge
    size_t mid = data.size() / 2;
    dispenso::parallel_invoke(
        tasks,
        [&data, mid]() { std::sort(data.begin(), data.begin() + mid); },
        [&data, mid]() { std::sort(data.begin() + mid, data.end()); });
    tasks.wait();

    // Merge the two sorted halves
    std::inplace_merge(data.begin(), data.begin() + mid, data.end());

    std::cout << "  Sorted: ";
    for (int v : data) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nAll parallel_invoke examples completed successfully!\n";
  return 0;
}
