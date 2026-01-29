/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example for_each_example.cpp
 * Demonstrates parallel iteration over containers using dispenso::for_each.
 */

#include <dispenso/for_each.h>

#include <cmath>
#include <iostream>
#include <list>
#include <vector>

int main() {
  // Example 1: for_each on a vector
  std::cout << "Example 1: for_each on a vector\n";
  std::vector<double> values = {1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0};

  // Apply square root to each element in parallel
  dispenso::for_each(values.begin(), values.end(), [](double& val) { val = std::sqrt(val); });

  std::cout << "  Square roots: ";
  for (double v : values) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  // Example 2: for_each on a list (non-contiguous memory)
  std::cout << "\nExample 2: for_each on a std::list\n";
  std::list<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Double each value in parallel
  dispenso::for_each(numbers.begin(), numbers.end(), [](int& n) { n *= 2; });

  std::cout << "  Doubled values: ";
  for (int n : numbers) {
    std::cout << n << " ";
  }
  std::cout << "\n";

  // Example 3: for_each_n with a count
  std::cout << "\nExample 3: for_each_n with explicit count\n";
  std::vector<int> partial = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  // Only process first 5 elements
  dispenso::for_each_n(partial.begin(), 5, [](int& n) { n += 100; });

  std::cout << "  After adding 100 to first 5: ";
  for (int n : partial) {
    std::cout << n << " ";
  }
  std::cout << "\n";

  // Example 4: for_each with a TaskSet for external synchronization
  std::cout << "\nExample 4: for_each with explicit TaskSet\n";
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());

  std::vector<int> data = {1, 2, 3, 4, 5};

  // Use options to control behavior
  dispenso::ForEachOptions options;
  options.wait = true;

  dispenso::for_each(taskSet, data.begin(), data.end(), [](int& n) { n = n * n; }, options);

  std::cout << "  Squared values: ";
  for (int n : data) {
    std::cout << n << " ";
  }
  std::cout << "\n";

  // Example 5: for_each with limited parallelism
  std::cout << "\nExample 5: for_each with limited parallelism\n";
  std::vector<int> limited(100, 1);

  dispenso::ForEachOptions limitedOptions;
  limitedOptions.maxThreads = 2;

  dispenso::for_each(limited.begin(), limited.end(), [](int& n) { n += 1; }, limitedOptions);

  std::cout << "  First few values: " << limited[0] << " " << limited[1] << " " << limited[2]
            << " (all should be 2)\n";

  std::cout << "\nAll for_each examples completed successfully!\n";
  return 0;
}
