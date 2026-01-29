/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example pipeline_example.cpp
 * Demonstrates multi-stage pipeline processing with dispenso::pipeline.
 */

#include <dispenso/pipeline.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  // Example 1: Simple 3-stage pipeline (generator -> transform -> sink)
  std::cout << "Example 1: Simple 3-stage pipeline\n";
  {
    std::vector<int> results;
    int counter = 0;

    dispenso::pipeline(
        // Stage 1: Generator - produces values
        [&counter]() -> dispenso::OpResult<int> {
          if (counter >= 10) {
            return {}; // Empty result signals end of input
          }
          return counter++;
        },
        // Stage 2: Transform - squares the value
        [](int value) { return value * value; },
        // Stage 3: Sink - collects results
        [&results](int value) { results.push_back(value); });

    std::cout << "  Squared values: ";
    for (int r : results) {
      std::cout << r << " ";
    }
    std::cout << "\n";
  }

  // Example 2: Pipeline with parallel stages
  std::cout << "\nExample 2: Pipeline with parallel transform stage\n";
  {
    std::vector<double> results;
    int counter = 0;

    dispenso::pipeline(
        // Generator (serial)
        [&counter]() -> dispenso::OpResult<int> {
          if (counter >= 100) {
            return {};
          }
          return counter++;
        },
        // Transform (parallel with limit of 4 concurrent operations)
        dispenso::stage(
            [](int value) {
              // Simulate expensive computation
              return std::sqrt(static_cast<double>(value));
            },
            4),
        // Sink (serial)
        [&results](double value) { results.push_back(value); });

    std::cout << "  First 5 sqrt results: ";
    for (size_t i = 0; i < 5 && i < results.size(); ++i) {
      std::cout << results[i] << " ";
    }
    std::cout << "...\n";
    std::cout << "  Total results: " << results.size() << "\n";
  }

  // Example 3: Pipeline with filtering
  std::cout << "\nExample 3: Pipeline with filtering (keep only even numbers)\n";
  {
    std::vector<int> results;
    int counter = 0;

    dispenso::pipeline(
        // Generator
        [&counter]() -> dispenso::OpResult<int> {
          if (counter >= 20) {
            return {};
          }
          return counter++;
        },
        // Filter: only pass through even numbers
        [](int value) -> dispenso::OpResult<int> {
          if (value % 2 == 0) {
            return value;
          }
          return {}; // Filter out odd numbers
        },
        // Sink
        [&results](int value) { results.push_back(value); });

    std::cout << "  Even numbers: ";
    for (int r : results) {
      std::cout << r << " ";
    }
    std::cout << "\n";
  }

  // Example 4: Pipeline with type transformation
  std::cout << "\nExample 4: Pipeline with type transformations\n";
  {
    std::vector<std::string> results;
    int counter = 0;

    dispenso::pipeline(
        // Generate integers
        [&counter]() -> dispenso::OpResult<int> {
          if (counter >= 5) {
            return {};
          }
          return counter++;
        },
        // Transform to double
        [](int value) { return static_cast<double>(value) * 1.5; },
        // Transform to string
        [](double value) {
          std::ostringstream oss;
          oss << "Value: " << value;
          return oss.str();
        },
        // Collect strings
        [&results](std::string value) { results.push_back(std::move(value)); });

    std::cout << "  String results:\n";
    for (const auto& r : results) {
      std::cout << "    " << r << "\n";
    }
  }

  // Example 5: Pipeline with custom thread pool
  std::cout << "\nExample 5: Pipeline with custom ThreadPool\n";
  {
    dispenso::ThreadPool customPool(2);
    std::vector<int> results;
    int counter = 0;

    dispenso::pipeline(
        customPool,
        // Generator
        [&counter]() -> dispenso::OpResult<int> {
          if (counter >= 10) {
            return {};
          }
          return counter++;
        },
        // Parallel transform
        dispenso::stage([](int value) { return value + 100; }, dispenso::kStageNoLimit),
        // Sink
        [&results](int value) { results.push_back(value); });

    std::cout << "  Results from custom pool: ";
    for (int r : results) {
      std::cout << r << " ";
    }
    std::cout << "\n";
  }

  // Example 6: Single-stage pipeline (just a generator loop)
  std::cout << "\nExample 6: Single-stage pipeline\n";
  {
    int sum = 0;
    int counter = 0;

    // Single stage that returns bool (true = continue, false = stop)
    dispenso::pipeline([&]() -> bool {
      if (counter >= 10) {
        return false;
      }
      sum += counter++;
      return true;
    });

    std::cout << "  Sum computed in single stage: " << sum << " (expected: 45)\n";
  }

  std::cout << "\nAll Pipeline examples completed successfully!\n";
  return 0;
}
