/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example parallel_for_example.cpp
 * Demonstrates basic usage of dispenso::parallel_for for parallel loop execution.
 */

#include <dispenso/parallel_for.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  constexpr size_t kArraySize = 1000000;

  // Create input and output vectors
  std::vector<double> input(kArraySize);
  std::vector<double> output(kArraySize);

  // Initialize input data
  for (size_t i = 0; i < kArraySize; ++i) {
    input[i] = static_cast<double>(i);
  }

  // Example 1: Simple parallel_for with index-based lambda
  // Process each element independently in parallel
  std::cout << "Example 1: Simple parallel_for with per-element lambda\n";
  dispenso::parallel_for(0, kArraySize, [&](size_t i) { output[i] = std::sqrt(input[i]); });

  std::cout << "  output[0] = " << output[0] << ", output[999999] = " << output[999999] << "\n";

  // Example 2: parallel_for with range-based lambda for better cache utilization
  // The lambda receives start and end indices for a chunk (automatic chunking)
  std::cout << "\nExample 2: parallel_for with range-based lambda (chunked)\n";
  dispenso::parallel_for(size_t{0}, kArraySize, [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      output[i] = input[i] * 2.0;
    }
  });

  std::cout << "  output[0] = " << output[0] << ", output[999999] = " << output[999999] << "\n";

  // Example 3: parallel_for with per-thread state
  // Useful for reduction operations or when threads need local accumulators
  std::cout << "\nExample 3: parallel_for with per-thread state (reduction)\n";
  std::vector<double> partialSums;
  dispenso::parallel_for(
      partialSums,
      []() { return 0.0; }, // State initializer
      size_t{0},
      kArraySize,
      [&](double& localSum, size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          localSum += input[i];
        }
      });

  // Combine partial sums
  double totalSum = 0.0;
  for (double partial : partialSums) {
    totalSum += partial;
  }
  std::cout << "  Sum of all elements: " << totalSum << "\n";

  // Example 4: parallel_for with options to limit parallelism
  std::cout << "\nExample 4: parallel_for with limited parallelism\n";
  dispenso::ParForOptions options;
  options.maxThreads = 2; // Limit to 2 threads
  dispenso::parallel_for(
      0,
      100,
      [](size_t i) {
        // Light work that doesn't need many threads
        (void)i;
      },
      options);
  std::cout << "  Completed with maxThreads = 2\n";

  // Example 5: kAdaptive chunking for uneven per-iteration work.
  // The default (kStatic) splits the range into equal chunks, which is ideal
  // when every iteration costs about the same. When some iterations are much
  // more expensive than others, kAdaptive lets idle workers steal remaining
  // work so no thread is left waiting on a slow chunk.
  std::cout << "\nExample 5: kAdaptive chunking for uneven work\n";
  {
    std::vector<uint64_t> work(1000);
    dispenso::ParForOptions adaptiveOptions;
    adaptiveOptions.defaultChunking = dispenso::ParForChunking::kAdaptive;
    dispenso::parallel_for(
        0,
        work.size(),
        [&work](size_t i) {
          // Later iterations cost more, so equal static chunks would imbalance.
          uint64_t acc = 0;
          for (size_t k = 0; k <= i; ++k) {
            acc += k;
          }
          work[i] = acc;
        },
        adaptiveOptions);
    std::cout << "  Completed adaptive loop; work[999] = " << work[999] << "\n";
  }

  std::cout << "\nAll parallel_for examples completed successfully!\n";
  return 0;
}
