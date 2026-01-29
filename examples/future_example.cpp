/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example future_example.cpp
 * Demonstrates creating and chaining futures with dispenso::Future.
 */

#include <dispenso/future.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

int main() {
  // Example 1: Basic async execution
  std::cout << "Example 1: Basic async execution\n";
  {
    // Launch an async computation
    dispenso::Future<int> future = dispenso::async([]() {
      // Simulate some work
      int result = 0;
      for (int i = 1; i <= 100; ++i) {
        result += i;
      }
      return result;
    });

    // Do other work while computation runs...
    std::cout << "  Computing sum of 1..100 asynchronously...\n";

    // Get the result (blocks until ready)
    int result = future.get();
    std::cout << "  Result: " << result << " (expected: 5050)\n";
  }

  // Example 2: Chaining futures with then()
  std::cout << "\nExample 2: Chaining futures with then()\n";
  {
    dispenso::Future<double> chainedFuture = dispenso::async([]() {
                                               return 16.0; // First computation
                                             })
                                                 .then([](dispenso::Future<double>&& prev) {
                                                   return std::sqrt(prev.get()); // Chain: take sqrt
                                                 })
                                                 .then([](dispenso::Future<double>&& prev) {
                                                   return prev.get() * 2.0; // Chain: multiply by 2
                                                 });

    std::cout << "  sqrt(16) * 2 = " << chainedFuture.get() << " (expected: 8)\n";
  }

  // Example 3: make_ready_future for immediate values
  std::cout << "\nExample 3: make_ready_future for immediate values\n";
  {
    // Create a future that's already ready
    dispenso::Future<std::string> ready = dispenso::make_ready_future(std::string("Hello, World!"));

    std::cout << "  is_ready: " << (ready.is_ready() ? "true" : "false") << "\n";
    std::cout << "  Value: " << ready.get() << "\n";
  }

  // Example 4: wait_for with timeout
  std::cout << "\nExample 4: wait_for with timeout\n";
  {
    dispenso::Future<int> slowFuture = dispenso::async(std::launch::async, []() {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      return 42;
    });

    // Try to wait with a short timeout
    auto status = slowFuture.wait_for(std::chrono::milliseconds(10));
    std::cout << "  After 10ms wait: "
              << (status == std::future_status::ready ? "ready" : "timeout") << "\n";

    // Wait for completion
    slowFuture.wait();
    std::cout << "  After full wait: " << slowFuture.get() << "\n";
  }

  // Example 5: when_all to wait for multiple futures
  std::cout << "\nExample 5: when_all for multiple futures\n";
  {
    dispenso::Future<int> f1 = dispenso::async([]() { return 10; });
    dispenso::Future<int> f2 = dispenso::async([]() { return 20; });
    dispenso::Future<int> f3 = dispenso::async([]() { return 30; });

    // Wait for all futures to complete
    auto allFutures = dispenso::when_all(std::move(f1), std::move(f2), std::move(f3));

    // Get the results
    auto tuple = allFutures.get();
    int sum = std::get<0>(tuple).get() + std::get<1>(tuple).get() + std::get<2>(tuple).get();

    std::cout << "  Sum of all futures: " << sum << " (expected: 60)\n";
  }

  // Example 6: Using Future with a custom thread pool
  std::cout << "\nExample 6: Future with custom ThreadPool\n";
  {
    dispenso::ThreadPool customPool(2);

    dispenso::Future<double> future = dispenso::async(customPool, []() { return 3.14159; });

    std::cout << "  Pi from custom pool: " << future.get() << "\n";
  }

  // Example 7: Future<void> for side effects
  std::cout << "\nExample 7: Future<void> for side effects\n";
  {
    int sideEffectValue = 0;

    dispenso::Future<void> voidFuture = dispenso::async([&sideEffectValue]() {
      sideEffectValue = 123;
      // No return value
    });

    voidFuture.get(); // Wait for completion

    std::cout << "  Side effect value: " << sideEffectValue << "\n";
  }

  std::cout << "\nAll Future examples completed successfully!\n";
  return 0;
}
