/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example task_set_example.cpp
 * Demonstrates scheduling tasks and waiting for completion with TaskSet.
 */

#include <dispenso/task_set.h>
#include <dispenso/thread_pool.h>

#include <atomic>
#include <iostream>
#include <vector>

int main() {
  // Example 1: Basic TaskSet usage
  std::cout << "Example 1: Basic TaskSet with simple tasks\n";
  {
    dispenso::TaskSet taskSet(dispenso::globalThreadPool());

    std::atomic<int> counter(0);

    // Schedule several tasks
    for (int i = 0; i < 10; ++i) {
      taskSet.schedule([&counter, i]() { counter.fetch_add(i, std::memory_order_relaxed); });
    }

    // Wait for all tasks to complete
    taskSet.wait();

    std::cout << "  Sum of 0..9 = " << counter.load() << " (expected: 45)\n";
  }

  // Example 2: ConcurrentTaskSet for multi-threaded scheduling
  std::cout << "\nExample 2: ConcurrentTaskSet (schedule from multiple threads)\n";
  {
    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    std::atomic<int> total(0);

    // Schedule tasks that themselves schedule more tasks
    for (int i = 0; i < 5; ++i) {
      taskSet.schedule([&taskSet, &total, i]() {
        // Each task schedules two sub-tasks
        for (int j = 0; j < 2; ++j) {
          taskSet.schedule(
              [&total, i, j]() { total.fetch_add(i * 10 + j, std::memory_order_relaxed); });
        }
      });
    }

    taskSet.wait();

    std::cout << "  Total from nested scheduling = " << total.load() << "\n";
  }

  // Example 3: Using a custom thread pool
  std::cout << "\nExample 3: TaskSet with custom ThreadPool\n";
  {
    // Create a small thread pool with 2 threads
    dispenso::ThreadPool customPool(2);

    dispenso::TaskSet taskSet(customPool);

    std::vector<int> results(4, 0);

    for (size_t i = 0; i < results.size(); ++i) {
      taskSet.schedule([&results, i]() { results[i] = static_cast<int>(i * i); });
    }

    taskSet.wait();

    std::cout << "  Squares: ";
    for (int r : results) {
      std::cout << r << " ";
    }
    std::cout << "\n";
  }

  // Example 4: TaskSet cancellation
  std::cout << "\nExample 4: TaskSet cancellation\n";
  {
    dispenso::TaskSet taskSet(dispenso::globalThreadPool());

    std::atomic<int> completed(0);
    std::atomic<int> skipped(0);

    // Schedule many tasks, but cancel after scheduling some
    for (int i = 0; i < 1000; ++i) {
      taskSet.schedule([&taskSet, &completed, &skipped]() {
        if (taskSet.canceled()) {
          skipped.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        completed.fetch_add(1, std::memory_order_relaxed);
      });

      // Cancel after scheduling 100 tasks
      if (i == 100) {
        taskSet.cancel();
      }
    }

    bool wasCanceled = taskSet.wait();

    std::cout << "  Was canceled: " << (wasCanceled ? "yes" : "no") << "\n";
    std::cout << "  Completed tasks: " << completed.load() << "\n";
    std::cout << "  Skipped tasks: " << skipped.load() << "\n";
  }

  // Example 5: Checking canceled status within tasks
  std::cout << "\nExample 5: Early exit from tasks using canceled()\n";
  {
    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

    std::atomic<int> itemsProcessed(0);

    // Schedule a task that processes items and checks for cancellation
    taskSet.schedule([&taskSet, &itemsProcessed]() {
      for (int i = 0; i < 1000; ++i) {
        if (taskSet.canceled()) {
          // Exit early if canceled
          break;
        }
        itemsProcessed.fetch_add(1, std::memory_order_relaxed);
      }
    });

    // Cancel from another task after processing starts
    taskSet.schedule([&taskSet]() { taskSet.cancel(); });

    taskSet.wait();

    std::cout << "  Items processed before cancellation: " << itemsProcessed.load() << "\n";
  }

  std::cout << "\nAll TaskSet examples completed successfully!\n";
  return 0;
}
