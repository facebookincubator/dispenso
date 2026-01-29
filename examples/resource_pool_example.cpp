/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example resource_pool_example.cpp
 * Demonstrates managing pooled resources with dispenso::ResourcePool.
 */

#include <dispenso/parallel_for.h>
#include <dispenso/resource_pool.h>

#include <atomic>
#include <iostream>
#include <sstream>

// Example resource class - a reusable buffer
class Buffer {
 public:
  Buffer() : data_(1024, 0), useCount_(0) {}

  void process(int value) {
    // Simulate using the buffer
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = value + static_cast<int>(i);
    }
    useCount_++;
  }

  int checksum() const {
    int sum = 0;
    for (int val : data_) {
      sum += val;
    }
    return sum;
  }

  int useCount() const {
    return useCount_;
  }

 private:
  std::vector<int> data_;
  int useCount_;
};

// Example expensive-to-create resource
class DatabaseConnection {
 public:
  DatabaseConnection(int id) : connectionId_(id), queryCount_(0) {}

  int executeQuery(int param) {
    queryCount_++;
    return connectionId_ * 1000 + param;
  }

  int connectionId() const {
    return connectionId_;
  }
  int queryCount() const {
    return queryCount_;
  }

 private:
  int connectionId_;
  int queryCount_;
};

int main() {
  // Example 1: Basic resource pool usage
  std::cout << "Example 1: Basic ResourcePool usage\n";
  {
    // Create a pool of 4 buffers
    dispenso::ResourcePool<Buffer> bufferPool(4, []() { return Buffer(); });

    std::atomic<int> totalChecksum(0);

    // Use resources from multiple threads
    dispenso::parallel_for(0, 100, [&bufferPool, &totalChecksum](size_t i) {
      // Acquire a resource from the pool (blocks if none available)
      auto resource = bufferPool.acquire();

      // Use the resource
      resource.get().process(static_cast<int>(i));
      totalChecksum.fetch_add(resource.get().checksum(), std::memory_order_relaxed);

      // Resource automatically returned to pool when 'resource' goes out of scope
    });

    std::cout << "  Total checksum from 100 operations: " << totalChecksum.load() << "\n";
  }

  // Example 2: Resource pool with connection-like objects
  std::cout << "\nExample 2: Connection pool pattern\n";
  {
    std::atomic<int> nextConnectionId(0);

    // Create a pool of 3 "database connections"
    dispenso::ResourcePool<DatabaseConnection> connectionPool(3, [&nextConnectionId]() {
      return DatabaseConnection(nextConnectionId.fetch_add(1, std::memory_order_relaxed));
    });

    std::atomic<int> totalResult(0);

    dispenso::parallel_for(0, 50, [&connectionPool, &totalResult](size_t i) {
      auto conn = connectionPool.acquire();
      int result = conn.get().executeQuery(static_cast<int>(i));
      totalResult.fetch_add(result, std::memory_order_relaxed);
    });

    std::cout << "  Total query result sum: " << totalResult.load() << "\n";
    std::cout << "  (50 queries distributed across 3 connections)\n";
  }

  // Example 3: Resource pool limiting concurrency
  std::cout << "\nExample 3: Using ResourcePool to limit concurrency\n";
  {
    // Use a pool of "permits" to limit concurrent operations
    struct Permit {};

    constexpr size_t kMaxConcurrent = 2;
    dispenso::ResourcePool<Permit> permits(kMaxConcurrent, []() { return Permit(); });

    std::atomic<int> currentActive(0);
    std::atomic<int> maxObserved(0);

    dispenso::parallel_for(0, 100, [&](size_t i) {
      // Acquire permit (blocks if max concurrency reached)
      auto permit = permits.acquire();

      int active = currentActive.fetch_add(1, std::memory_order_relaxed) + 1;

      // Track max concurrent
      int maxSeen = maxObserved.load(std::memory_order_relaxed);
      while (active > maxSeen &&
             !maxObserved.compare_exchange_weak(maxSeen, active, std::memory_order_relaxed)) {
      }

      // Simulate work
      volatile int work = 0;
      for (int j = 0; j < 1000; ++j) {
        work += j;
      }
      (void)work;
      (void)i;

      currentActive.fetch_sub(1, std::memory_order_relaxed);
      // Permit returned when scope exits
    });

    std::cout << "  Max concurrent operations observed: " << maxObserved.load() << " (limit was "
              << kMaxConcurrent << ")\n";
  }

  // Example 4: Resource with expensive initialization
  std::cout << "\nExample 4: Resources with expensive initialization\n";
  {
    std::atomic<int> initCount(0);

    // Each resource is expensive to create, so we pool them
    dispenso::ResourcePool<std::stringstream> streamPool(2, [&initCount]() {
      initCount.fetch_add(1, std::memory_order_relaxed);
      // Simulate expensive setup
      std::stringstream ss;
      ss.precision(10);
      return ss;
    });

    dispenso::parallel_for(0, 20, [&streamPool](size_t i) {
      auto stream = streamPool.acquire();
      stream.get().str("");
      stream.get() << "Value: " << i * 3.14159;
      // Stream reused, not recreated
    });

    std::cout << "  Total initializations: " << initCount.load() << " (pool size: 2)\n";
    std::cout << "  (Only 2 streams were created, reused for 20 operations)\n";
  }

  // Example 5: Nested resource acquisition (be careful with deadlock!)
  std::cout << "\nExample 5: Sequential resource acquisition pattern\n";
  {
    dispenso::ResourcePool<int> poolA(2, []() { return 100; });
    dispenso::ResourcePool<int> poolB(2, []() { return 200; });

    std::atomic<int> resultSum(0);

    // Safe pattern: acquire resources sequentially, release in reverse order
    dispenso::parallel_for(0, 10, [&](size_t i) {
      auto resA = poolA.acquire();
      auto resB = poolB.acquire();

      resultSum.fetch_add(resA.get() + resB.get(), std::memory_order_relaxed);
      (void)i;

      // resB released first, then resA (LIFO order)
    });

    std::cout << "  Result sum: " << resultSum.load() << " (expected: 3000)\n";
  }

  std::cout << "\nAll ResourcePool examples completed successfully!\n";
  return 0;
}
