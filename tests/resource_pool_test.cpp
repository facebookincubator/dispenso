// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <dispenso/resource_pool.h>
#include <dispenso/thread_pool.h>

namespace {
// In real use cases, the buffer may perform some expensive initialization such as allocate a large
// chunk of memory.
struct Buffer {
  Buffer(std::atomic_int& _total_count, std::atomic_int& _num_buffers)
      : total_count(_total_count), num_buffers(_num_buffers), count(0) {}
  // On destruction, add the count
  ~Buffer() {
    total_count += count;
    num_buffers += 1;
  }
  std::atomic_int& total_count;
  std::atomic_int& num_buffers;
  int count;
};

void BuffersTest(const int num_threads, const int num_buffers) {
  constexpr int kNumTasks = 100000;
  std::atomic_int total_count(0);
  std::atomic_int num_buffers_created(0);
  {
    dispenso::ResourcePool<Buffer> buffer_pool(num_buffers, [&total_count, &num_buffers_created]() {
      return Buffer(total_count, num_buffers_created);
    });
    dispenso::ThreadPool thread_pool(num_threads);
    for (int i = 0; i < kNumTasks; ++i) {
      thread_pool.schedule([&]() {
        auto buffer_resource = buffer_pool.acquire();
        ++buffer_resource.get().count;
      });
    }
  }

  // The sum of all the buffers counts should be equal to the number of tasks.
  EXPECT_EQ(total_count, kNumTasks);
  EXPECT_EQ(num_buffers_created, num_buffers);
}

} // namespace

TEST(ResourcePool, SameNumBuffersAsThreadsTest) {
  constexpr int kNumBuffers = 2;
  constexpr int kNumThreads = 2;
  BuffersTest(kNumBuffers, kNumThreads);
}

TEST(ResourcePool, FewerBuffersThanThreadsTest) {
  constexpr int kNumBuffers = 1;
  constexpr int kNumThreads = 2;
  BuffersTest(kNumBuffers, kNumThreads);
}

TEST(ResourcePool, MoreBuffersThanThreadsTest) {
  constexpr int kNumBuffers = 2;
  constexpr int kNumThreads = 1;
  BuffersTest(kNumBuffers, kNumThreads);
}
