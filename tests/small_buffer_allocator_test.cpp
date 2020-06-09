// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/small_buffer_allocator.h>

#include <deque>

#include <gtest/gtest.h>

constexpr size_t kSmall = 32;
constexpr size_t kMedium = 128;
constexpr size_t kLarge = 256;

using dispenso::SmallBufferAllocator;

template <size_t kSize>
void testEmpty() {
  ASSERT_EQ(SmallBufferAllocator<kSize>::bytesAllocated(), 0);
  SmallBufferAllocator<kSize>::dealloc(SmallBufferAllocator<kSize>::alloc());
  ASSERT_GT(SmallBufferAllocator<kSize>::bytesAllocated(), 0);
}

TEST(SmallBufferAllocator, Empty) {
  testEmpty<kSmall>();
  testEmpty<kMedium>();
  testEmpty<kLarge>();
}

template <size_t kSize>
void testSimple() {
  std::vector<char*> buffers(1 << 20);
  auto doIt = [&buffers]() {
    for (char*& b : buffers) {
      b = SmallBufferAllocator<kSize>::alloc();
    }
    for (char* b : buffers) {
      SmallBufferAllocator<kSize>::dealloc(b);
    }
  };
  // Warm up to allocate.
  doIt();

  size_t allocatedSoFar = SmallBufferAllocator<kSize>::bytesAllocated();

  for (int i = 0; i < 10; ++i) {
    doIt();
    ASSERT_EQ(SmallBufferAllocator<kSize>::bytesAllocated(), allocatedSoFar);
  }
}
TEST(SmallBufferAllocator, Simple) {
  testSimple<kSmall>();
  testSimple<kMedium>();
  testSimple<kLarge>();
}

template <size_t kSize>
void testThreads() {
  constexpr int kThreads = 8;
  std::vector<std::vector<char*>> threadBuffers(kThreads);
  std::deque<std::thread> threads;
  for (auto& tb : threadBuffers) {
    tb.resize(1 << 20);
    threads.emplace_back([& buffers = tb]() {
      for (char*& b : buffers) {
        b = SmallBufferAllocator<kSize>::alloc();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  int i = 0;
  for (auto& tb : threadBuffers) {
    threads[i++] = std::thread([& buffers = tb]() {
      for (char* b : buffers) {
        SmallBufferAllocator<kSize>::dealloc(b);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  size_t allocatedSoFar = SmallBufferAllocator<kSize>::bytesAllocated();

  for (int j = 0; j < 50; ++j) {
    i = 0;
    for (auto& tb : threadBuffers) {
      threads[i++] = std::thread([& buffers = tb]() {
        for (char*& b : buffers) {
          b = SmallBufferAllocator<kSize>::alloc();
        }
        for (char* b : buffers) {
          SmallBufferAllocator<kSize>::dealloc(b);
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }
    ASSERT_LE(SmallBufferAllocator<kSize>::bytesAllocated(), 1.2 * allocatedSoFar);
  }
}

TEST(SmallBufferAllocator, ThreadsSimple) {
  testThreads<kSmall>();
  testThreads<kMedium>();
  testThreads<kLarge>();
}

template <size_t kSize>
void testThreadsHandoff() {
  constexpr int kThreads = 8;
  std::vector<std::deque<std::pair<char*, std::atomic<bool>>>> threadBuffers(kThreads);
  std::deque<std::thread> threads;
  for (auto& tb : threadBuffers) {
    for (int i = 0; i < 10000; ++i) {
      tb.emplace_back();
    }
    threads.emplace_back([& buffers = tb]() {
      for (auto& b : buffers) {
        b.first = SmallBufferAllocator<kSize>::alloc();
        b.second.store(false, std::memory_order_relaxed);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  int i = 0;
  for (auto& tb : threadBuffers) {
    threads[i++] = std::thread([& buffers = tb]() {
      for (auto& b : buffers) {
        SmallBufferAllocator<kSize>::dealloc(b.first);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  size_t allocatedSoFar = SmallBufferAllocator<kSize>::bytesAllocated();

  for (int j = 0; j < 50; ++j) {
    for (i = 0; i < kThreads; ++i) {
      threads[i] = std::thread(
          [& buffers = threadBuffers[i], &buffersOther = threadBuffers[(i + 1) % kThreads]]() {
            for (auto& b : buffers) {
              b.first = SmallBufferAllocator<kSize>::alloc();
              b.second.store(true, std::memory_order_release);
            }
            for (auto& b : buffersOther) {
              while (!b.second.load(std::memory_order_acquire)) {
                std::this_thread::yield();
              }
              SmallBufferAllocator<kSize>::dealloc(b.first);
              b.second.store(false, std::memory_order_relaxed);
            }
          });
    }

    for (auto& t : threads) {
      t.join();
    }
    ASSERT_LE(SmallBufferAllocator<kSize>::bytesAllocated(), 1.05 * allocatedSoFar);
  }
}

TEST(SmallBufferAllocator, ThreadsHandoff) {
  testThreadsHandoff<kSmall>();
  testThreadsHandoff<kMedium>();
  testThreadsHandoff<kLarge>();
}
