// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/small_buffer_allocator.h>

#include <deque>

#include <dispenso/tsan_annotations.h>
#include <gtest/gtest.h>

constexpr size_t kSmall = 32;
constexpr size_t kMedium = 128;
constexpr size_t kLarge = 256;

constexpr size_t kSimpleNumBuffers = 1 << 15;
constexpr size_t kThreadedNumBuffers = 1 << 12;

// Note that this multiplier must be larger when we have  fewer buffer.  That is because there is
// more slop-space in the SmallBufferAllocator relative to what is allocated.  If we change the
// kThreadedNumBuffers, we will need to change this multiplier.  Additionally, TSAN very rarely will
// push this overhead over 2, so we have to account for that here.  Without TSAN, this should be
// astronomically unlikely (e.g. a gamma ray might cause the test to fail also, but we don't account
// for that).
#if !DISPENSO_HAS_TSAN
constexpr size_t kMaxOverheadMultiplier = 2;
#else
constexpr size_t kMaxOverheadMultiplier = 3;
#endif // !DISPENSO_HAS_TSAN

using dispenso::allocSmallBuffer;
using dispenso::approxBytesAllocatedSmallBuffer;
using dispenso::deallocSmallBuffer;

template <size_t kSize>
void testEmpty() {
  ASSERT_EQ(approxBytesAllocatedSmallBuffer<kSize>(), 0);
  deallocSmallBuffer<kSize>(allocSmallBuffer<kSize>());
  ASSERT_GT(approxBytesAllocatedSmallBuffer<kSize>(), 0);
}

TEST(SmallBufferAllocator, Empty) {
  // It can be useful to run gtest with --gtest_repeat option.  Since repeats run in the same
  // process, the allocator will no longer be empty after the first iteration.  Guard against this
  // here.
  static bool firstTime = true;
  if (firstTime) {
    firstTime = false;
    testEmpty<kSmall>();
    testEmpty<kMedium>();
    testEmpty<kLarge>();
  }
}

template <size_t kSize>
void testSimple() {
  std::vector<char*> buffers(kSimpleNumBuffers);
  auto doIt = [&buffers]() {
    for (char*& b : buffers) {
      b = allocSmallBuffer<kSize>();
    }
    for (char* b : buffers) {
      deallocSmallBuffer<kSize>(b);
    }
  };
  // Warm up to allocate.
  doIt();

  size_t allocatedSoFar = approxBytesAllocatedSmallBuffer<kSize>();

  for (int i = 0; i < 10; ++i) {
    doIt();
    ASSERT_EQ(approxBytesAllocatedSmallBuffer<kSize>(), allocatedSoFar);
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
    tb.resize(kThreadedNumBuffers);
    threads.emplace_back([&buffers = tb]() {
      for (char*& b : buffers) {
        b = allocSmallBuffer<kSize>();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  int i = 0;
  for (auto& tb : threadBuffers) {
    threads[i++] = std::thread([&buffers = tb]() {
      for (char* b : buffers) {
        deallocSmallBuffer<kSize>(b);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  size_t allocatedSoFar = approxBytesAllocatedSmallBuffer<kSize>();

  for (int j = 0; j < 50; ++j) {
    i = 0;
    for (auto& tb : threadBuffers) {
      threads[i++] = std::thread([&buffers = tb]() {
        for (char*& b : buffers) {
          b = allocSmallBuffer<kSize>();
        }
        for (char* b : buffers) {
          deallocSmallBuffer<kSize>(b);
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    ASSERT_LE(approxBytesAllocatedSmallBuffer<kSize>(), kMaxOverheadMultiplier * allocatedSoFar);
  }
}

TEST(SmallBufferAllocator, ThreadsSimpleSmall) {
  testThreads<kSmall>();
}

TEST(SmallBufferAllocator, ThreadsSimpleMedium) {
  testThreads<kMedium>();
}

TEST(SmallBufferAllocator, ThreadsSimpleLarge) {
  testThreads<kLarge>();
}

template <size_t kSize>
void testThreadsHandoff() {
  constexpr int kThreads = 8;
  std::vector<std::deque<std::pair<char*, std::atomic<bool>>>> threadBuffers(kThreads);
  std::deque<std::thread> threads;
  for (auto& tb : threadBuffers) {
    tb.resize(kThreadedNumBuffers);
    threads.emplace_back([&buffers = tb]() {
      for (auto& b : buffers) {
        b.first = allocSmallBuffer<kSize>();
        b.second.store(false, std::memory_order_relaxed);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  int i = 0;
  for (auto& tb : threadBuffers) {
    threads[i++] = std::thread([&buffers = tb]() {
      for (auto& b : buffers) {
        deallocSmallBuffer<kSize>(b.first);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  size_t allocatedSoFar = approxBytesAllocatedSmallBuffer<kSize>();

  for (int j = 0; j < 50; ++j) {
    for (i = 0; i < kThreads; ++i) {
      threads[i] = std::thread(
          [&buffers = threadBuffers[i], &buffersOther = threadBuffers[(i + 1) % kThreads]]() {
            for (auto& b : buffers) {
              b.first = allocSmallBuffer<kSize>();
              b.second.store(true, std::memory_order_release);
            }
            for (auto& b : buffersOther) {
              while (!b.second.load(std::memory_order_acquire)) {
                std::this_thread::yield();
              }
              deallocSmallBuffer<kSize>(b.first);
              b.second.store(false, std::memory_order_relaxed);
            }
          });
    }

    for (auto& t : threads) {
      t.join();
    }

    ASSERT_LE(approxBytesAllocatedSmallBuffer<kSize>(), kMaxOverheadMultiplier * allocatedSoFar);
  }
}

TEST(SmallBufferAllocator, ThreadsHandoffSmall) {
  testThreadsHandoff<kSmall>();
}

TEST(SmallBufferAllocator, ThreadsHandoffMedium) {
  testThreadsHandoff<kMedium>();
}

TEST(SmallBufferAllocator, ThreadsHandoffLarge) {
  testThreadsHandoff<kLarge>();
}
