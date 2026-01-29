/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/util.h>

#include <dispenso/platform.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::ssize_t;

// Global test helpers for reference counting tests
int RefCountedCount = 0;
struct RefCounted {
  RefCounted() {
    ++RefCountedCount;
  }
  ~RefCounted() {
    --RefCountedCount;
  }
};

int NonTrivialConstructCount = 0;
int NonTrivialDestructCount = 0;
struct NonTrivial {
  int value;

  NonTrivial(int v) : value(v) {
    ++NonTrivialConstructCount;
  }

  NonTrivial(const NonTrivial& other) : value(other.value) {
    ++NonTrivialConstructCount;
  }

  NonTrivial(NonTrivial&& other) : value(other.value) {
    other.value = -1;
    ++NonTrivialConstructCount;
  }

  ~NonTrivial() {
    ++NonTrivialDestructCount;
  }
};

TEST(Util, AlignedMallocAndFree) {
  // Test basic allocation and deallocation
  void* ptr = dispenso::alignedMalloc(1024, 64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u) << "Pointer should be 64-byte aligned";
  dispenso::alignedFree(ptr);

  // Test cache-line aligned version
  ptr = dispenso::alignedMalloc(256);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % dispenso::kCacheLineSize, 0u)
      << "Pointer should be cache-line aligned";
  dispenso::alignedFree(ptr);

  // Test nullptr is safe to free
  dispenso::alignedFree(nullptr);

  // Test various alignments
  for (size_t alignment : {8, 16, 32, 64, 128, 256}) {
    ptr = dispenso::alignedMalloc(100, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0u)
        << "Pointer should be aligned to " << alignment;
    dispenso::alignedFree(ptr);
  }
}

TEST(Util, AlignedMallocUsable) {
  // Verify allocated memory is actually usable
  constexpr size_t kSize = 1024;
  uint8_t* ptr = static_cast<uint8_t*>(dispenso::alignedMalloc(kSize, 64));
  ASSERT_NE(ptr, nullptr);

  // Write pattern to memory
  for (size_t i = 0; i < kSize; ++i) {
    ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Verify pattern
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(ptr[i], static_cast<uint8_t>(i & 0xFF));
  }

  dispenso::alignedFree(ptr);
}

TEST(Util, AlignedDeleter) {
  // Test with unique_ptr
  {
    struct TestStruct {
      int value;
      ~TestStruct() {
        value = -1;
      }
    };

    void* mem = dispenso::alignedMalloc(sizeof(TestStruct), 64);
    ASSERT_NE(mem, nullptr);

    std::unique_ptr<TestStruct, dispenso::AlignedDeleter<TestStruct>> ptr(new (mem) TestStruct{42});
    EXPECT_EQ(ptr->value, 42);
  }
  // Deleter should have freed memory and called destructor

  // Test with shared_ptr - use global RefCounted
  RefCountedCount = 0;

  {
    void* mem = dispenso::alignedMalloc(sizeof(RefCounted), 64);
    std::shared_ptr<RefCounted> ptr1(
        new (mem) RefCounted(), dispenso::AlignedDeleter<RefCounted>());
    EXPECT_EQ(RefCountedCount, 1);

    std::shared_ptr<RefCounted> ptr2 = ptr1;
    EXPECT_EQ(RefCountedCount, 1);
  }

  EXPECT_EQ(RefCountedCount, 0);
}

TEST(Util, AlignToCacheLine) {
  // alignToCacheLine rounds up to next cache line boundary
  // For 0, it stays 0 (already aligned)
  EXPECT_EQ(dispenso::alignToCacheLine(0), 0u);
  EXPECT_EQ(dispenso::alignToCacheLine(1), dispenso::kCacheLineSize);
  EXPECT_EQ(dispenso::alignToCacheLine(63), dispenso::kCacheLineSize);
  EXPECT_EQ(dispenso::alignToCacheLine(64), dispenso::kCacheLineSize);
  EXPECT_EQ(dispenso::alignToCacheLine(65), dispenso::kCacheLineSize * 2);
  EXPECT_EQ(dispenso::alignToCacheLine(128), dispenso::kCacheLineSize * 2);
  EXPECT_EQ(dispenso::alignToCacheLine(129), dispenso::kCacheLineSize * 3);
}

TEST(Util, CpuRelax) {
  // Just verify it doesn't crash
  for (int i = 0; i < 10; ++i) {
    dispenso::cpuRelax();
  }
}

TEST(Util, NextPow2) {
  // Compile-time tests
  static_assert(dispenso::nextPow2(0) == 0, "nextPow2(0) should be 0");
  static_assert(dispenso::nextPow2(1) == 1, "nextPow2(1) should be 1");
  static_assert(dispenso::nextPow2(2) == 2, "nextPow2(2) should be 2");
  static_assert(dispenso::nextPow2(3) == 4, "nextPow2(3) should be 4");
  static_assert(dispenso::nextPow2(4) == 4, "nextPow2(4) should be 4");
  static_assert(dispenso::nextPow2(5) == 8, "nextPow2(5) should be 8");
  static_assert(dispenso::nextPow2(17) == 32, "nextPow2(17) should be 32");
  static_assert(dispenso::nextPow2(64) == 64, "nextPow2(64) should be 64");
  static_assert(dispenso::nextPow2(65) == 128, "nextPow2(65) should be 128");
  static_assert(dispenso::nextPow2(1000) == 1024, "nextPow2(1000) should be 1024");

  // Runtime tests
  EXPECT_EQ(dispenso::nextPow2(0), 0);
  EXPECT_EQ(dispenso::nextPow2(1), 1);
  EXPECT_EQ(dispenso::nextPow2(3), 4);
  EXPECT_EQ(dispenso::nextPow2(17), 32);
  EXPECT_EQ(dispenso::nextPow2(1ULL << 30), 1ULL << 30);
  EXPECT_EQ(dispenso::nextPow2((1ULL << 30) + 1), 1ULL << 31);
}

TEST(Util, Log2Const) {
  // Compile-time tests
  static_assert(dispenso::log2const(1) == 0, "log2(1) should be 0");
  static_assert(dispenso::log2const(2) == 1, "log2(2) should be 1");
  static_assert(dispenso::log2const(4) == 2, "log2(4) should be 2");
  static_assert(dispenso::log2const(8) == 3, "log2(8) should be 3");
  static_assert(dispenso::log2const(16) == 4, "log2(16) should be 4");
  static_assert(dispenso::log2const(64) == 6, "log2(64) should be 6");
  static_assert(dispenso::log2const(100) == 6, "log2(100) should be 6");
  static_assert(dispenso::log2const(127) == 6, "log2(127) should be 6");
  static_assert(dispenso::log2const(128) == 7, "log2(128) should be 7");

  // Runtime tests
  EXPECT_EQ(dispenso::log2const(1), 0u);
  EXPECT_EQ(dispenso::log2const(2), 1u);
  EXPECT_EQ(dispenso::log2const(64), 6u);
  EXPECT_EQ(dispenso::log2const(1024), 10u);
  EXPECT_EQ(dispenso::log2const(1ULL << 32), 32u);
  EXPECT_EQ(dispenso::log2const(1ULL << 63), 63u);
}

TEST(Util, Log2) {
  // Runtime log2 should match constexpr version
  EXPECT_EQ(dispenso::log2(1), dispenso::log2const(1));
  EXPECT_EQ(dispenso::log2(2), dispenso::log2const(2));
  EXPECT_EQ(dispenso::log2(64), dispenso::log2const(64));
  EXPECT_EQ(dispenso::log2(100), dispenso::log2const(100));
  EXPECT_EQ(dispenso::log2(1024), dispenso::log2const(1024));
  EXPECT_EQ(dispenso::log2(1ULL << 32), dispenso::log2const(1ULL << 32));

  // Test powers of 2
  for (uint32_t i = 0; i < 64; ++i) {
    uint64_t val = 1ULL << i;
    EXPECT_EQ(dispenso::log2(val), i);
  }

  // Test non-powers of 2 (floor behavior)
  EXPECT_EQ(dispenso::log2(3), 1u);
  EXPECT_EQ(dispenso::log2(5), 2u);
  EXPECT_EQ(dispenso::log2(127), 6u);
  EXPECT_EQ(dispenso::log2(129), 7u);
}

TEST(Util, AlignedBuffer) {
  struct TestType {
    alignas(32) char data[64];
  };

  dispenso::AlignedBuffer<TestType> buf;
  // The struct containing the buffer has proper alignment, not the char array itself
  EXPECT_GE(alignof(decltype(buf)), alignof(TestType));
  EXPECT_GE(sizeof(buf.b), sizeof(TestType));

  // Verify we can construct in the buffer
  TestType* obj = new (buf.b) TestType();
  obj->data[0] = 42;
  EXPECT_EQ(obj->data[0], 42);
  obj->~TestType();
}

TEST(Util, AlignedAtomic) {
  dispenso::AlignedAtomic<int> atomic;

  // Verify alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&atomic) % dispenso::kCacheLineSize, 0u);

  // Verify it works as atomic
  int val = 42;
  atomic.store(&val);
  EXPECT_EQ(atomic.load(), &val);

  // Test from multiple threads to verify atomicity
  std::atomic<bool> ready{false};
  std::atomic<bool> done{false};

  std::thread t([&]() {
    while (!ready.load(std::memory_order_acquire)) {
      dispenso::cpuRelax();
    }
    int* expected = nullptr;
    atomic.compare_exchange_strong(expected, &val);
    done.store(true, std::memory_order_release);
  });

  int* ptr = nullptr;
  atomic.store(ptr);
  ready.store(true, std::memory_order_release);

  while (!done.load(std::memory_order_acquire)) {
    dispenso::cpuRelax();
  }

  EXPECT_EQ(atomic.load(), &val);
  t.join();
}

TEST(Util, OpResult) {
  // Test default construction
  dispenso::OpResult<int> r1;
  EXPECT_FALSE(r1);
  EXPECT_FALSE(r1.has_value());

  // Test value construction
  dispenso::OpResult<int> r2(42);
  EXPECT_TRUE(r2);
  EXPECT_TRUE(r2.has_value());
  EXPECT_EQ(r2.value(), 42);

  // Test copy construction
  dispenso::OpResult<int> r3(r2);
  EXPECT_TRUE(r3);
  EXPECT_EQ(r3.value(), 42);

  // Test move construction
  dispenso::OpResult<int> r4(std::move(r3));
  EXPECT_TRUE(r4);
  EXPECT_EQ(r4.value(), 42);
  EXPECT_FALSE(r3); // Moved-from should be empty

  // Test copy assignment
  dispenso::OpResult<int> r5;
  r5 = r2;
  EXPECT_TRUE(r5);
  EXPECT_EQ(r5.value(), 42);

  // Test move assignment
  dispenso::OpResult<int> r6;
  r6 = std::move(r5);
  EXPECT_TRUE(r6);
  EXPECT_EQ(r6.value(), 42);
  EXPECT_FALSE(r5);

  // Test emplace
  dispenso::OpResult<int> r7;
  r7.emplace(100);
  EXPECT_TRUE(r7);
  EXPECT_EQ(r7.value(), 100);

  // Test self-assignment (suppressing warnings as this is intentional for testing)
  // clang-format off
  DISPENSO_DISABLE_WARNING_PUSH
#if defined(__clang__)
  DISPENSO_DISABLE_WARNING(-Wself-assign-overloaded)
#endif
#if defined(__clang__) || defined(__GNUC__)
  DISPENSO_DISABLE_WARNING(-Wself-move)
#endif
  // clang-format on
  r7 = r7;
  EXPECT_TRUE(r7);
  EXPECT_EQ(r7.value(), 100);

  r7 = std::move(r7);
  EXPECT_TRUE(r7);
  EXPECT_EQ(r7.value(), 100);
  DISPENSO_DISABLE_WARNING_POP
}

TEST(Util, OpResultWithNonTrivialType) {
  NonTrivialConstructCount = 0;
  NonTrivialDestructCount = 0;

  {
    dispenso::OpResult<NonTrivial> r1(NonTrivial(42));
    EXPECT_EQ(r1.value().value, 42);
    EXPECT_GT(NonTrivialConstructCount, 0);

    dispenso::OpResult<NonTrivial> r2(r1);
    EXPECT_EQ(r2.value().value, 42);

    r1.emplace(100);
    EXPECT_EQ(r1.value().value, 100);
  }

  // All objects should be destroyed
  EXPECT_EQ(NonTrivialConstructCount, NonTrivialDestructCount);
}

TEST(Util, StaticChunkSize) {
  // Test exact division
  auto chunking = dispenso::staticChunkSize(100, 10);
  EXPECT_EQ(chunking.ceilChunkSize, 10);
  EXPECT_EQ(chunking.transitionTaskIndex, 10);

  // Verify all items are accounted for
  ssize_t total = 0;
  for (ssize_t i = 0; i < 10; ++i) {
    ssize_t chunkSize =
        (i < chunking.transitionTaskIndex) ? chunking.ceilChunkSize : chunking.ceilChunkSize - 1;
    total += chunkSize;
  }
  EXPECT_EQ(total, 100);

  // Test with remainder
  chunking = dispenso::staticChunkSize(100, 8);
  EXPECT_EQ(chunking.ceilChunkSize, 13);
  EXPECT_EQ(chunking.transitionTaskIndex, 4);

  // Verify: 4 chunks of 13, 4 chunks of 12
  total = 0;
  for (ssize_t i = 0; i < 8; ++i) {
    ssize_t chunkSize =
        (i < chunking.transitionTaskIndex) ? chunking.ceilChunkSize : chunking.ceilChunkSize - 1;
    total += chunkSize;
  }
  EXPECT_EQ(total, 100);

  // Test edge cases
  chunking = dispenso::staticChunkSize(1, 10);
  EXPECT_EQ(chunking.ceilChunkSize, 1);
  EXPECT_EQ(chunking.transitionTaskIndex, 1);

  chunking = dispenso::staticChunkSize(7, 3);
  EXPECT_EQ(chunking.ceilChunkSize, 3);
  EXPECT_EQ(chunking.transitionTaskIndex, 1);
  // 1 chunk of 3, 2 chunks of 2 = 3 + 2 + 2 = 7
  total = 0;
  for (ssize_t i = 0; i < 3; ++i) {
    ssize_t chunkSize =
        (i < chunking.transitionTaskIndex) ? chunking.ceilChunkSize : chunking.ceilChunkSize - 1;
    total += chunkSize;
  }
  EXPECT_EQ(total, 7);
}

TEST(Util, StaticChunkSizeLoadBalancing) {
  // Verify load balancing is good
  // Maximum difference between chunks should be 1
  for (ssize_t items : {100, 101, 127, 256, 1000, 10001}) {
    for (ssize_t chunks : {2, 4, 7, 8, 16, 32}) {
      auto chunking = dispenso::staticChunkSize(items, chunks);

      ssize_t minChunk = chunking.ceilChunkSize - 1;
      ssize_t maxChunk = chunking.ceilChunkSize;

      EXPECT_LE(maxChunk - minChunk, 1)
          << "Chunk sizes should differ by at most 1 for items=" << items << " chunks=" << chunks;

      // Verify total
      ssize_t total = 0;
      for (ssize_t i = 0; i < chunks; ++i) {
        ssize_t chunkSize = (i < chunking.transitionTaskIndex) ? chunking.ceilChunkSize
                                                               : chunking.ceilChunkSize - 1;
        total += chunkSize;
      }
      EXPECT_EQ(total, items) << "Total items mismatch for items=" << items << " chunks=" << chunks;
    }
  }
}
