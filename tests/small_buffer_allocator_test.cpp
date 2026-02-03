/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/small_buffer_allocator.h>

#include <thread>
#include <vector>

#include <gtest/gtest.h>

// Helper to test alloc/dealloc for a specific size
template <size_t kSize>
void testAllocDealloc() {
  char* buf = dispenso::allocSmallBuffer<kSize>();
  ASSERT_NE(buf, nullptr);
  // Write to first and last byte to verify usability
  buf[0] = 'a';
  buf[kSize - 1] = 'z';
  dispenso::deallocSmallBuffer<kSize>(buf);
}

// Test allocation and deallocation for various sizes from 1 to 1024
// Note: The allocator supports sizes 4-256 in pools (smaller sizes map to 4-byte blocks),
// larger sizes use alignedMalloc
TEST(SmallBufferAllocator, AllocDeallocVariousSizes) {
  testAllocDealloc<1>(); // Maps to 4-byte block
  testAllocDealloc<2>(); // Maps to 4-byte block
  testAllocDealloc<3>(); // Maps to 4-byte block (non-power-of-2)
  testAllocDealloc<4>();
  testAllocDealloc<8>();
  testAllocDealloc<16>();
  testAllocDealloc<32>();
  testAllocDealloc<64>();
  testAllocDealloc<128>();
  testAllocDealloc<256>();
  testAllocDealloc<512>();
  testAllocDealloc<1024>();
}

// Helper to test bytesAllocated for a specific size
template <
    size_t kSize,
    typename std::enable_if<(kSize <= dispenso::kMaxSmallBufferSize), int>::type = 0>
void testBytesAllocated() {
  char* buf = dispenso::allocSmallBuffer<kSize>();
  EXPECT_GT(dispenso::approxBytesAllocatedSmallBuffer<kSize>(), size_t{0});
  dispenso::deallocSmallBuffer<kSize>(buf);
}

// Overload for sizes > kMaxSmallBufferSize (no-op since they use aligned malloc)
template <
    size_t kSize,
    typename std::enable_if<(kSize > dispenso::kMaxSmallBufferSize), int>::type = 0>
void testBytesAllocated() {
  // Sizes larger than kMaxSmallBufferSize fall back to aligned malloc
  // and don't track bytes in the pool, so nothing to test
}

// Test bytesAllocated for various sizes
TEST(SmallBufferAllocator, BytesAllocatedVariousSizes) {
  testBytesAllocated<4>();
  testBytesAllocated<8>();
  testBytesAllocated<16>();
  testBytesAllocated<32>();
  testBytesAllocated<64>();
  testBytesAllocated<128>();
  testBytesAllocated<256>();
}

// Test multiple allocations of same size
template <size_t kSize>
void testMultipleAllocs() {
  constexpr size_t kNumAllocs = 100;
  std::vector<char*> buffers(kNumAllocs);

  // Allocate many buffers
  for (size_t i = 0; i < kNumAllocs; ++i) {
    buffers[i] = dispenso::allocSmallBuffer<kSize>();
    ASSERT_NE(buffers[i], nullptr);
    // Write a pattern to verify no overlap
    buffers[i][0] = static_cast<char>(i & 0xFF);
  }

  // Verify patterns
  for (size_t i = 0; i < kNumAllocs; ++i) {
    EXPECT_EQ(buffers[i][0], static_cast<char>(i & 0xFF));
  }

  // Deallocate all
  for (size_t i = 0; i < kNumAllocs; ++i) {
    dispenso::deallocSmallBuffer<kSize>(buffers[i]);
  }
}

TEST(SmallBufferAllocator, MultipleAllocsSmallSize) {
  testMultipleAllocs<16>();
}

TEST(SmallBufferAllocator, MultipleAllocsMediumSize) {
  testMultipleAllocs<64>();
}

TEST(SmallBufferAllocator, MultipleAllocsLargeSize) {
  testMultipleAllocs<256>();
}

TEST(SmallBufferAllocator, MultipleAllocsOverMaxSize) {
  testMultipleAllocs<512>();
}

// Test allocation reuse after deallocation
TEST(SmallBufferAllocator, AllocationReuse) {
  // Allocate and deallocate, then allocate again
  char* buf1 = dispenso::allocSmallBuffer<32>();
  ASSERT_NE(buf1, nullptr);
  dispenso::deallocSmallBuffer<32>(buf1);

  char* buf2 = dispenso::allocSmallBuffer<32>();
  ASSERT_NE(buf2, nullptr);
  dispenso::deallocSmallBuffer<32>(buf2);
}

// Test threaded allocation/deallocation with various sizes
TEST(SmallBufferAllocator, ThreadedAllocDealloc) {
  constexpr size_t kNumThreads = 8;
  constexpr size_t kNumAllocsPerThread = 500;

  auto threadFunc = [&](size_t threadId) {
    for (size_t i = 0; i < kNumAllocsPerThread; ++i) {
      // Rotate through different sizes
      switch ((threadId + i) % 5) {
        case 0: {
          char* buf = dispenso::allocSmallBuffer<8>();
          buf[0] = 'x';
          dispenso::deallocSmallBuffer<8>(buf);
          break;
        }
        case 1: {
          char* buf = dispenso::allocSmallBuffer<32>();
          buf[0] = 'x';
          dispenso::deallocSmallBuffer<32>(buf);
          break;
        }
        case 2: {
          char* buf = dispenso::allocSmallBuffer<64>();
          buf[0] = 'x';
          dispenso::deallocSmallBuffer<64>(buf);
          break;
        }
        case 3: {
          char* buf = dispenso::allocSmallBuffer<128>();
          buf[0] = 'x';
          dispenso::deallocSmallBuffer<128>(buf);
          break;
        }
        case 4: {
          char* buf = dispenso::allocSmallBuffer<256>();
          buf[0] = 'x';
          dispenso::deallocSmallBuffer<256>(buf);
          break;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back(threadFunc, t);
  }

  for (auto& t : threads) {
    t.join();
  }
}
