/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/mpmc_ring_buffer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::MpmcRingBuffer;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(MpmcRingBuffer, DefaultConstructionIsEmpty) {
  MpmcRingBuffer<int> buffer;
  EXPECT_TRUE(buffer.empty());
  EXPECT_FALSE(buffer.full());
  EXPECT_EQ(buffer.size(), 0u);
}

TEST(MpmcRingBuffer, CapacityIsCorrect) {
  // With RoundUpToPowerOfTwo=true (default), capacity is rounded up to next power of two
  // Capacity=8: buffer size = 8, actual capacity = 8
  MpmcRingBuffer<int, 8> buffer8;
  EXPECT_EQ(buffer8.capacity(), 8u);

  // Capacity=2: buffer size = 2, actual capacity = 2
  MpmcRingBuffer<int, 2> buffer2;
  EXPECT_EQ(buffer2.capacity(), 2u);

  // Capacity=100: buffer size = nextPowerOfTwo(100) = 128
  MpmcRingBuffer<int, 100> buffer100;
  EXPECT_EQ(buffer100.capacity(), 128u);

  // Default capacity (16): buffer size = 16
  MpmcRingBuffer<int> bufferDefault;
  EXPECT_EQ(bufferDefault.capacity(), 16u);

  // With RoundUpToPowerOfTwo=false, capacity matches the template parameter exactly
  // Note: masking requires power-of-two, so non-power-of-two only works when
  // the capacity happens to be a power of two or kMask wrapping is correct.
  MpmcRingBuffer<int, 8, false> exactBuffer8;
  EXPECT_EQ(exactBuffer8.capacity(), 8u);

  MpmcRingBuffer<int, 16, false> exactBuffer16;
  EXPECT_EQ(exactBuffer16.capacity(), 16u);
}

TEST(MpmcRingBuffer, PushAndPopSingleElement) {
  MpmcRingBuffer<int, 4> buffer;

  EXPECT_TRUE(buffer.try_push(42));
  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size(), 1u);

  int value = 0;
  EXPECT_TRUE(buffer.try_pop(value));
  EXPECT_EQ(value, 42);
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0u);
}

TEST(MpmcRingBuffer, PushMoveSemantics) {
  MpmcRingBuffer<std::string, 4> buffer;

  std::string original = "hello world";
  EXPECT_TRUE(buffer.try_push(std::move(original)));
  EXPECT_TRUE(original.empty());

  std::string result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, "hello world");
}

TEST(MpmcRingBuffer, PushCopySemantics) {
  MpmcRingBuffer<int, 4> buffer;

  int original = 42;
  EXPECT_TRUE(buffer.try_push(original));
  EXPECT_EQ(original, 42);

  int result = 0;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, 42);
}

TEST(MpmcRingBuffer, TryEmplace) {
  MpmcRingBuffer<int, 4> buffer;

  EXPECT_TRUE(buffer.try_emplace(42));
  EXPECT_EQ(buffer.size(), 1u);

  int result = 0;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, 42);
}

TEST(MpmcRingBuffer, FIFOOrdering) {
  MpmcRingBuffer<int, 8> buffer;

  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }

  for (int i = 0; i < 8; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

TEST(MpmcRingBuffer, FullBufferRejectsPush) {
  MpmcRingBuffer<int, 4> buffer;

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }
  EXPECT_TRUE(buffer.full());
  EXPECT_EQ(buffer.size(), 4u);

  EXPECT_FALSE(buffer.try_push(99));
}

TEST(MpmcRingBuffer, EmptyBufferRejectsPop) {
  MpmcRingBuffer<int, 4> buffer;
  int value = -1;
  EXPECT_FALSE(buffer.try_pop(value));
  EXPECT_EQ(value, -1);
}

TEST(MpmcRingBuffer, TryPopOpResult) {
  MpmcRingBuffer<int, 4> buffer;

  // Pop from empty
  auto emptyResult = buffer.try_pop();
  EXPECT_FALSE(emptyResult.has_value());

  // Push and pop
  buffer.try_push(42);
  auto result = buffer.try_pop();
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

TEST(MpmcRingBuffer, TryPopInto) {
  MpmcRingBuffer<std::string, 4> buffer;
  buffer.try_push(std::string("hello"));

  alignas(std::string) char storage[sizeof(std::string)];
  auto* ptr = reinterpret_cast<std::string*>(storage);

  EXPECT_TRUE(buffer.try_pop_into(ptr));
  EXPECT_EQ(*ptr, "hello");
  ptr->~basic_string();
}

TEST(MpmcRingBuffer, WrapAround) {
  MpmcRingBuffer<int, 4> buffer;

  // Fill and drain multiple times to exercise wrap-around
  for (int round = 0; round < 10; ++round) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(buffer.try_push(round * 4 + i));
    }
    EXPECT_TRUE(buffer.full());

    for (int i = 0; i < 4; ++i) {
      int value = -1;
      EXPECT_TRUE(buffer.try_pop(value));
      EXPECT_EQ(value, round * 4 + i);
    }
    EXPECT_TRUE(buffer.empty());
  }
}

TEST(MpmcRingBuffer, AlternatingPushPop) {
  MpmcRingBuffer<int, 4> buffer;

  // Alternate push/pop to exercise wrap-around with partial fills
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
    EXPECT_TRUE(buffer.empty());
  }
}

// =============================================================================
// Move-Only Type Tests
// =============================================================================

TEST(MpmcRingBuffer, MoveOnlyType) {
  MpmcRingBuffer<std::unique_ptr<int>, 4> buffer;

  EXPECT_TRUE(buffer.try_push(std::make_unique<int>(42)));
  EXPECT_EQ(buffer.size(), 1u);

  std::unique_ptr<int> result;
  EXPECT_TRUE(buffer.try_pop(result));
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(*result, 42);
}

// =============================================================================
// Non-Trivial Type Tests
// =============================================================================

namespace {
struct NonTrivial {
  static std::atomic<int> constructCount;
  static std::atomic<int> destructCount;
  static std::atomic<int> moveCount;

  int value;

  explicit NonTrivial(int v) noexcept : value(v) {
    constructCount.fetch_add(1, std::memory_order_relaxed);
  }
  NonTrivial(NonTrivial&& other) noexcept : value(other.value) {
    other.value = -1;
    moveCount.fetch_add(1, std::memory_order_relaxed);
  }
  NonTrivial& operator=(NonTrivial&& other) noexcept {
    value = other.value;
    other.value = -1;
    moveCount.fetch_add(1, std::memory_order_relaxed);
    return *this;
  }
  ~NonTrivial() {
    destructCount.fetch_add(1, std::memory_order_relaxed);
  }

  NonTrivial(const NonTrivial&) = delete;
  NonTrivial& operator=(const NonTrivial&) = delete;

  static void resetCounts() {
    constructCount.store(0, std::memory_order_relaxed);
    destructCount.store(0, std::memory_order_relaxed);
    moveCount.store(0, std::memory_order_relaxed);
  }
};

std::atomic<int> NonTrivial::constructCount{0};
std::atomic<int> NonTrivial::destructCount{0};
std::atomic<int> NonTrivial::moveCount{0};
} // namespace

// Fixture for tests that assert on NonTrivial's global lifetime counters. The
// SetUp() reset makes the per-test reset automatic, so it cannot be forgotten
// when new NonTrivial-based tests are added.
class MpmcRingBufferNonTrivialTest : public ::testing::Test {
 protected:
  void SetUp() override {
    NonTrivial::resetCounts();
  }
};

TEST_F(MpmcRingBufferNonTrivialTest, NonTrivialLifetime) {
  {
    MpmcRingBuffer<NonTrivial, 4> buffer;

    EXPECT_TRUE(buffer.try_emplace(1));
    EXPECT_TRUE(buffer.try_emplace(2));
    EXPECT_TRUE(buffer.try_emplace(3));
    EXPECT_EQ(NonTrivial::constructCount.load(), 3);
    EXPECT_EQ(NonTrivial::destructCount.load(), 0);

    NonTrivial val(0);
    EXPECT_TRUE(buffer.try_pop(val));
    EXPECT_EQ(val.value, 1);

    // Remaining 2 elements destroyed when buffer goes out of scope
  }

  // 4 constructed (3 emplaced + 1 for val), all destroyed
  EXPECT_EQ(NonTrivial::constructCount.load(), 4);
  int totalDestructions = NonTrivial::destructCount.load();
  // 1 popped element destroyed in ring + 2 remaining destroyed at scope end + val destroyed
  EXPECT_EQ(totalDestructions, 4);
}

TEST_F(MpmcRingBufferNonTrivialTest, DestructorCleansUpRemainingElements) {
  {
    MpmcRingBuffer<NonTrivial, 8> buffer;
    for (int i = 0; i < 5; ++i) {
      EXPECT_TRUE(buffer.try_emplace(i));
    }
    EXPECT_EQ(NonTrivial::constructCount.load(), 5);
    // Pop 2, leaving 3 in the buffer
    NonTrivial val(0);
    EXPECT_TRUE(buffer.try_pop(val));
    EXPECT_TRUE(buffer.try_pop(val));
  }

  // All 6 objects (5 emplaced + 1 val) should be destroyed
  EXPECT_EQ(NonTrivial::constructCount.load(), 6);
  // 2 popped in ring + 3 remaining at scope end + val at scope end
  EXPECT_EQ(NonTrivial::destructCount.load(), 6);
}

// =============================================================================
// Bulk Operation Tests
// =============================================================================

TEST(MpmcRingBuffer, BulkPushAll) {
  MpmcRingBuffer<int, 8> buffer;

  std::array<int, 8> items = {10, 20, 30, 40, 50, 60, 70, 80};
  size_t pushed = buffer.try_push_batch(items.data(), items.size());
  EXPECT_EQ(pushed, 8u);
  EXPECT_TRUE(buffer.full());
  EXPECT_EQ(buffer.size(), 8u);

  // Verify FIFO order
  for (int i = 0; i < 8; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, items[static_cast<size_t>(i)]);
  }
}

TEST(MpmcRingBuffer, BulkPushPartial) {
  MpmcRingBuffer<int, 4> buffer;

  // Fill 2 slots first
  EXPECT_TRUE(buffer.try_push(1));
  EXPECT_TRUE(buffer.try_push(2));

  // Try to push 4 items into remaining 2 slots
  std::array<int, 4> items = {10, 20, 30, 40};
  size_t pushed = buffer.try_push_batch(items.data(), items.size());
  // Exactly 2 remaining slots, single-threaded, deterministic
  EXPECT_EQ(pushed, 2u);

  // Drain and verify: 2 original + 2 batch-pushed = 4
  std::vector<int> results;
  int value;
  while (buffer.try_pop(value)) {
    results.push_back(value);
  }
  EXPECT_EQ(results.size(), 4u);
  EXPECT_EQ(results[0], 1);
  EXPECT_EQ(results[1], 2);
  EXPECT_EQ(results[2], 10);
  EXPECT_EQ(results[3], 20);
}

TEST(MpmcRingBuffer, BulkPushEmpty) {
  MpmcRingBuffer<int, 4> buffer;
  size_t pushed = buffer.try_push_batch(nullptr, 0);
  EXPECT_EQ(pushed, 0u);
}

TEST(MpmcRingBuffer, BulkPushToFull) {
  MpmcRingBuffer<int, 4> buffer;

  // Fill completely
  for (int i = 0; i < 4; ++i) {
    buffer.try_push(i);
  }

  // Bulk push should return 0
  std::array<int, 2> items = {10, 20};
  size_t pushed = buffer.try_push_batch(items.data(), items.size());
  EXPECT_EQ(pushed, 0u);
}

TEST(MpmcRingBuffer, BulkPushAfterWrapAround) {
  MpmcRingBuffer<int, 4> buffer;

  // Fill and drain to advance indices past the buffer boundary
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 4; ++i) {
      buffer.try_push(i);
    }
    int v;
    for (int i = 0; i < 4; ++i) {
      buffer.try_pop(v);
    }
  }

  // Now bulk push should still work
  std::array<int, 4> items = {100, 200, 300, 400};
  size_t pushed = buffer.try_push_batch(items.data(), items.size());
  EXPECT_EQ(pushed, 4u);

  for (int i = 0; i < 4; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, items[static_cast<size_t>(i)]);
  }
}

TEST(MpmcRingBuffer, BulkPushSingleItem) {
  MpmcRingBuffer<int, 4> buffer;

  int item = 42;
  size_t pushed = buffer.try_push_batch(&item, 1);
  EXPECT_EQ(pushed, 1u);

  int value = -1;
  EXPECT_TRUE(buffer.try_pop(value));
  EXPECT_EQ(value, 42);
}

// =============================================================================
// String Element Tests
// =============================================================================

TEST(MpmcRingBuffer, StringElements) {
  MpmcRingBuffer<std::string, 4> buffer;

  EXPECT_TRUE(buffer.try_push(std::string("hello")));
  EXPECT_TRUE(buffer.try_push(std::string("world")));

  std::string result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, "hello");
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, "world");
}

// =============================================================================
// Non-Default-Constructible Type Tests
// =============================================================================

namespace {
struct NoDefault {
  int value;
  explicit NoDefault(int v) noexcept : value(v) {}
  NoDefault(NoDefault&& other) noexcept : value(other.value) {
    other.value = -1;
  }
  NoDefault& operator=(NoDefault&& other) noexcept {
    value = other.value;
    other.value = -1;
    return *this;
  }
  NoDefault() = delete;
  NoDefault(const NoDefault&) = delete;
  NoDefault& operator=(const NoDefault&) = delete;
};
} // namespace

TEST(MpmcRingBuffer, NonDefaultConstructible) {
  MpmcRingBuffer<NoDefault, 4> buffer;

  EXPECT_TRUE(buffer.try_emplace(42));
  EXPECT_EQ(buffer.size(), 1u);

  // Can't use try_pop(T&) since T isn't default-constructible, use try_pop_into
  alignas(NoDefault) char storage[sizeof(NoDefault)];
  auto* ptr = reinterpret_cast<NoDefault*>(storage);
  EXPECT_TRUE(buffer.try_pop_into(ptr));
  EXPECT_EQ(ptr->value, 42);
  ptr->~NoDefault();
}

// =============================================================================
// Concurrent Tests
// =============================================================================

TEST(MpmcRingBuffer, SingleProducerSingleConsumer) {
  MpmcRingBuffer<int, 16> buffer;
  constexpr int kCount = 100000;
  std::atomic<bool> done{false};

  std::thread producer([&]() {
    for (int i = 0; i < kCount; ++i) {
      while (!buffer.try_push(i)) {
        std::this_thread::yield();
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::vector<int> results;
  results.reserve(kCount);

  while (!done.load(std::memory_order_acquire) || !buffer.empty()) {
    int value;
    if (buffer.try_pop(value)) {
      results.push_back(value);
    } else {
      std::this_thread::yield();
    }
  }

  producer.join();

  // Drain any remaining
  int value;
  while (buffer.try_pop(value)) {
    results.push_back(value);
  }

  ASSERT_EQ(results.size(), static_cast<size_t>(kCount));
  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i);
  }
}

TEST(MpmcRingBuffer, MultipleProducersSingleConsumer) {
  MpmcRingBuffer<int, 32> buffer;
  constexpr int kProducers = 4;
  constexpr int kItemsPerProducer = 25000;
  constexpr int kTotalItems = kProducers * kItemsPerProducer;

  std::atomic<int> producersDone{0};
  std::vector<std::thread> producers;

  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&buffer, &producersDone, p]() {
      int base = p * kItemsPerProducer;
      for (int i = 0; i < kItemsPerProducer; ++i) {
        while (!buffer.try_push(base + i)) {
          std::this_thread::yield();
        }
      }
      producersDone.fetch_add(1, std::memory_order_release);
    });
  }

  std::vector<int> results;
  results.reserve(kTotalItems);

  while (producersDone.load(std::memory_order_acquire) < kProducers || !buffer.empty()) {
    int value;
    if (buffer.try_pop(value)) {
      results.push_back(value);
    } else {
      std::this_thread::yield();
    }
  }

  for (auto& t : producers) {
    t.join();
  }

  // Drain any remaining
  int value;
  while (buffer.try_pop(value)) {
    results.push_back(value);
  }

  ASSERT_EQ(results.size(), static_cast<size_t>(kTotalItems));

  // Each producer's items should appear in order relative to each other
  std::vector<int> perProducerNext(kProducers, 0);
  for (int v : results) {
    int p = v / kItemsPerProducer;
    int idx = v % kItemsPerProducer;
    ASSERT_GE(p, 0);
    ASSERT_LT(p, kProducers);
    EXPECT_EQ(idx, perProducerNext[static_cast<size_t>(p)])
        << "Producer " << p << " items out of order";
    perProducerNext[static_cast<size_t>(p)]++;
  }
}

TEST(MpmcRingBuffer, SingleProducerMultipleConsumers) {
  MpmcRingBuffer<int, 32> buffer;
  constexpr int kConsumers = 4;
  constexpr int kTotalItems = 100000;

  std::atomic<bool> producerDone{false};
  std::vector<std::thread> consumers;
  std::vector<std::vector<int>> perConsumerResults(kConsumers);

  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&buffer, &producerDone, &perConsumerResults, c]() {
      auto& myResults = perConsumerResults[static_cast<size_t>(c)];
      while (!producerDone.load(std::memory_order_acquire) || !buffer.empty()) {
        int value;
        if (buffer.try_pop(value)) {
          myResults.push_back(value);
        } else {
          std::this_thread::yield();
        }
      }
      // Final drain
      int value;
      while (buffer.try_pop(value)) {
        myResults.push_back(value);
      }
    });
  }

  // Producer
  for (int i = 0; i < kTotalItems; ++i) {
    while (!buffer.try_push(i)) {
      std::this_thread::yield();
    }
  }
  producerDone.store(true, std::memory_order_release);

  for (auto& t : consumers) {
    t.join();
  }

  // Collect all results, sort, and verify completeness
  std::vector<int> allResults;
  for (auto& cr : perConsumerResults) {
    allResults.insert(allResults.end(), cr.begin(), cr.end());
  }
  std::sort(allResults.begin(), allResults.end());

  ASSERT_EQ(allResults.size(), static_cast<size_t>(kTotalItems));
  for (int i = 0; i < kTotalItems; ++i) {
    EXPECT_EQ(allResults[static_cast<size_t>(i)], i);
  }
}

TEST(MpmcRingBuffer, MultipleProducersMultipleConsumers) {
  MpmcRingBuffer<int, 32> buffer;
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kItemsPerProducer = 25000;
  constexpr int kTotalItems = kProducers * kItemsPerProducer;

  std::atomic<int> producersDone{0};
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;
  std::vector<std::vector<int>> perConsumerResults(kConsumers);

  // Launch consumers first
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&buffer, &producersDone, &perConsumerResults, c]() {
      auto& myResults = perConsumerResults[static_cast<size_t>(c)];
      while (producersDone.load(std::memory_order_acquire) < kProducers || !buffer.empty()) {
        int value;
        if (buffer.try_pop(value)) {
          myResults.push_back(value);
        } else {
          std::this_thread::yield();
        }
      }
      int value;
      while (buffer.try_pop(value)) {
        myResults.push_back(value);
      }
    });
  }

  // Launch producers
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&buffer, &producersDone, p]() {
      int base = p * kItemsPerProducer;
      for (int i = 0; i < kItemsPerProducer; ++i) {
        while (!buffer.try_push(base + i)) {
          std::this_thread::yield();
        }
      }
      producersDone.fetch_add(1, std::memory_order_release);
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  for (auto& t : consumers) {
    t.join();
  }

  std::vector<int> allResults;
  for (auto& cr : perConsumerResults) {
    allResults.insert(allResults.end(), cr.begin(), cr.end());
  }
  std::sort(allResults.begin(), allResults.end());

  ASSERT_EQ(allResults.size(), static_cast<size_t>(kTotalItems));
  for (int i = 0; i < kTotalItems; ++i) {
    EXPECT_EQ(allResults[static_cast<size_t>(i)], i);
  }
}

// =============================================================================
// Concurrent Bulk Push Tests
// =============================================================================

TEST(MpmcRingBuffer, ConcurrentBulkPush) {
  MpmcRingBuffer<int, 32> buffer;
  constexpr int kProducers = 4;
  constexpr int kBatchesPerProducer = 1000;
  constexpr int kBatchSize = 4;
  constexpr int kTotalItems = kProducers * kBatchesPerProducer * kBatchSize;

  std::atomic<int> producersDone{0};
  std::vector<std::thread> producers;

  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&buffer, &producersDone, p]() {
      for (int b = 0; b < kBatchesPerProducer; ++b) {
        std::array<int, kBatchSize> items;
        int base = p * kBatchesPerProducer * kBatchSize + b * kBatchSize;
        for (int i = 0; i < kBatchSize; ++i) {
          items[static_cast<size_t>(i)] = base + i;
        }

        size_t remaining = kBatchSize;
        size_t offset = 0;
        while (remaining > 0) {
          size_t pushed = buffer.try_push_batch(items.data() + offset, remaining);
          if (pushed > 0) {
            offset += pushed;
            remaining -= pushed;
          } else {
            std::this_thread::yield();
          }
        }
      }
      producersDone.fetch_add(1, std::memory_order_release);
    });
  }

  // Consumer
  std::vector<int> results;
  results.reserve(kTotalItems);

  while (producersDone.load(std::memory_order_acquire) < kProducers || !buffer.empty()) {
    int value;
    if (buffer.try_pop(value)) {
      results.push_back(value);
    } else {
      std::this_thread::yield();
    }
  }
  int value;
  while (buffer.try_pop(value)) {
    results.push_back(value);
  }

  for (auto& t : producers) {
    t.join();
  }

  std::sort(results.begin(), results.end());
  ASSERT_EQ(results.size(), static_cast<size_t>(kTotalItems));
  for (int i = 0; i < kTotalItems; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i);
  }
}

TEST(MpmcRingBuffer, BatchPushWithConcurrentConsumers) {
  // Regression test for the batch-push race: consumers completing out of order
  // could leave earlier slots unfinished while the batch producer sees a later
  // slot as available. The fix validates each slot individually.
  MpmcRingBuffer<int, 4> buffer;
  constexpr int kIterations = 200000;
  std::atomic<int64_t> pushSum{0};
  std::atomic<int64_t> popSum{0};
  std::atomic<bool> done{false};

  // Multiple consumers that deliberately yield between reservation and
  // completion to increase the window for out-of-order slot release.
  constexpr int kConsumers = 3;
  std::vector<std::thread> consumers;
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&]() {
      int64_t localSum = 0;
      while (!done.load(std::memory_order_acquire) || !buffer.empty()) {
        int value;
        if (buffer.try_pop(value)) {
          localSum += value;
        } else {
          std::this_thread::yield();
        }
      }
      int value;
      while (buffer.try_pop(value)) {
        localSum += value;
      }
      popSum.fetch_add(localSum, std::memory_order_relaxed);
    });
  }

  // Producer uses batch push with small batches into the tiny (4-slot) ring.
  int64_t localPushSum = 0;
  int next = 0;
  for (int iter = 0; iter < kIterations;) {
    int batchSize = 2 + (iter % 3); // 2, 3, or 4
    int remaining = kIterations - iter;
    if (batchSize > remaining) {
      batchSize = remaining;
    }
    std::array<int, 4> items;
    for (int i = 0; i < batchSize; ++i) {
      items[static_cast<size_t>(i)] = next + i;
    }
    size_t pushed = buffer.try_push_batch(items.data(), static_cast<size_t>(batchSize));
    if (pushed > 0) {
      for (size_t i = 0; i < pushed; ++i) {
        localPushSum += items[i];
      }
      next += static_cast<int>(pushed);
      iter += static_cast<int>(pushed);
    } else {
      std::this_thread::yield();
    }
  }
  pushSum.store(localPushSum, std::memory_order_relaxed);
  done.store(true, std::memory_order_release);

  for (auto& t : consumers) {
    t.join();
  }

  EXPECT_EQ(pushSum.load(), popSum.load());
  EXPECT_TRUE(buffer.empty());
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(MpmcRingBuffer, StressTest) {
  MpmcRingBuffer<int, 16> buffer;
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kItemsPerProducer = 100000;
  std::atomic<int> producersDone{0};
  std::atomic<int64_t> pushSum{0};
  std::atomic<int64_t> popSum{0};

  std::vector<std::thread> threads;

  // Producers
  for (int p = 0; p < kProducers; ++p) {
    threads.emplace_back([&buffer, &producersDone, &pushSum, p]() {
      int64_t localSum = 0;
      int base = p * kItemsPerProducer;
      for (int i = 0; i < kItemsPerProducer; ++i) {
        int val = base + i;
        while (!buffer.try_push(val)) {
          std::this_thread::yield();
        }
        localSum += val;
      }
      pushSum.fetch_add(localSum, std::memory_order_relaxed);
      producersDone.fetch_add(1, std::memory_order_release);
    });
  }

  // Consumers
  for (int c = 0; c < kConsumers; ++c) {
    threads.emplace_back([&buffer, &producersDone, &popSum]() {
      int64_t localSum = 0;
      while (producersDone.load(std::memory_order_acquire) < kProducers || !buffer.empty()) {
        int value;
        if (buffer.try_pop(value)) {
          localSum += value;
        } else {
          std::this_thread::yield();
        }
      }
      int value;
      while (buffer.try_pop(value)) {
        localSum += value;
      }
      popSum.fetch_add(localSum, std::memory_order_relaxed);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Verify all items were transferred correctly via sum comparison
  EXPECT_EQ(pushSum.load(), popSum.load());
  EXPECT_TRUE(buffer.empty());

  // Verify the expected sum
  int64_t expectedSum = 0;
  for (int p = 0; p < kProducers; ++p) {
    int base = p * kItemsPerProducer;
    expectedSum += static_cast<int64_t>(kItemsPerProducer) * (2 * base + kItemsPerProducer - 1) / 2;
  }
  EXPECT_EQ(pushSum.load(), expectedSum);
}

// =============================================================================
// Large Capacity Tests
// =============================================================================

TEST(MpmcRingBuffer, LargeCapacity) {
  MpmcRingBuffer<int, 1024> buffer;
  EXPECT_EQ(buffer.capacity(), 1024u);

  for (int i = 0; i < 1024; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }
  EXPECT_TRUE(buffer.full());

  for (int i = 0; i < 1024; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
  EXPECT_TRUE(buffer.empty());
}

TEST(MpmcRingBuffer, ExactCapacityModePow2) {
  MpmcRingBuffer<int, 4, false> buffer;
  EXPECT_EQ(buffer.capacity(), 4u);

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }
  EXPECT_TRUE(buffer.full());

  for (int i = 0; i < 4; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

TEST(MpmcRingBuffer, ExactCapacityModeNonPow2) {
  MpmcRingBuffer<int, 10, false> buffer;
  EXPECT_EQ(buffer.capacity(), 10u);

  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }
  EXPECT_TRUE(buffer.full());
  EXPECT_FALSE(buffer.try_push(99));

  for (int i = 0; i < 10; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
  EXPECT_TRUE(buffer.empty());

  // Wrap around: fill and drain again to exercise modulo index wrapping.
  for (int i = 100; i < 110; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }
  for (int i = 100; i < 110; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

// =============================================================================
// Fork-Join Use Case Tests
// =============================================================================

TEST(MpmcRingBuffer, TargetedSchedulingPattern) {
  // Simulate the fork-join pattern: one scheduler pushes to per-thread rings,
  // each thread pops from its own ring
  constexpr int kThreads = 8;
  std::array<MpmcRingBuffer<int, 16>, kThreads> rings;
  std::array<int, kThreads> results;
  results.fill(-1);

  // Scheduler pushes chunk i to ring i
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_TRUE(rings[static_cast<size_t>(i)].try_push(i * 100));
  }

  // Each "thread" pops from its own ring
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&rings, &results, i]() {
      int value;
      if (rings[static_cast<size_t>(i)].try_pop(value)) {
        results[static_cast<size_t>(i)] = value;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < kThreads; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i * 100) << "Thread " << i << " got wrong chunk";
  }
}

TEST(MpmcRingBuffer, BulkSchedulingWithOverflow) {
  // Simulate kAuto scheduling: bulk push to a ring, handle overflow
  MpmcRingBuffer<int, 4> ring;
  constexpr int kChunks = 10;

  std::array<int, kChunks> chunks;
  std::iota(chunks.begin(), chunks.end(), 0);

  size_t pushed = ring.try_push_batch(chunks.data(), kChunks);
  EXPECT_EQ(pushed, 4u); // Empty 4-slot ring, single-threaded, deterministic

  // "Overflow" items (chunks[pushed..kChunks]) would go to central queue
  size_t overflow = kChunks - pushed;
  EXPECT_EQ(overflow, 6u);

  // Verify pushed items are in the ring
  for (size_t i = 0; i < pushed; ++i) {
    int value = -1;
    EXPECT_TRUE(ring.try_pop(value));
    EXPECT_EQ(value, static_cast<int>(i));
  }
}

TEST(MpmcRingBuffer, WorkStealingPattern) {
  // Genuine concurrent work stealing: a producer fills ring0 while an "owner"
  // and a "thief" both drain it. Verifies every item is consumed exactly once
  // across both consumers under real concurrency — no loss, no duplication.
  MpmcRingBuffer<int, 16> ring0;
  constexpr int kNumItems = 100000;

  std::vector<std::atomic<uint8_t>> consumed(kNumItems);
  for (auto& c : consumed) {
    c.store(0, std::memory_order_relaxed);
  }
  std::atomic<int> consumedCount{0};
  std::atomic<bool> producerDone{false};

  // Producer: push 0..kNumItems-1 into ring0, retrying on full/contended.
  std::thread producer([&]() {
    for (int i = 0; i < kNumItems;) {
      if (ring0.try_push(int(i))) {
        ++i;
      } else {
        std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  // Two competing consumers (owner + thief) steal from the same ring.
  auto consume = [&]() {
    int value;
    while (true) {
      if (ring0.try_pop(value)) {
        // Each value must be popped exactly once: the flag transitions 0 -> 1.
        uint8_t prev = consumed[value].fetch_add(1, std::memory_order_relaxed);
        EXPECT_EQ(prev, 0u) << "value " << value << " consumed more than once";
        consumedCount.fetch_add(1, std::memory_order_relaxed);
      } else if (
          producerDone.load(std::memory_order_acquire) &&
          consumedCount.load(std::memory_order_acquire) >= kNumItems) {
        break; // producer finished and everything has been drained
      } else {
        std::this_thread::yield(); // empty/contended; keep stealing
      }
    }
  };

  std::thread owner(consume);
  std::thread thief(consume);

  producer.join();
  owner.join();
  thief.join();

  EXPECT_EQ(consumedCount.load(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i].load(), 1u) << "value " << i << " not consumed exactly once";
  }
}
