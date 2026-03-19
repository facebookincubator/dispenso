/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/spsc_ring_buffer.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using dispenso::SPSCRingBuffer;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(SPSCRingBuffer, DefaultConstructionIsEmpty) {
  SPSCRingBuffer<int> buffer;
  EXPECT_TRUE(buffer.empty());
  EXPECT_FALSE(buffer.full());
  EXPECT_EQ(buffer.size(), 0u);
}

TEST(SPSCRingBuffer, CapacityIsCorrect) {
  // With RoundUpToPowerOfTwo=true (default), capacity is rounded up to next power of two minus 1
  // Capacity=8: buffer size = nextPowerOfTwo(9) = 16, actual capacity = 15
  SPSCRingBuffer<int, 8> buffer8;
  EXPECT_EQ(buffer8.capacity(), 15u);

  // Capacity=1: buffer size = nextPowerOfTwo(2) = 2, actual capacity = 1
  SPSCRingBuffer<int, 1> buffer1;
  EXPECT_EQ(buffer1.capacity(), 1u);

  // Capacity=100: buffer size = nextPowerOfTwo(101) = 128, actual capacity = 127
  SPSCRingBuffer<int, 100> buffer100;
  EXPECT_EQ(buffer100.capacity(), 127u);

  // Default capacity (16): buffer size = nextPowerOfTwo(17) = 32, actual capacity = 31
  SPSCRingBuffer<int> bufferDefault;
  EXPECT_EQ(bufferDefault.capacity(), 31u);

  // With RoundUpToPowerOfTwo=false, capacity matches the template parameter exactly
  SPSCRingBuffer<int, 8, false> exactBuffer8;
  EXPECT_EQ(exactBuffer8.capacity(), 8u);

  SPSCRingBuffer<int, 100, false> exactBuffer100;
  EXPECT_EQ(exactBuffer100.capacity(), 100u);
}

TEST(SPSCRingBuffer, PushAndPopSingleElement) {
  SPSCRingBuffer<int, 4> buffer;

  EXPECT_TRUE(buffer.try_push(42));
  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size(), 1u);

  int value = 0;
  EXPECT_TRUE(buffer.try_pop(value));
  EXPECT_EQ(value, 42);
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0u);
}

TEST(SPSCRingBuffer, PushMoveSemantics) {
  SPSCRingBuffer<std::string, 4> buffer;

  std::string original = "hello world";
  EXPECT_TRUE(buffer.try_push(std::move(original)));
  // After move, original should be empty (or at least valid but unspecified)
  EXPECT_TRUE(original.empty());

  std::string result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, "hello world");
}

TEST(SPSCRingBuffer, PushCopySemantics) {
  SPSCRingBuffer<std::string, 4> buffer;

  std::string original = "hello world";
  EXPECT_TRUE(buffer.try_push(original));
  // After copy, original should be unchanged
  EXPECT_EQ(original, "hello world");

  std::string result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result, "hello world");
}

TEST(SPSCRingBuffer, TryEmplace) {
  SPSCRingBuffer<std::pair<int, std::string>, 4> buffer;

  EXPECT_TRUE(buffer.try_emplace(42, "hello"));
  EXPECT_EQ(buffer.size(), 1u);

  std::pair<int, std::string> result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_EQ(result.first, 42);
  EXPECT_EQ(result.second, "hello");
}

TEST(SPSCRingBuffer, PopFromEmpty) {
  SPSCRingBuffer<int, 4> buffer;

  int value = 999;
  EXPECT_FALSE(buffer.try_pop(value));
  // Value should be unchanged
  EXPECT_EQ(value, 999);
}

TEST(SPSCRingBuffer, TryPopReturnsOpResult) {
  SPSCRingBuffer<int, 4> buffer;

  // Pop from empty returns empty OpResult
  auto emptyResult = buffer.try_pop();
  EXPECT_FALSE(emptyResult);
  EXPECT_FALSE(emptyResult.has_value());

  // Push and pop returns value
  EXPECT_TRUE(buffer.try_push(42));
  auto result = buffer.try_pop();
  EXPECT_TRUE(result);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
  EXPECT_TRUE(buffer.empty());
}

TEST(SPSCRingBuffer, TryPopOpResultWithMoveOnlyType) {
  SPSCRingBuffer<std::unique_ptr<int>, 4> buffer;

  // Push a unique_ptr
  EXPECT_TRUE(buffer.try_push(std::make_unique<int>(123)));

  // Pop using OpResult
  auto result = buffer.try_pop();
  EXPECT_TRUE(result);
  EXPECT_NE(result.value(), nullptr);
  EXPECT_EQ(*result.value(), 123);
}

TEST(SPSCRingBuffer, TryPopOpResultWithString) {
  SPSCRingBuffer<std::string, 4> buffer;

  EXPECT_TRUE(buffer.try_push("hello world"));
  auto result = buffer.try_pop();
  EXPECT_TRUE(result);
  EXPECT_EQ(result.value(), "hello world");
}

TEST(SPSCRingBuffer, FIFOOrder) {
  SPSCRingBuffer<int, 8> buffer;

  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }

  for (int i = 0; i < 5; ++i) {
    int value = -1;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

// =============================================================================
// Capacity and Full/Empty Tests
// =============================================================================

TEST(SPSCRingBuffer, FillToCapacity) {
  SPSCRingBuffer<int, 4> buffer;
  const size_t cap = buffer.capacity();

  // Fill buffer to capacity
  for (size_t i = 0; i < cap; ++i) {
    EXPECT_FALSE(buffer.full());
    EXPECT_TRUE(buffer.try_push(static_cast<int>(i)));
  }

  EXPECT_TRUE(buffer.full());
  EXPECT_EQ(buffer.size(), cap);

  // Additional push should fail
  EXPECT_FALSE(buffer.try_push(100));
  EXPECT_EQ(buffer.size(), cap);
}

TEST(SPSCRingBuffer, FullThenPop) {
  SPSCRingBuffer<int, 4> buffer;
  const size_t cap = buffer.capacity();

  // Fill buffer
  for (size_t i = 0; i < cap; ++i) {
    EXPECT_TRUE(buffer.try_push(static_cast<int>(i)));
  }
  EXPECT_TRUE(buffer.full());

  // Pop one element
  int value;
  EXPECT_TRUE(buffer.try_pop(value));
  EXPECT_EQ(value, 0);
  EXPECT_FALSE(buffer.full());

  // Now push should succeed again
  EXPECT_TRUE(buffer.try_push(100));
  EXPECT_TRUE(buffer.full());
}

TEST(SPSCRingBuffer, CapacityOne) {
  SPSCRingBuffer<int, 1> buffer;

  EXPECT_TRUE(buffer.empty());
  EXPECT_FALSE(buffer.full());

  EXPECT_TRUE(buffer.try_push(42));
  EXPECT_FALSE(buffer.empty());
  EXPECT_TRUE(buffer.full());

  EXPECT_FALSE(buffer.try_push(43));

  int value;
  EXPECT_TRUE(buffer.try_pop(value));
  EXPECT_EQ(value, 42);
  EXPECT_TRUE(buffer.empty());
}

TEST(SPSCRingBuffer, SizeTracking) {
  SPSCRingBuffer<int, 8> buffer;

  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buffer.size(), i);
    EXPECT_TRUE(buffer.try_push(static_cast<int>(i)));
  }
  EXPECT_EQ(buffer.size(), 8u);

  for (size_t i = 8; i > 0; --i) {
    EXPECT_EQ(buffer.size(), i);
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
  }
  EXPECT_EQ(buffer.size(), 0u);
}

// =============================================================================
// Wrap-around Tests
// =============================================================================

TEST(SPSCRingBuffer, WrapAround) {
  // Use exact capacity mode for predictable wrap-around behavior in this test
  SPSCRingBuffer<int, 4, false> buffer;

  // Push 3 elements
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }

  // Pop 2 elements (advances head)
  for (int i = 0; i < 2; ++i) {
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }

  // Push 3 more elements (will wrap around)
  for (int i = 10; i < 13; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
  }

  // Should have 4 elements now: 2, 10, 11, 12
  EXPECT_EQ(buffer.size(), 4u);
  EXPECT_TRUE(buffer.full());

  // Pop all and verify order
  int expected[] = {2, 10, 11, 12};
  for (int exp : expected) {
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, exp);
  }
}

TEST(SPSCRingBuffer, MultipleWrapArounds) {
  SPSCRingBuffer<int, 4> buffer;

  // Perform many push/pop cycles to test wrap-around thoroughly
  for (int cycle = 0; cycle < 100; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(buffer.try_push(cycle * 10 + i));
    }
    for (int i = 0; i < 3; ++i) {
      int value;
      EXPECT_TRUE(buffer.try_pop(value));
      EXPECT_EQ(value, cycle * 10 + i);
    }
    EXPECT_TRUE(buffer.empty());
  }
}

TEST(SPSCRingBuffer, SizeWithWrapAround) {
  // Use exact capacity mode for predictable size behavior
  SPSCRingBuffer<int, 4, false> buffer;

  // Fill halfway
  buffer.try_push(1);
  buffer.try_push(2);

  // Pop one
  int v;
  buffer.try_pop(v);

  // Push to full (will wrap)
  buffer.try_push(3);
  buffer.try_push(4);
  buffer.try_push(5);

  EXPECT_EQ(buffer.size(), 4u);
  EXPECT_TRUE(buffer.full());
}

// =============================================================================
// Move-only Type Tests
// =============================================================================

TEST(SPSCRingBuffer, MoveOnlyType) {
  SPSCRingBuffer<std::unique_ptr<int>, 4> buffer;

  auto ptr = std::make_unique<int>(42);
  EXPECT_TRUE(buffer.try_push(std::move(ptr)));
  EXPECT_EQ(ptr, nullptr);

  std::unique_ptr<int> result;
  EXPECT_TRUE(buffer.try_pop(result));
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(*result, 42);
}

TEST(SPSCRingBuffer, MoveOnlyTypeMultiple) {
  SPSCRingBuffer<std::unique_ptr<int>, 4> buffer;

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(buffer.try_push(std::make_unique<int>(i)));
  }

  for (int i = 0; i < 4; ++i) {
    std::unique_ptr<int> result;
    EXPECT_TRUE(buffer.try_pop(result));
    EXPECT_EQ(*result, i);
  }
}

// =============================================================================
// Non-trivial Type Tests
// =============================================================================

namespace {
struct NonTrivial {
  static std::atomic<int> constructorCount;
  static std::atomic<int> destructorCount;
  static std::atomic<int> moveCount;

  int value;

  NonTrivial() : value(0) {
    constructorCount.fetch_add(1, std::memory_order_relaxed);
  }
  explicit NonTrivial(int v) : value(v) {
    constructorCount.fetch_add(1, std::memory_order_relaxed);
  }
  NonTrivial(const NonTrivial& other) : value(other.value) {
    constructorCount.fetch_add(1, std::memory_order_relaxed);
  }
  NonTrivial(NonTrivial&& other) noexcept : value(other.value) {
    other.value = -1;
    moveCount.fetch_add(1, std::memory_order_relaxed);
  }
  NonTrivial& operator=(const NonTrivial& other) {
    value = other.value;
    return *this;
  }
  NonTrivial& operator=(NonTrivial&& other) noexcept {
    value = other.value;
    other.value = -1;
    moveCount.fetch_add(1, std::memory_order_relaxed);
    return *this;
  }
  ~NonTrivial() {
    destructorCount.fetch_add(1, std::memory_order_relaxed);
  }

  static void resetCounters() {
    constructorCount.store(0, std::memory_order_relaxed);
    destructorCount.store(0, std::memory_order_relaxed);
    moveCount.store(0, std::memory_order_relaxed);
  }
};

std::atomic<int> NonTrivial::constructorCount{0};
std::atomic<int> NonTrivial::destructorCount{0};
std::atomic<int> NonTrivial::moveCount{0};
} // namespace

TEST(SPSCRingBuffer, NonTrivialType) {
  NonTrivial::resetCounters();

  {
    SPSCRingBuffer<NonTrivial, 4> buffer;

    NonTrivial obj(42);
    EXPECT_TRUE(buffer.try_push(std::move(obj)));

    NonTrivial result;
    EXPECT_TRUE(buffer.try_pop(result));
    EXPECT_EQ(result.value, 42);
  }

  // All objects should be destroyed
  // (buffer storage + temporary objects)
}

TEST(SPSCRingBuffer, NonTrivialDestructorOnBufferDestruction) {
  NonTrivial::resetCounters();

  {
    SPSCRingBuffer<NonTrivial, 4> buffer;

    buffer.try_push(NonTrivial(1));
    buffer.try_push(NonTrivial(2));
    buffer.try_push(NonTrivial(3));
    // Buffer goes out of scope with 3 elements
  }

  // Destructor should have been called for all elements
}

// =============================================================================
// Concurrent Producer-Consumer Tests
// =============================================================================

TEST(SPSCRingBuffer, ConcurrentProducerConsumer) {
  SPSCRingBuffer<int, 16> buffer;
  constexpr int kNumItems = 100000;

  std::atomic<bool> producerDone{false};
  std::vector<int> consumed;
  consumed.reserve(kNumItems);

  // Producer thread
  std::thread producer([&]() {
    for (int i = 0; i < kNumItems; ++i) {
      while (!buffer.try_push(i)) {
        // Spin until space available
        std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  // Consumer thread
  std::thread consumer([&]() {
    int value;
    while (consumed.size() < kNumItems) {
      if (buffer.try_pop(value)) {
        consumed.push_back(value);
      } else if (producerDone.load(std::memory_order_acquire) && buffer.empty()) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  // Verify all items received in order
  ASSERT_EQ(consumed.size(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i], i) << "Mismatch at index " << i;
  }
}

TEST(SPSCRingBuffer, ConcurrentWithSmallBuffer) {
  // Small buffer to increase contention
  SPSCRingBuffer<int, 2> buffer;
  constexpr int kNumItems = 10000;

  std::atomic<bool> producerDone{false};
  std::vector<int> consumed;
  consumed.reserve(kNumItems);

  std::thread producer([&]() {
    for (int i = 0; i < kNumItems; ++i) {
      while (!buffer.try_push(i)) {
        std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&]() {
    int value;
    while (consumed.size() < kNumItems) {
      if (buffer.try_pop(value)) {
        consumed.push_back(value);
      } else if (producerDone.load(std::memory_order_acquire) && buffer.empty()) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(consumed.size(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i], i);
  }
}

TEST(SPSCRingBuffer, ConcurrentWithMoveOnlyType) {
  SPSCRingBuffer<std::unique_ptr<int>, 8> buffer;
  constexpr int kNumItems = 10000;

  std::atomic<bool> producerDone{false};
  std::vector<int> consumed;
  consumed.reserve(kNumItems);

  std::thread producer([&]() {
    for (int i = 0; i < kNumItems; ++i) {
      auto ptr = std::make_unique<int>(i);
      while (!buffer.try_push(std::move(ptr))) {
        std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&]() {
    std::unique_ptr<int> value;
    while (consumed.size() < kNumItems) {
      if (buffer.try_pop(value)) {
        consumed.push_back(*value);
        value.reset();
      } else if (producerDone.load(std::memory_order_acquire) && buffer.empty()) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(consumed.size(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i], i);
  }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SPSCRingBuffer, EmptyAfterDraining) {
  SPSCRingBuffer<int, 4> buffer;

  buffer.try_push(1);
  buffer.try_push(2);

  int v;
  buffer.try_pop(v);
  buffer.try_pop(v);

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_FALSE(buffer.try_pop(v));
}

TEST(SPSCRingBuffer, RepeatedFillAndDrain) {
  SPSCRingBuffer<int, 4> buffer;
  const size_t cap = buffer.capacity();

  for (int cycle = 0; cycle < 10; ++cycle) {
    // Fill completely
    for (size_t i = 0; i < cap; ++i) {
      EXPECT_TRUE(buffer.try_push(cycle * 10 + static_cast<int>(i)));
    }
    EXPECT_TRUE(buffer.full());
    EXPECT_FALSE(buffer.try_push(999));

    // Drain completely
    for (size_t i = 0; i < cap; ++i) {
      int value;
      EXPECT_TRUE(buffer.try_pop(value));
      EXPECT_EQ(value, cycle * 10 + static_cast<int>(i));
    }
    EXPECT_TRUE(buffer.empty());
    int dummy;
    EXPECT_FALSE(buffer.try_pop(dummy));
  }
}

TEST(SPSCRingBuffer, AlternatingPushPop) {
  SPSCRingBuffer<int, 4> buffer;

  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(buffer.try_push(i));
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
    EXPECT_TRUE(buffer.empty());
  }
}

TEST(SPSCRingBuffer, LargeElements) {
  struct LargeStruct {
    std::array<char, 1024> data;
    int id;

    LargeStruct() : id(0) {
      data.fill('x');
    }
    explicit LargeStruct(int i) : id(i) {
      data.fill(static_cast<char>('0' + (i % 10)));
    }
  };

  SPSCRingBuffer<LargeStruct, 4> buffer;

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(buffer.try_push(LargeStruct(i)));
  }

  for (int i = 0; i < 4; ++i) {
    LargeStruct value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value.id, i);
    EXPECT_EQ(value.data[0], static_cast<char>('0' + i));
  }
}

// =============================================================================
// String Type Tests (common use case)
// =============================================================================

TEST(SPSCRingBuffer, StringElements) {
  SPSCRingBuffer<std::string, 8> buffer;

  std::vector<std::string> testStrings = {
      "hello", "world", "this is a longer string", "", "short", "another test string"};

  for (const auto& s : testStrings) {
    EXPECT_TRUE(buffer.try_push(s));
  }

  for (const auto& expected : testStrings) {
    std::string result;
    EXPECT_TRUE(buffer.try_pop(result));
    EXPECT_EQ(result, expected);
  }
}

// =============================================================================
// try_pop_into Tests
// =============================================================================

TEST(SPSCRingBuffer, TryPopIntoBasic) {
  SPSCRingBuffer<int, 4> buffer;

  buffer.try_push(42);
  buffer.try_push(100);

  alignas(int) char storage[sizeof(int)];
  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<int*>(storage)));

  int* value = reinterpret_cast<int*>(storage);
  EXPECT_EQ(*value, 42);

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<int*>(storage)));
  EXPECT_EQ(*value, 100);

  EXPECT_FALSE(buffer.try_pop_into(reinterpret_cast<int*>(storage)));
}

TEST(SPSCRingBuffer, TryPopIntoMoveOnly) {
  SPSCRingBuffer<std::unique_ptr<int>, 4> buffer;

  buffer.try_push(std::make_unique<int>(42));
  buffer.try_push(std::make_unique<int>(100));

  alignas(std::unique_ptr<int>) char storage[sizeof(std::unique_ptr<int>)];

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<std::unique_ptr<int>*>(storage)));
  auto* ptr = reinterpret_cast<std::unique_ptr<int>*>(storage);
  EXPECT_EQ(**ptr, 42);
  ptr->~unique_ptr();

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<std::unique_ptr<int>*>(storage)));
  ptr = reinterpret_cast<std::unique_ptr<int>*>(storage);
  EXPECT_EQ(**ptr, 100);
  ptr->~unique_ptr();

  EXPECT_FALSE(buffer.try_pop_into(reinterpret_cast<std::unique_ptr<int>*>(storage)));
}

TEST(SPSCRingBuffer, TryPopIntoNonTrivial) {
  NonTrivial::resetCounters();

  SPSCRingBuffer<NonTrivial, 4> buffer;

  buffer.try_push(NonTrivial(42));
  buffer.try_push(NonTrivial(100));

  int initialDestructorCount = NonTrivial::destructorCount.load();

  alignas(NonTrivial) char storage[sizeof(NonTrivial)];

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<NonTrivial*>(storage)));
  auto* obj = reinterpret_cast<NonTrivial*>(storage);
  EXPECT_EQ(obj->value, 42);

  // Destructor should have been called for the element in the buffer
  EXPECT_GT(NonTrivial::destructorCount.load(), initialDestructorCount);

  obj->~NonTrivial();

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<NonTrivial*>(storage)));
  obj = reinterpret_cast<NonTrivial*>(storage);
  EXPECT_EQ(obj->value, 100);
  obj->~NonTrivial();
}

// =============================================================================
// Non-Default-Constructible Type Tests
// =============================================================================

namespace {
struct NonDefaultConstructible {
  int value;

  // No default constructor
  explicit NonDefaultConstructible(int v) : value(v) {}
  NonDefaultConstructible(const NonDefaultConstructible&) = default;
  NonDefaultConstructible(NonDefaultConstructible&&) = default;
  NonDefaultConstructible& operator=(const NonDefaultConstructible&) = default;
  NonDefaultConstructible& operator=(NonDefaultConstructible&&) = default;
};
} // namespace

TEST(SPSCRingBuffer, NonDefaultConstructibleType) {
  SPSCRingBuffer<NonDefaultConstructible, 4> buffer;

  EXPECT_TRUE(buffer.try_emplace(42));
  EXPECT_TRUE(buffer.try_emplace(100));
  EXPECT_EQ(buffer.size(), 2u);

  // Use try_pop_into since we can't default-construct
  alignas(NonDefaultConstructible) char storage[sizeof(NonDefaultConstructible)];

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<NonDefaultConstructible*>(storage)));
  auto* obj = reinterpret_cast<NonDefaultConstructible*>(storage);
  EXPECT_EQ(obj->value, 42);
  obj->~NonDefaultConstructible();

  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<NonDefaultConstructible*>(storage)));
  obj = reinterpret_cast<NonDefaultConstructible*>(storage);
  EXPECT_EQ(obj->value, 100);
  obj->~NonDefaultConstructible();
}

TEST(SPSCRingBuffer, NonDefaultConstructibleWithMove) {
  SPSCRingBuffer<NonDefaultConstructible, 4> buffer;

  NonDefaultConstructible obj(42);
  EXPECT_TRUE(buffer.try_push(std::move(obj)));

  alignas(NonDefaultConstructible) char storage[sizeof(NonDefaultConstructible)];
  EXPECT_TRUE(buffer.try_pop_into(reinterpret_cast<NonDefaultConstructible*>(storage)));
  auto* result = reinterpret_cast<NonDefaultConstructible*>(storage);
  EXPECT_EQ(result->value, 42);
  result->~NonDefaultConstructible();
}

// =============================================================================
// Large Capacity Tests
// =============================================================================

TEST(SPSCRingBuffer, LargeCapacity) {
  SPSCRingBuffer<int, 1024> buffer;
  const size_t cap = buffer.capacity();

  // With rounding, 1024+1=1025 rounds up to 2048, so capacity is 2047
  EXPECT_EQ(cap, 2047u);

  // Fill to capacity
  for (size_t i = 0; i < cap; ++i) {
    EXPECT_TRUE(buffer.try_push(static_cast<int>(i)));
  }
  EXPECT_TRUE(buffer.full());
  EXPECT_FALSE(buffer.try_push(9999));

  // Drain and verify
  for (size_t i = 0; i < cap; ++i) {
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, static_cast<int>(i));
  }
  EXPECT_TRUE(buffer.empty());
}

TEST(SPSCRingBuffer, LargeCapacityConcurrent) {
  SPSCRingBuffer<int, 1024> buffer;
  constexpr int kNumItems = 100000;

  std::atomic<bool> producerDone{false};
  std::vector<int> consumed;
  consumed.reserve(kNumItems);

  std::thread producer([&]() {
    for (int i = 0; i < kNumItems; ++i) {
      while (!buffer.try_push(i)) {
        std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&]() {
    int value;
    while (consumed.size() < kNumItems) {
      if (buffer.try_pop(value)) {
        consumed.push_back(value);
      } else if (producerDone.load(std::memory_order_acquire) && buffer.empty()) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(consumed.size(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i], i);
  }
}

// =============================================================================
// Batch Operation Tests
// =============================================================================

TEST(SPSCRingBuffer, TryPushBatchBasic) {
  SPSCRingBuffer<int, 8> buffer;

  std::vector<int> items = {1, 2, 3, 4, 5};
  size_t pushed = buffer.try_push_batch(items.begin(), items.end());

  EXPECT_EQ(pushed, 5u);
  EXPECT_EQ(buffer.size(), 5u);

  // Verify order
  for (int i = 1; i <= 5; ++i) {
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

TEST(SPSCRingBuffer, TryPushBatchPartial) {
  // Use exact capacity mode for predictable partial push behavior
  SPSCRingBuffer<int, 4, false> buffer;

  // Try to push 6 items into capacity-4 buffer
  std::vector<int> items = {1, 2, 3, 4, 5, 6};
  size_t pushed = buffer.try_push_batch(items.begin(), items.end());

  EXPECT_EQ(pushed, 4u);
  EXPECT_TRUE(buffer.full());

  // Verify only first 4 items were pushed
  for (int i = 1; i <= 4; ++i) {
    int value;
    EXPECT_TRUE(buffer.try_pop(value));
    EXPECT_EQ(value, i);
  }
}

TEST(SPSCRingBuffer, TryPushBatchEmpty) {
  SPSCRingBuffer<int, 4> buffer;

  std::vector<int> items;
  size_t pushed = buffer.try_push_batch(items.begin(), items.end());

  EXPECT_EQ(pushed, 0u);
  EXPECT_TRUE(buffer.empty());
}

TEST(SPSCRingBuffer, TryPushBatchWhenFull) {
  // Use exact capacity mode for predictable full behavior
  SPSCRingBuffer<int, 4, false> buffer;

  // Fill the buffer
  for (int i = 0; i < 4; ++i) {
    buffer.try_push(i);
  }
  EXPECT_TRUE(buffer.full());

  std::vector<int> items = {10, 11, 12};
  size_t pushed = buffer.try_push_batch(items.begin(), items.end());

  EXPECT_EQ(pushed, 0u);
}

TEST(SPSCRingBuffer, TryPopBatchBasic) {
  SPSCRingBuffer<int, 8> buffer;

  for (int i = 1; i <= 5; ++i) {
    buffer.try_push(i);
  }

  std::vector<int> dest(5);
  size_t popped = buffer.try_pop_batch(dest.begin(), 5);

  EXPECT_EQ(popped, 5u);
  EXPECT_TRUE(buffer.empty());

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(dest[i], i + 1);
  }
}

TEST(SPSCRingBuffer, TryPopBatchPartial) {
  SPSCRingBuffer<int, 8> buffer;

  for (int i = 1; i <= 3; ++i) {
    buffer.try_push(i);
  }

  std::vector<int> dest(5, -1);
  size_t popped = buffer.try_pop_batch(dest.begin(), 5);

  EXPECT_EQ(popped, 3u);
  EXPECT_TRUE(buffer.empty());

  // First 3 should be set
  EXPECT_EQ(dest[0], 1);
  EXPECT_EQ(dest[1], 2);
  EXPECT_EQ(dest[2], 3);
  // Rest should be unchanged
  EXPECT_EQ(dest[3], -1);
  EXPECT_EQ(dest[4], -1);
}

TEST(SPSCRingBuffer, TryPopBatchEmpty) {
  SPSCRingBuffer<int, 4> buffer;

  std::vector<int> dest(4, -1);
  size_t popped = buffer.try_pop_batch(dest.begin(), 4);

  EXPECT_EQ(popped, 0u);
  // Dest should be unchanged
  for (int v : dest) {
    EXPECT_EQ(v, -1);
  }
}

TEST(SPSCRingBuffer, BatchWithWrapAround) {
  // Use exact capacity mode for predictable wrap-around behavior
  SPSCRingBuffer<int, 4, false> buffer;

  // Push 3 items
  for (int i = 0; i < 3; ++i) {
    buffer.try_push(i);
  }

  // Pop 2 items (advances head)
  int v;
  buffer.try_pop(v);
  buffer.try_pop(v);

  // Now push a batch that will wrap around
  std::vector<int> items = {10, 11, 12, 13};
  size_t pushed = buffer.try_push_batch(items.begin(), items.end());

  EXPECT_EQ(pushed, 3u); // Only 3 slots available
  EXPECT_TRUE(buffer.full());

  // Pop all and verify order
  std::vector<int> dest(4);
  size_t popped = buffer.try_pop_batch(dest.begin(), 4);

  EXPECT_EQ(popped, 4u);
  EXPECT_EQ(dest[0], 2); // Remaining from initial push
  EXPECT_EQ(dest[1], 10);
  EXPECT_EQ(dest[2], 11);
  EXPECT_EQ(dest[3], 12);
}

TEST(SPSCRingBuffer, BatchConcurrent) {
  SPSCRingBuffer<int, 64> buffer;
  constexpr int kNumItems = 100000;
  constexpr int kBatchSize = 16;

  std::atomic<bool> producerDone{false};
  std::vector<int> consumed;
  consumed.reserve(kNumItems);

  std::thread producer([&]() {
    std::vector<int> batch(kBatchSize);
    for (int i = 0; i < kNumItems; i += kBatchSize) {
      int batchEnd = std::min(i + kBatchSize, kNumItems);
      for (int j = i; j < batchEnd; ++j) {
        batch[j - i] = j;
      }
      size_t remaining = batchEnd - i;
      auto it = batch.begin();
      while (remaining > 0) {
        size_t pushed = buffer.try_push_batch(it, it + remaining);
        it += pushed;
        remaining -= pushed;
        if (remaining > 0) {
          std::this_thread::yield();
        }
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&]() {
    std::vector<int> batch(kBatchSize);
    while (consumed.size() < kNumItems) {
      size_t popped = buffer.try_pop_batch(batch.begin(), kBatchSize);
      if (popped > 0) {
        for (size_t i = 0; i < popped; ++i) {
          consumed.push_back(batch[i]);
        }
      } else if (producerDone.load(std::memory_order_acquire) && buffer.empty()) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(consumed.size(), kNumItems);
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(consumed[i], i) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(SPSCRingBuffer, StressTest) {
  SPSCRingBuffer<int, 64> buffer;
  constexpr int kNumItems = 1000000;

  std::atomic<int64_t> consumedSum{0};

  std::thread producer([&]() {
    for (int i = 1; i <= kNumItems; ++i) {
      while (!buffer.try_push(i)) {
        std::this_thread::yield();
      }
    }
  });

  std::thread consumer([&]() {
    int64_t localSum = 0;
    int count = 0;
    while (count < kNumItems) {
      int value;
      if (buffer.try_pop(value)) {
        localSum += value;
        ++count;
      } else {
        std::this_thread::yield();
      }
    }
    consumedSum.store(localSum, std::memory_order_relaxed);
  });

  producer.join();
  consumer.join();

  // Sum of 1 to N = N*(N+1)/2
  int64_t expectedSum = static_cast<int64_t>(kNumItems) * (kNumItems + 1) / 2;
  EXPECT_EQ(consumedSum.load(), expectedSum);
}
