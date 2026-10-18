/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/small_vector.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

using dispenso::SmallVector;

// Test helper to track construction/destruction
struct Tracked {
  static int constructed;
  static int destructed;
  static int moveConstructed;
  static int copyConstructed;

  static void reset() {
    constructed = 0;
    destructed = 0;
    moveConstructed = 0;
    copyConstructed = 0;
  }

  int value;

  Tracked() : value(0) {
    ++constructed;
  }
  explicit Tracked(int v) : value(v) {
    ++constructed;
  }
  Tracked(const Tracked& other) : value(other.value) {
    ++constructed;
    ++copyConstructed;
  }
  Tracked(Tracked&& other) noexcept : value(other.value) {
    other.value = -1;
    ++constructed;
    ++moveConstructed;
  }
  ~Tracked() {
    ++destructed;
  }
  Tracked& operator=(const Tracked& other) {
    value = other.value;
    return *this;
  }
  Tracked& operator=(Tracked&& other) noexcept {
    value = other.value;
    other.value = -1;
    return *this;
  }
};

int Tracked::constructed = 0;
int Tracked::destructed = 0;
int Tracked::moveConstructed = 0;
int Tracked::copyConstructed = 0;

// --- Construction Tests ---

TEST(SmallVector, DefaultConstruction) {
  SmallVector<int> v;
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0u);
  EXPECT_GE(v.capacity(), 4u);
}

TEST(SmallVector, ConstructionWithSize) {
  SmallVector<int> v(3);
  EXPECT_EQ(v.size(), 3u);
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], 0);
  }
}

TEST(SmallVector, ConstructionWithSizeAndValue) {
  SmallVector<int> v(5, 42);
  EXPECT_EQ(v.size(), 5u);
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], 42);
  }
}

TEST(SmallVector, InitializerListConstruction) {
  SmallVector<int> v = {1, 2, 3, 4, 5};
  EXPECT_EQ(v.size(), 5u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[4], 5);
}

TEST(SmallVector, CopyConstruction) {
  SmallVector<int> v1 = {1, 2, 3};
  SmallVector<int> v2(v1);
  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);
  // Verify independence
  v1[0] = 100;
  EXPECT_EQ(v2[0], 1);
}

TEST(SmallVector, MoveConstructionInline) {
  SmallVector<int, 8> v1 = {1, 2, 3};
  SmallVector<int, 8> v2(std::move(v1));
  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2[2], 3);
  EXPECT_EQ(v1.size(), 0u);
}

TEST(SmallVector, MoveConstructionHeap) {
  SmallVector<int, 2> v1 = {1, 2, 3, 4, 5}; // Forces heap allocation
  const int* oldData = v1.data();
  SmallVector<int, 2> v2(std::move(v1));
  EXPECT_EQ(v2.size(), 5u);
  EXPECT_EQ(v2.data(), oldData); // Should steal the pointer
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[4], 5);
  EXPECT_EQ(v1.size(), 0u);
}

// --- Assignment Tests ---

TEST(SmallVector, CopyAssignment) {
  SmallVector<int> v1 = {1, 2, 3};
  SmallVector<int> v2;
  v2 = v1;
  EXPECT_EQ(v2.size(), 3u);
  EXPECT_EQ(v2[0], 1);
  v1[0] = 100;
  EXPECT_EQ(v2[0], 1);
}

TEST(SmallVector, MoveAssignment) {
  SmallVector<int, 2> v1 = {1, 2, 3, 4, 5};
  SmallVector<int, 2> v2;
  v2 = std::move(v1);
  EXPECT_EQ(v2.size(), 5u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v1.size(), 0u);
}

TEST(SmallVector, MoveAssignmentHeapToHeap) {
  // Target has heap storage that needs to be freed
  SmallVector<int, 2> v1 = {1, 2, 3, 4, 5};
  SmallVector<int, 2> v2 = {10, 20, 30, 40}; // Also on heap
  v2 = std::move(v1);
  EXPECT_EQ(v2.size(), 5u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v1.size(), 0u);
}

TEST(SmallVector, MoveAssignmentInlineToHeap) {
  // Target has heap storage, source is inline
  SmallVector<int, 4> v1 = {1, 2}; // Inline
  SmallVector<int, 4> v2 = {10, 20, 30, 40, 50, 60}; // Heap
  v2 = std::move(v1);
  EXPECT_EQ(v2.size(), 2u);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v1.size(), 0u);
}

TEST(SmallVector, SelfAssignment) {
  SmallVector<int> v = {1, 2, 3};
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
  v = v;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 1);
}

// --- Element Access Tests ---

TEST(SmallVector, OperatorBracket) {
  SmallVector<int> v = {10, 20, 30};
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
  v[1] = 25;
  EXPECT_EQ(v[1], 25);
}

TEST(SmallVector, FrontAndBack) {
  SmallVector<int> v = {1, 2, 3};
  EXPECT_EQ(v.front(), 1);
  EXPECT_EQ(v.back(), 3);
  v.front() = 10;
  v.back() = 30;
  EXPECT_EQ(v.front(), 10);
  EXPECT_EQ(v.back(), 30);
}

TEST(SmallVector, Data) {
  SmallVector<int> v = {1, 2, 3};
  int* ptr = v.data();
  EXPECT_EQ(ptr[0], 1);
  EXPECT_EQ(ptr[1], 2);
  EXPECT_EQ(ptr[2], 3);
}

// --- Capacity Tests ---

TEST(SmallVector, Empty) {
  SmallVector<int> v;
  EXPECT_TRUE(v.empty());
  v.push_back(1);
  EXPECT_FALSE(v.empty());
}

TEST(SmallVector, Size) {
  SmallVector<int> v;
  EXPECT_EQ(v.size(), 0u);
  v.push_back(1);
  EXPECT_EQ(v.size(), 1u);
  v.push_back(2);
  EXPECT_EQ(v.size(), 2u);
}

// --- Modifier Tests ---

TEST(SmallVector, PushBack) {
  SmallVector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
}

TEST(SmallVector, PushBackGrowth) {
  SmallVector<int, 2> v;
  v.push_back(1);
  v.push_back(2);
  EXPECT_EQ(v.capacity(), 2u);
  v.push_back(3); // Should trigger growth
  EXPECT_GT(v.capacity(), 2u);
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[2], 3);
}

TEST(SmallVector, EmplaceBack) {
  SmallVector<std::pair<int, int>> v;
  v.emplace_back(1, 2);
  v.emplace_back(3, 4);
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0].first, 1);
  EXPECT_EQ(v[0].second, 2);
  EXPECT_EQ(v[1].first, 3);
  EXPECT_EQ(v[1].second, 4);
}

TEST(SmallVector, PopBack) {
  SmallVector<int> v = {1, 2, 3};
  v.pop_back();
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v.back(), 2);
  v.pop_back();
  EXPECT_EQ(v.size(), 1u);
  EXPECT_EQ(v.back(), 1);
}

TEST(SmallVector, Clear) {
  SmallVector<int> v = {1, 2, 3, 4, 5};
  v.clear();
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0u);
}

TEST(SmallVector, ResizeGrow) {
  SmallVector<int> v = {1, 2};
  v.resize(5);
  EXPECT_EQ(v.size(), 5u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 0);
  EXPECT_EQ(v[3], 0);
  EXPECT_EQ(v[4], 0);
}

TEST(SmallVector, ResizeShrink) {
  SmallVector<int> v = {1, 2, 3, 4, 5};
  v.resize(2);
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
}

TEST(SmallVector, ResizeWithValue) {
  SmallVector<int> v = {1, 2};
  v.resize(5, 42);
  EXPECT_EQ(v.size(), 5u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 42);
  EXPECT_EQ(v[3], 42);
  EXPECT_EQ(v[4], 42);
}

TEST(SmallVector, EraseMiddle) {
  SmallVector<int> v = {1, 2, 3, 4, 5};
  auto it = v.erase(v.begin() + 2);
  EXPECT_EQ(v.size(), 4u);
  EXPECT_EQ(*it, 4);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 4);
  EXPECT_EQ(v[3], 5);
}

TEST(SmallVector, EraseFirst) {
  SmallVector<int> v = {1, 2, 3};
  v.erase(v.begin());
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 3);
}

TEST(SmallVector, EraseLast) {
  SmallVector<int> v = {1, 2, 3};
  v.erase(v.end() - 1);
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
}

// --- Iterator Tests ---

TEST(SmallVector, RangeBasedFor) {
  SmallVector<int> v = {1, 2, 3, 4, 5};
  int sum = 0;
  for (int x : v) {
    sum += x;
  }
  EXPECT_EQ(sum, 15);
}

TEST(SmallVector, ConstIteration) {
  const SmallVector<int> v = {1, 2, 3};
  int sum = 0;
  for (const int& x : v) {
    sum += x;
  }
  EXPECT_EQ(sum, 6);
}

TEST(SmallVector, BeginEnd) {
  SmallVector<int> v = {1, 2, 3};
  EXPECT_EQ(*v.begin(), 1);
  EXPECT_EQ(*(v.end() - 1), 3);
  EXPECT_EQ(v.end() - v.begin(), 3);
}

TEST(SmallVector, CBeginCEnd) {
  SmallVector<int> v = {1, 2, 3};
  EXPECT_EQ(*v.cbegin(), 1);
  EXPECT_EQ(*(v.cend() - 1), 3);
  EXPECT_EQ(v.cend() - v.cbegin(), 3);
}

TEST(SmallVector, ConstAccessors) {
  const SmallVector<int> v = {10, 20, 30};
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[2], 30);
  EXPECT_EQ(v.front(), 10);
  EXPECT_EQ(v.back(), 30);
  EXPECT_EQ(v.data()[1], 20);
  EXPECT_EQ(*v.begin(), 10);
  EXPECT_EQ(*(v.end() - 1), 30);
}

// --- Object Lifetime Tests ---

TEST(SmallVector, DestructorsCalled) {
  Tracked::reset();
  {
    SmallVector<Tracked, 4> v;
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    EXPECT_EQ(Tracked::constructed, 3);
    EXPECT_EQ(Tracked::destructed, 0);
  }
  EXPECT_EQ(Tracked::constructed, Tracked::destructed);
}

TEST(SmallVector, DestructorsCalledOnClear) {
  Tracked::reset();
  SmallVector<Tracked, 4> v;
  v.emplace_back(1);
  v.emplace_back(2);
  int beforeClear = Tracked::destructed;
  v.clear();
  EXPECT_EQ(Tracked::destructed - beforeClear, 2);
}

TEST(SmallVector, DestructorsCalledOnGrowth) {
  Tracked::reset();
  SmallVector<Tracked, 2> v;
  v.emplace_back(1);
  v.emplace_back(2);
  int beforeGrowth = Tracked::destructed;
  v.emplace_back(3); // Triggers growth, should move and destroy old
  EXPECT_EQ(Tracked::destructed - beforeGrowth, 2); // Old elements destroyed
  EXPECT_EQ(Tracked::moveConstructed, 2); // Old elements moved to new storage
}

TEST(SmallVector, MoveSemantics) {
  Tracked::reset();
  SmallVector<Tracked, 2> v1;
  v1.emplace_back(1);
  v1.emplace_back(2);

  int constructedBefore = Tracked::constructed;
  SmallVector<Tracked, 2> v2(std::move(v1));
  // Inline move should move-construct elements
  EXPECT_EQ(Tracked::constructed - constructedBefore, 2);
  EXPECT_EQ(v2.size(), 2u);
  EXPECT_EQ(v2[0].value, 1);
  EXPECT_EQ(v2[1].value, 2);
}

// --- String Tests (non-trivial type) ---

TEST(SmallVector, StringOperations) {
  SmallVector<std::string, 4> v;
  v.push_back("hello");
  v.push_back("world");
  v.emplace_back("!");
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], "hello");
  EXPECT_EQ(v[1], "world");
  EXPECT_EQ(v[2], "!");
}

TEST(SmallVector, StringGrowth) {
  SmallVector<std::string, 2> v;
  v.push_back("one");
  v.push_back("two");
  v.push_back("three");
  v.push_back("four");
  EXPECT_EQ(v.size(), 4u);
  EXPECT_EQ(v[0], "one");
  EXPECT_EQ(v[3], "four");
}

// --- Edge Cases ---

TEST(SmallVector, LargeInlineCapacity) {
  SmallVector<int, 100> v;
  for (int i = 0; i < 100; ++i) {
    v.push_back(i);
  }
  EXPECT_EQ(v.size(), 100u);
  EXPECT_EQ(v.capacity(), 100u); // Should still be inline
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(v[static_cast<size_t>(i)], i);
  }
}

TEST(SmallVector, DifferentInlineCapacities) {
  SmallVector<int, 2> v2 = {1, 2};
  SmallVector<int, 8> v8 = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(v2.size(), 2u);
  EXPECT_EQ(v8.size(), 8u);
}

// --- Heap regrowth tests (exceed 2*N to trigger heap→heap growth) ---

TEST(SmallVector, HeapRegrowth) {
  SmallVector<int, 4> v;
  // Push 20 elements: inline(4) → heap(8) → heap(16) → heap(32)
  for (int i = 0; i < 20; ++i) {
    v.push_back(i);
  }
  EXPECT_EQ(v.size(), 20u);
  EXPECT_GE(v.capacity(), 20u);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(v[static_cast<size_t>(i)], i);
  }
}

TEST(SmallVector, HeapRegrowthWithReserve) {
  SmallVector<int, 4> v;
  v.reserve(100);
  EXPECT_GE(v.capacity(), 100u);
  for (int i = 0; i < 100; ++i) {
    v.push_back(i);
  }
  EXPECT_EQ(v.size(), 100u);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(v[static_cast<size_t>(i)], i);
  }
}

TEST(SmallVector, HeapRegrowthTracked) {
  Tracked::reset();
  {
    SmallVector<Tracked, 2> v;
    for (int i = 0; i < 10; ++i) {
      v.emplace_back(i);
    }
    EXPECT_EQ(v.size(), 10u);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(v[static_cast<size_t>(i)].value, i);
    }
  }
  EXPECT_EQ(Tracked::constructed, Tracked::destructed);
}

TEST(SmallVector, EraseOnHeap) {
  SmallVector<int, 2> v;
  for (int i = 0; i < 8; ++i) {
    v.push_back(i);
  }
  // Erase from the middle while on heap
  v.erase(v.begin() + 3);
  EXPECT_EQ(v.size(), 7u);
  EXPECT_EQ(v[0], 0);
  EXPECT_EQ(v[1], 1);
  EXPECT_EQ(v[2], 2);
  EXPECT_EQ(v[3], 4); // was 3, now shifted
  EXPECT_EQ(v[6], 7);
}

TEST(SmallVector, PopBackOnHeap) {
  SmallVector<int, 2> v;
  for (int i = 0; i < 8; ++i) {
    v.push_back(i);
  }
  v.pop_back();
  v.pop_back();
  EXPECT_EQ(v.size(), 6u);
  EXPECT_EQ(v.back(), 5);
}

// push_back/emplace_back of one of the vector's own elements must work across growth, as it does
// for std::vector: the argument refers into the storage that growth moves from and frees. Strings
// are long enough to defeat SSO, so a use-after-move or use-after-free is visible (and ASAN-able).
TEST(SmallVector, PushBackOwnElementAcrossInlineToHeapGrowth) {
  const std::string a(64, 'a');
  const std::string b(64, 'b');
  SmallVector<std::string, 2> v;
  v.push_back(a);
  v.push_back(b);
  ASSERT_EQ(v.capacity(), 2u);
  v.push_back(v[0]);
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], a);
  EXPECT_EQ(v[1], b);
  EXPECT_EQ(v[2], a);
}

TEST(SmallVector, PushBackOwnElementAcrossHeapGrowth) {
  SmallVector<std::string, 2> v;
  for (int i = 0; i < 4; ++i) {
    v.push_back(std::string(64, static_cast<char>('a' + i)));
  }
  ASSERT_EQ(v.size(), v.capacity());
  v.push_back(v[1]);
  ASSERT_EQ(v.size(), 5u);
  EXPECT_EQ(v[1], std::string(64, 'b'));
  EXPECT_EQ(v[4], std::string(64, 'b'));
}

TEST(SmallVector, EmplaceBackOwnElementAcrossGrowth) {
  SmallVector<std::string, 1> v;
  v.emplace_back(64, 'x');
  ASSERT_EQ(v.size(), v.capacity());
  v.emplace_back(v.back());
  ASSERT_EQ(v.size(), v.capacity());
  // Constructor arguments taken from an element: std::string(const std::string&, pos, count).
  v.emplace_back(v[0], 0, 10);
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], std::string(64, 'x'));
  EXPECT_EQ(v[1], std::string(64, 'x'));
  EXPECT_EQ(v[2], std::string(10, 'x'));
}

namespace {
struct alignas(64) Aligned64 {
  int value;
};
struct alignas(128) Aligned128 {
  int value;
  Aligned128(int v) : value(v) {}
};

template <typename Vec>
bool dataIsAligned(const Vec& v, size_t alignment) {
  return reinterpret_cast<uintptr_t>(v.data()) % alignment == 0;
}
} // namespace

// Heap storage must honour alignof(T), not just the default new alignment.
TEST(SmallVector, HeapStorageHonoursOverAlignment64) {
  SmallVector<Aligned64, 2> v;
  for (int i = 0; i < 40; ++i) {
    v.push_back(Aligned64{i});
    ASSERT_TRUE(dataIsAligned(v, 64)) << "size " << v.size();
  }
  for (int i = 0; i < 40; ++i) {
    EXPECT_EQ(v[static_cast<size_t>(i)].value, i);
  }
  SmallVector<Aligned64, 2> copy(v);
  EXPECT_TRUE(dataIsAligned(copy, 64));
  v.reserve(1000);
  EXPECT_TRUE(dataIsAligned(v, 64));
  EXPECT_EQ(v[39].value, 39);
}

TEST(SmallVector, HeapStorageHonoursOverAlignment128) {
  SmallVector<Aligned128, 1> v;
  for (int i = 0; i < 40; ++i) {
    v.emplace_back(i);
    ASSERT_TRUE(dataIsAligned(v, 128)) << "size " << v.size();
  }
  v.resize(100, Aligned128(7));
  EXPECT_TRUE(dataIsAligned(v, 128));
  EXPECT_EQ(v[39].value, 39);
  EXPECT_EQ(v[99].value, 7);
  v.clear();
  EXPECT_TRUE(dataIsAligned(v, 128));
}

namespace {
// Copyable type whose move constructor is not noexcept, so growth must copy it (move_if_noexcept),
// and whose copy constructor can be told to throw after a number of copies.
struct ThrowingCopy {
  static int live;
  static int copiesBeforeThrow; // negative: never throw
  static int moves;
  int value;
  explicit ThrowingCopy(int v) : value(v) {
    ++live;
  }
  ThrowingCopy(const ThrowingCopy& o) : value(o.value) {
    if (copiesBeforeThrow == 0) {
#if defined(__cpp_exceptions)
      throw std::runtime_error("copy");
#else
      std::abort(); // Only the __cpp_exceptions tests arm the throw.
#endif // __cpp_exceptions
    }
    if (copiesBeforeThrow > 0) {
      --copiesBeforeThrow;
    }
    ++live;
  }
  ThrowingCopy(ThrowingCopy&& o) : value(o.value) { // not noexcept
    o.value = -1;
    ++moves;
    ++live;
  }
  ~ThrowingCopy() {
    --live;
  }
};
int ThrowingCopy::live = 0;
int ThrowingCopy::copiesBeforeThrow = -1;
int ThrowingCopy::moves = 0;

template <size_t N>
void fillThrowingCopy(SmallVector<ThrowingCopy, N>& v, int count) {
  for (int i = 0; i < count; ++i) {
    v.emplace_back(i);
  }
}

template <size_t N>
void expectThrowingCopyValues(const SmallVector<ThrowingCopy, N>& v, int count) {
  ASSERT_EQ(v.size(), static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(v[static_cast<size_t>(i)].value, i);
  }
}
} // namespace

// Growth uses move_if_noexcept: a type whose move may throw is copied, leaving the source intact.
TEST(SmallVector, GrowthCopiesWhenMoveMayThrow) {
  ThrowingCopy::live = 0;
  ThrowingCopy::copiesBeforeThrow = -1;
  ThrowingCopy::moves = 0;
  {
    SmallVector<ThrowingCopy, 2> v;
    fillThrowingCopy(v, 10);
    v.reserve(100);
    EXPECT_EQ(ThrowingCopy::moves, 0);
    expectThrowingCopyValues(v, 10);
    EXPECT_EQ(ThrowingCopy::live, 10);
  }
  EXPECT_EQ(ThrowingCopy::live, 0);
}

#if defined(__cpp_exceptions)
// If transferring elements during growth throws, the vector is unchanged and nothing leaks.
TEST(SmallVector, ReserveThrowLeavesVectorUnchanged) {
  ThrowingCopy::live = 0;
  ThrowingCopy::copiesBeforeThrow = -1;
  {
    // Inline -> heap.
    SmallVector<ThrowingCopy, 4> inl;
    fillThrowingCopy(inl, 4);
    ThrowingCopy::copiesBeforeThrow = 2;
    EXPECT_THROW(inl.reserve(16), std::runtime_error);
    ThrowingCopy::copiesBeforeThrow = -1;
    EXPECT_EQ(inl.capacity(), 4u);
    expectThrowingCopyValues(inl, 4);
    EXPECT_EQ(ThrowingCopy::live, 4);

    // Heap -> bigger heap.
    SmallVector<ThrowingCopy, 2> heap;
    fillThrowingCopy(heap, 6);
    const size_t cap = heap.capacity();
    const ThrowingCopy* oldData = heap.data();
    ThrowingCopy::copiesBeforeThrow = 3;
    EXPECT_THROW(heap.reserve(cap * 4), std::runtime_error);
    ThrowingCopy::copiesBeforeThrow = -1;
    EXPECT_EQ(heap.capacity(), cap);
    EXPECT_EQ(heap.data(), oldData);
    expectThrowingCopyValues(heap, 6);
    EXPECT_EQ(ThrowingCopy::live, 10);

    // resize() grows through the same path.
    ThrowingCopy::copiesBeforeThrow = 1;
    EXPECT_THROW(heap.resize(50, ThrowingCopy(7)), std::runtime_error);
    ThrowingCopy::copiesBeforeThrow = -1;
    expectThrowingCopyValues(heap, 6);
    EXPECT_EQ(ThrowingCopy::live, 10);

    // The vector is still fully usable.
    heap.reserve(64);
    fillThrowingCopy(heap, 1);
    EXPECT_EQ(heap.size(), 7u);
  }
  EXPECT_EQ(ThrowingCopy::live, 0);
}

TEST(SmallVector, PushBackGrowthThrowLeavesVectorUnchanged) {
  ThrowingCopy::live = 0;
  ThrowingCopy::copiesBeforeThrow = -1;
  {
    SmallVector<ThrowingCopy, 2> v;
    fillThrowingCopy(v, 4);
    ASSERT_EQ(v.size(), v.capacity());
    const ThrowingCopy* oldData = v.data();
    // The new element is constructed from an int, then transferring the second old element throws.
    ThrowingCopy::copiesBeforeThrow = 1;
    EXPECT_THROW(v.emplace_back(99), std::runtime_error);
    ThrowingCopy::copiesBeforeThrow = -1;
    EXPECT_EQ(v.data(), oldData);
    expectThrowingCopyValues(v, 4);
    EXPECT_EQ(ThrowingCopy::live, 4);
  }
  EXPECT_EQ(ThrowingCopy::live, 0);
}
#endif // __cpp_exceptions

#if defined(__cpp_exceptions)
namespace {
// Default and copy construction throw once `constructionsBeforeThrow` reaches zero (negative:
// never). Moves never throw, so growth itself cannot fail.
struct ThrowingConstruct {
  static int live;
  static int constructionsBeforeThrow;
  int value;
  ThrowingConstruct() : value(0) {
    countConstruction();
  }
  explicit ThrowingConstruct(int v) : value(v) {
    ++live;
  }
  ThrowingConstruct(const ThrowingConstruct& o) : value(o.value) {
    countConstruction();
  }
  ThrowingConstruct(ThrowingConstruct&& o) noexcept : value(o.value) {
    ++live;
  }
  ~ThrowingConstruct() {
    --live;
  }
  void countConstruction() {
    if (constructionsBeforeThrow == 0) {
      throw std::runtime_error("construct");
    }
    if (constructionsBeforeThrow > 0) {
      --constructionsBeforeThrow;
    }
    ++live;
  }
};
int ThrowingConstruct::live = 0;
int ThrowingConstruct::constructionsBeforeThrow = -1;

using TCVec = SmallVector<ThrowingConstruct, 4>;
} // namespace

// A constructor that throws partway must destroy the elements it built and free its heap storage
// (LeakSanitizer catches the latter). Covers inline and heap storage.
TEST(SmallVector, ConstructorThrowReleasesElements) {
  ThrowingConstruct::live = 0;
  const ThrowingConstruct proto(5);
  for (size_t count : {3u, 10u}) {
    ThrowingConstruct::constructionsBeforeThrow = 2;
    EXPECT_THROW(TCVec v(count), std::runtime_error);
    EXPECT_EQ(ThrowingConstruct::live, 1);

    ThrowingConstruct::constructionsBeforeThrow = 2;
    EXPECT_THROW(TCVec v(count, proto), std::runtime_error);
    EXPECT_EQ(ThrowingConstruct::live, 1);
  }

  ThrowingConstruct::constructionsBeforeThrow = 3;
  EXPECT_THROW((TCVec{proto, proto, proto, proto, proto, proto}), std::runtime_error);
  ThrowingConstruct::constructionsBeforeThrow = -1;
  EXPECT_EQ(ThrowingConstruct::live, 1);

  {
    TCVec small(3, proto);
    TCVec big(9, proto);
    EXPECT_EQ(ThrowingConstruct::live, 13);
    ThrowingConstruct::constructionsBeforeThrow = 1;
    EXPECT_THROW(TCVec copy(small), std::runtime_error);
    ThrowingConstruct::constructionsBeforeThrow = 5;
    EXPECT_THROW(TCVec copy(big), std::runtime_error);
    ThrowingConstruct::constructionsBeforeThrow = -1;
    EXPECT_EQ(ThrowingConstruct::live, 13);
  }
  EXPECT_EQ(ThrowingConstruct::live, 1);
}

// resize() that throws while constructing the new elements destroys the ones it built and keeps
// the original elements and size.
TEST(SmallVector, ResizeThrowKeepsSize) {
  ThrowingConstruct::live = 0;
  {
    TCVec v;
    for (int i = 0; i < 3; ++i) {
      v.emplace_back(i);
    }
    ThrowingConstruct::constructionsBeforeThrow = 4;
    EXPECT_THROW(v.resize(10), std::runtime_error);
    ThrowingConstruct::constructionsBeforeThrow = 2;
    EXPECT_THROW(v.resize(10, ThrowingConstruct(7)), std::runtime_error);
    ThrowingConstruct::constructionsBeforeThrow = -1;
    ASSERT_EQ(v.size(), 3u);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(v[static_cast<size_t>(i)].value, i);
    }
    EXPECT_EQ(ThrowingConstruct::live, 3);
    v.resize(10);
    EXPECT_EQ(ThrowingConstruct::live, 10);
  }
  EXPECT_EQ(ThrowingConstruct::live, 0);
}
#endif // __cpp_exceptions
