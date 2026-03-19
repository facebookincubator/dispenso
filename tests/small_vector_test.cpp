/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/small_vector.h>

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
