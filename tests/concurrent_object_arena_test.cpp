/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/concurrent_object_arena.h>
#include <dispenso/task_set.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(ConcurrentObjectArena, ParallelGrowBy) {
  constexpr size_t numTasks = 20;
  constexpr size_t numLoops = 100;
  constexpr size_t delta = 7;
  constexpr size_t bufSize = 16;

  dispenso::ConcurrentObjectArena<size_t> arena(bufSize);

  dispenso::TaskSet taskSet(dispenso::globalThreadPool());

  for (size_t ti = 0; ti < numTasks; ++ti) {
    taskSet.schedule([=, &arena]() {
      for (size_t i = 0; i < numLoops; i++) {
        const size_t p = arena.grow_by(delta);
        for (size_t j = 0; j < delta; j++) {
          arena[p + j] = ti * numLoops * delta + i;
        }
      }
    });
  }
  taskSet.wait();

  EXPECT_EQ(delta * numLoops * numTasks, arena.size());
  EXPECT_EQ(arena.capacity() / arena.numBuffers(), arena.getBufferSize(0));

  size_t totalSize = 0;
  for (size_t i = 0; i < arena.numBuffers(); ++i) {
    totalSize += arena.getBufferSize(i);
  }
  EXPECT_EQ(totalSize, arena.size());

  for (size_t i = 0; i < numLoops * numTasks; i++) {
    const size_t firstElement = arena[i * delta];
    for (size_t j = 1; j < delta; j++) {
      EXPECT_EQ(arena[i * delta + j], firstElement);
    }
  }
}

TEST(ConcurrentObjectArena, ObjectsConstuction) {
  constexpr size_t defaultValue = 17;
  constexpr size_t bufSize = 16;
  constexpr size_t smallGrow = bufSize / 3;
  constexpr size_t bigGrow = bufSize * 3;

  struct TestData {
    TestData() : value(defaultValue) {}
    size_t value;
  };

  dispenso::ConcurrentObjectArena<TestData>* arena =
      new dispenso::ConcurrentObjectArena<TestData>(bufSize);

  arena->grow_by(smallGrow);
  arena->grow_by(bigGrow);

  const size_t num = arena->size();
  for (size_t i = 0; i < num; ++i) {
    EXPECT_EQ((*arena)[i].value, defaultValue);
  }

  dispenso::ConcurrentObjectArena<TestData> copyArena(*arena);

  dispenso::ConcurrentObjectArena<TestData> copyAssignmentArena(bufSize / 2);
  copyAssignmentArena = *arena;

  EXPECT_EQ(copyArena.size(), arena->size());
  EXPECT_EQ(copyAssignmentArena.size(), arena->size());

  const size_t numBuffers = arena->numBuffers();
  std::vector<const TestData*> bufferPtrs(numBuffers);
  for (size_t i = 0; i < numBuffers; ++i) {
    bufferPtrs[i] = arena->getBuffer(i);
  }

  dispenso::ConcurrentObjectArena<TestData> moveArena(std::move(*arena));

  EXPECT_EQ(arena->size(), 0);
  EXPECT_EQ(arena->numBuffers(), 0);
  EXPECT_EQ(arena->capacity(), 0);

  delete arena;

  EXPECT_EQ(copyArena.numBuffers(), numBuffers);
  EXPECT_EQ(copyAssignmentArena.numBuffers(), numBuffers);

  for (size_t i = 0; i < num; ++i) {
    EXPECT_EQ(copyArena[i].value, defaultValue);
    EXPECT_EQ(copyAssignmentArena[i].value, defaultValue);
  }

  for (size_t i = 0; i < numBuffers; ++i) {
    EXPECT_NE(copyArena.getBuffer(i), bufferPtrs[i]);
    EXPECT_NE(copyAssignmentArena.getBuffer(i), bufferPtrs[i]);
    EXPECT_EQ(moveArena.getBuffer(i), bufferPtrs[i]);
  }
}

TEST(ConcurrentObjectArena, BufferSizeRounding) {
  // Test that non-power-of-2 buffer sizes are rounded up
  // minBuffSize = 10 should round up to 16 (next power of 2)
  dispenso::ConcurrentObjectArena<int> arena(10);

  arena.grow_by(20);

  // Capacity should be a multiple of 16 (the rounded buffer size)
  EXPECT_EQ(arena.capacity() % 16, 0u);

  // Buffer size should be 16
  EXPECT_EQ(arena.capacity() / arena.numBuffers(), 16u);
}

TEST(ConcurrentObjectArena, ExactPowerOfTwoBufferSize) {
  // Test that exact power-of-2 buffer sizes are preserved
  dispenso::ConcurrentObjectArena<int> arena(32);

  arena.grow_by(100);

  // Buffer size should be exactly 32
  EXPECT_EQ(arena.capacity() / arena.numBuffers(), 32u);
}

TEST(ConcurrentObjectArena, MoveAssignment) {
  dispenso::ConcurrentObjectArena<int> arena1(16);
  dispenso::ConcurrentObjectArena<int> arena2(8);

  arena1.grow_by(50);
  for (size_t i = 0; i < 50; ++i) {
    arena1[i] = static_cast<int>(i * 2);
  }

  arena2.grow_by(10);
  for (size_t i = 0; i < 10; ++i) {
    arena2[i] = static_cast<int>(i * 3);
  }

  size_t arena1Size = arena1.size();
  size_t arena1NumBuffers = arena1.numBuffers();
  size_t arena2Size = arena2.size();

  // Move assign arena1 to arena2 (uses swap internally)
  arena2 = std::move(arena1);

  // arena2 should now have arena1's contents
  EXPECT_EQ(arena2.size(), arena1Size);
  EXPECT_EQ(arena2.numBuffers(), arena1NumBuffers);
  for (size_t i = 0; i < arena1Size; ++i) {
    EXPECT_EQ(arena2[i], static_cast<int>(i * 2));
  }

  // arena1 now has arena2's old contents (swap behavior)
  EXPECT_EQ(arena1.size(), arena2Size);
}

TEST(ConcurrentObjectArena, SwapFunction) {
  dispenso::ConcurrentObjectArena<int> arena1(16);
  dispenso::ConcurrentObjectArena<int> arena2(32);

  arena1.grow_by(20);
  for (size_t i = 0; i < 20; ++i) {
    arena1[i] = 100;
  }

  arena2.grow_by(40);
  for (size_t i = 0; i < 40; ++i) {
    arena2[i] = 200;
  }

  size_t size1 = arena1.size();
  size_t size2 = arena2.size();

  swap(arena1, arena2);

  EXPECT_EQ(arena1.size(), size2);
  EXPECT_EQ(arena2.size(), size1);

  for (size_t i = 0; i < arena1.size(); ++i) {
    EXPECT_EQ(arena1[i], 200);
  }
  for (size_t i = 0; i < arena2.size(); ++i) {
    EXPECT_EQ(arena2[i], 100);
  }
}

TEST(ConcurrentObjectArena, ConstAccess) {
  dispenso::ConcurrentObjectArena<int> arena(16);
  arena.grow_by(10);
  for (size_t i = 0; i < 10; ++i) {
    arena[i] = static_cast<int>(i);
  }

  const dispenso::ConcurrentObjectArena<int>& constArena = arena;

  // Test const operator[]
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(constArena[i], static_cast<int>(i));
  }

  // Test const getBuffer
  const int* buf = constArena.getBuffer(0);
  EXPECT_NE(buf, nullptr);

  // Test const size, capacity, numBuffers
  EXPECT_EQ(constArena.size(), 10u);
  EXPECT_GE(constArena.capacity(), 10u);
  EXPECT_GE(constArena.numBuffers(), 1u);

  // Test const getBufferSize
  EXPECT_GT(constArena.getBufferSize(0), 0u);
}

TEST(ConcurrentObjectArena, DifferentIndexType) {
  // Test with uint32_t as Index type
  dispenso::ConcurrentObjectArena<int, uint32_t> arena(16);

  arena.grow_by(100);
  for (uint32_t i = 0; i < 100; ++i) {
    arena[i] = static_cast<int>(i * 5);
  }

  for (uint32_t i = 0; i < 100; ++i) {
    EXPECT_EQ(arena[i], static_cast<int>(i * 5));
  }

  EXPECT_EQ(arena.size(), 100u);
}

TEST(ConcurrentObjectArena, CustomAlignment) {
  // Test with custom alignment
  constexpr size_t kAlignment = 128;
  dispenso::ConcurrentObjectArena<int, size_t, kAlignment> arena(16);

  arena.grow_by(10);

  // Verify buffer pointer alignment
  const int* buf = arena.getBuffer(0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf) % kAlignment, 0u);
}

TEST(ConcurrentObjectArena, GrowByZero) {
  dispenso::ConcurrentObjectArena<int> arena(16);

  size_t initialSize = arena.size();
  arena.grow_by(0);
  EXPECT_EQ(arena.size(), initialSize);
}

TEST(ConcurrentObjectArena, SingleElementGrowth) {
  dispenso::ConcurrentObjectArena<int> arena(16);

  // Grow by 1 repeatedly
  for (int i = 0; i < 50; ++i) {
    size_t idx = arena.grow_by(1);
    arena[idx] = i;
  }

  EXPECT_EQ(arena.size(), 50u);

  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(arena[static_cast<size_t>(i)], i);
  }
}

// The buffer pointer table grows by doubling, so it usually has spare, unset slots past the last
// allocated buffer. Copying must only copy the allocated buffers.
TEST(ConcurrentObjectArena, CopyWithSparePointerSlots) {
  constexpr size_t bufSize = 16;
  dispenso::ConcurrentObjectArena<size_t> arena(bufSize);
  // 40 elements need 3 buffers; the pointer table doubles 2 -> 4, leaving one spare slot.
  const size_t first = arena.grow_by(40);
  ASSERT_EQ(first, 0u);
  ASSERT_EQ(arena.numBuffers(), 3u);
  for (size_t i = 0; i < arena.size(); ++i) {
    arena[i] = i * 3;
  }

  dispenso::ConcurrentObjectArena<size_t> copy(arena);
  ASSERT_EQ(copy.size(), arena.size());
  ASSERT_EQ(copy.numBuffers(), arena.numBuffers());
  for (size_t i = 0; i < copy.size(); ++i) {
    EXPECT_EQ(copy[i], i * 3);
  }

  // The copy must keep growing correctly into its spare slot and beyond.
  copy.grow_by(40);
  EXPECT_EQ(copy.size(), 80u);
  EXPECT_EQ(copy.numBuffers(), 6u);
  for (size_t i = 0; i < 40; ++i) {
    EXPECT_EQ(copy[i], i * 3);
  }

  dispenso::ConcurrentObjectArena<size_t> assigned(bufSize);
  assigned = arena;
  ASSERT_EQ(assigned.size(), arena.size());
  for (size_t i = 0; i < assigned.size(); ++i) {
    EXPECT_EQ(assigned[i], i * 3);
  }
}

namespace {
// Counts live instances so tests can check every constructed element is destroyed exactly once.
struct Counted {
  static std::atomic<int> live;
  Counted() {
    ++live;
  }
  ~Counted() {
    --live;
  }
  // Non-trivial destructor, so the arena must run it.
  int payload = 7;
};
std::atomic<int> Counted::live{0};
} // namespace

TEST(ConcurrentObjectArena, DestroysElements) {
  Counted::live = 0;
  {
    dispenso::ConcurrentObjectArena<Counted> arena(16, 5);
    arena.grow_by(40);
    arena.grow_by(0);
    arena.grow_by(3);
    EXPECT_EQ(Counted::live.load(), 48);
  }
  EXPECT_EQ(Counted::live.load(), 0);
}

TEST(ConcurrentObjectArena, DestroysElementsOfNonTrivialType) {
  auto shared = std::make_shared<int>(3);
  {
    dispenso::ConcurrentObjectArena<std::shared_ptr<int>, uint32_t> arena(4);
    const uint32_t first = arena.grow_by(10);
    for (uint32_t i = first; i < first + 10; ++i) {
      arena[i] = shared;
    }
    EXPECT_EQ(shared.use_count(), 11);
  }
  EXPECT_EQ(shared.use_count(), 1);
}

TEST(ConcurrentObjectArena, AssignmentDestroysPreviousElements) {
  Counted::live = 0;
  {
    dispenso::ConcurrentObjectArena<Counted> a(8, 20);
    dispenso::ConcurrentObjectArena<Counted> b(8, 3);
    EXPECT_EQ(Counted::live.load(), 23);
    a = std::move(b);
    EXPECT_EQ(a.size(), 3u);
    // The moved-from arena now owns a's former elements and destroys them.
    dispenso::ConcurrentObjectArena<Counted> sink(std::move(b));
    EXPECT_EQ(Counted::live.load(), 23);
  }
  EXPECT_EQ(Counted::live.load(), 0);
}

TEST(ConcurrentObjectArena, ParallelGrowByDestroysEverything) {
  Counted::live = 0;
  {
    dispenso::ConcurrentObjectArena<Counted> arena(16);
    dispenso::TaskSet taskSet(dispenso::globalThreadPool());
    for (size_t t = 0; t < 16; ++t) {
      taskSet.schedule([&arena]() {
        for (size_t i = 0; i < 50; ++i) {
          arena.grow_by(3);
        }
      });
    }
    taskSet.wait();
    EXPECT_EQ(Counted::live.load(), 16 * 50 * 3);
  }
  EXPECT_EQ(Counted::live.load(), 0);
}

#if defined(__cpp_exceptions)
namespace {
// Throws from the constructor that brings the live count to throwAt, optionally growing
// growDuringThrow first so the failing grow_by is no longer at the end of the arena.
struct ThrowingCounted {
  static int live;
  static int throwAt;
  static dispenso::ConcurrentObjectArena<ThrowingCounted>* growDuringThrow;
  ThrowingCounted() {
    if (live + 1 == throwAt) {
      throwAt = -1;
      if (growDuringThrow != nullptr) {
        growDuringThrow->grow_by(2);
      }
      throw std::runtime_error("ThrowingCounted");
    }
    ++live;
  }
  ~ThrowingCounted() {
    --live;
  }
};
int ThrowingCounted::live = 0;
int ThrowingCounted::throwAt = -1;
dispenso::ConcurrentObjectArena<ThrowingCounted>* ThrowingCounted::growDuringThrow = nullptr;
} // namespace

TEST(ConcurrentObjectArena, ThrowingConstructorLeavesGap) {
  ThrowingCounted::live = 0;
  {
    dispenso::ConcurrentObjectArena<ThrowingCounted> arena(4, 6);
    // Fail on the 4th element of the next grow, after it has crossed into a new buffer.
    ThrowingCounted::throwAt = 6 + 4;
    EXPECT_THROW(arena.grow_by(8), std::runtime_error);
    // The 3 elements built before the throw were destroyed; the 8 slots stay reserved.
    EXPECT_EQ(ThrowingCounted::live, 6);
    EXPECT_EQ(arena.size(), 14u);

    // Growth continues after the gap.
    EXPECT_EQ(arena.grow_by(5), 14u);
    EXPECT_EQ(ThrowingCounted::live, 11);
  }
  // Only constructed elements were destroyed: no destruction of the gap, nothing left over.
  EXPECT_EQ(ThrowingCounted::live, 0);
}

TEST(ConcurrentObjectArena, ThrowingConstructorAfterLaterGrowLeavesGap) {
  ThrowingCounted::live = 0;
  {
    dispenso::ConcurrentObjectArena<ThrowingCounted> arena(4, 3);
    ThrowingCounted::growDuringThrow = &arena;
    // The 2nd element of the next grow throws after a nested grow_by(2) has claimed the slots
    // behind it, so the gap is followed by live elements.
    ThrowingCounted::throwAt = 3 + 2;
    EXPECT_THROW(arena.grow_by(5), std::runtime_error);
    ThrowingCounted::growDuringThrow = nullptr;
    EXPECT_EQ(ThrowingCounted::live, 3 + 2);
    EXPECT_EQ(arena.size(), 3u + 5u + 2u);

    arena.grow_by(4);
    EXPECT_EQ(ThrowingCounted::live, 3 + 2 + 4);
  }
  // Only constructed elements were destroyed: no double destruction, nothing left over.
  EXPECT_EQ(ThrowingCounted::live, 0);
}
#endif // __cpp_exceptions
