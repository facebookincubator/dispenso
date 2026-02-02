/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/concurrent_object_arena.h>
#include <dispenso/task_set.h>

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
