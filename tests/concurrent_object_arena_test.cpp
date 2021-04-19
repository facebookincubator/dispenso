
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
        for (size_t j = 0; j < delta; j++)
          arena[p + j] = ti * numLoops * delta + i;
      }
    });
  }
  taskSet.wait();

  EXPECT_EQ(delta * numLoops * numTasks, arena.size());
  EXPECT_EQ(arena.capacity() / arena.numBuffers(), arena.getBufferSize(0));

  size_t totalSize = 0;
  for (size_t i = 0; i < arena.numBuffers(); ++i)
    totalSize += arena.getBufferSize(i);
  EXPECT_EQ(totalSize, arena.size());

  for (size_t i = 0; i < numLoops * numTasks; i++) {
    const size_t firstElement = arena[i * delta];
    for (size_t j = 1; j < delta; j++)
      EXPECT_EQ(arena[i * delta + j], firstElement);
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

  dispenso::ConcurrentObjectArena<TestData> arena(bufSize);

  arena.grow_by(smallGrow);
  arena.grow_by(bigGrow);

  const size_t num = arena.size();
  for (size_t i = 0; i < num; ++i)
    EXPECT_EQ(arena[i].value, defaultValue);
}
