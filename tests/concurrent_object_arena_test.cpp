
#include <dispenso/concurrent_object_arena.h>
#include <dispenso/task_set.h>

#include <gtest/gtest.h>

TEST(ConcurrentObjectArena, ParallelGrowBy) {
  dispenso::ConcurrentObjectArena<size_t> arena(16);

  constexpr size_t numTasks = 20;
  constexpr size_t numLoops = 100;
  constexpr size_t delta = 7;

  dispenso::TaskSet taskSet(dispenso::globalThreadPool());

  for (size_t ti = 0; ti < numTasks; ++ti) {
    taskSet.schedule([&arena, ti]() {
      for (size_t i = 0; i < numLoops; i++) {
        const size_t p = arena.grow_by(delta);
        for (size_t j = 0; j < delta; j++)
          arena[p + j] = ti * numLoops * delta + i;
      }
    });
  }
  taskSet.wait();

  EXPECT_EQ(delta * numLoops * numTasks, arena.size());

  for (size_t i = 0; i < numLoops * numTasks; i++) {
    const size_t firstElement = arena[i * delta];
    for (size_t j = 1; j < delta; j++)
      EXPECT_EQ(arena[i * delta + j], firstElement);
  }
}
