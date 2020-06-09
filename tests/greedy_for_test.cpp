// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <list>
#include <vector>

#include <dispenso/parallel_for.h>

#include <gtest/gtest.h>

void simpleInner(int w, int y, const std::vector<int>& image, std::atomic<int64_t>& sum) {
  const int* row = image.data() + y * w;
  int64_t s = 0;
  for (int i = 0; i < w; ++i) {
    s += row[i];
  }
  sum.fetch_add(s, std::memory_order_relaxed);
}

TEST(GreedyFor, SimpleLoop) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sum(0);

  dispenso::parallel_for(0, h, [w, &image, &sum](int y) { simpleInner(w, y, image, sum); });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
}

template <typename StateContainer>
void loopWithStateImpl() {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  StateContainer state;
  dispenso::parallel_for(
      state,
      []() { return int64_t{0}; },
      0,
      h,
      [w, &image](int64_t& sum, int y) {
        int* row = image.data() + y * w;
        int64_t s = 0;
        for (int i = 0; i < w; ++i) {
          s += row[i];
        }
        sum += s;
      });

  int64_t sum = 0;
  for (int64_t s : state) {
    sum += s;
  }

  EXPECT_EQ(sum, w * h * 7);
}

TEST(GreedyFor, LoopWithDequeState) {
  loopWithStateImpl<std::deque<int64_t>>();
}
TEST(GreedyFor, LoopWithVectorState) {
  loopWithStateImpl<std::vector<int64_t>>();
}
TEST(GreedyFor, LoopWithListState) {
  loopWithStateImpl<std::list<int64_t>>();
}

TEST(GreedyFor, ConcurrentLoopNoCoordination) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sumA(0);
  std::atomic<int64_t> sumB(0);

  std::thread tA([w, h, &image, &sumA]() {
    dispenso::parallel_for(0, h, [w, &image, &sumA](int y) { simpleInner(w, y, image, sumA); });
  });
  std::thread tB([w, h, &image, &sumB]() {
    dispenso::parallel_for(0, h, [w, &image, &sumB](int y) { simpleInner(w, y, image, sumB); });
  });

  tA.join();
  tB.join();

  EXPECT_EQ(sumA.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_EQ(sumB.load(std::memory_order_relaxed), w * h * 7);
}

TEST(GreedyFor, CoordinatedLoops) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sumA(0);
  std::atomic<int64_t> sumB(0);

  dispenso::TaskSet taskSet(dispenso::globalThreadPool());

  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumA](int y) { simpleInner(w, y, image, sumA); }, false);

  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumB](int y) { simpleInner(w, y, image, sumB); }, true);

  EXPECT_EQ(sumA.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_EQ(sumB.load(std::memory_order_relaxed), w * h * 7);
}

void concurrentLoop(
    dispenso::ConcurrentTaskSet& taskSet,
    int w,
    int h,
    const std::vector<int>& image,
    std::atomic<int64_t>& sum) {
  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sum](int y) { simpleInner(w, y, image, sum); }, false);
}

TEST(GreedyFor, CoordinatedConcurrentLoops) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sumA(0);
  std::atomic<int64_t> sumB(0);

  dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumA](int y) { simpleInner(w, y, image, sumA); }, false);

  std::thread thread(
      [&taskSet, w, h, &image, &sumB]() { concurrentLoop(taskSet, w, h, image, sumB); });
  thread.join();
  taskSet.wait();

  EXPECT_EQ(sumA.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_EQ(sumB.load(std::memory_order_relaxed), w * h * 7);
}
