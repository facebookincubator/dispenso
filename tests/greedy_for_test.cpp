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

TEST(GreedyFor, ShouldNotInvokeIfEmptyRange) {
  int* myNullPtr = nullptr;

  dispenso::ParForOptions options;
  options.defaultChunking = dispenso::ParForOptions::kAuto;

  dispenso::parallel_for(
      0, 0, [myNullPtr](int i) { *myNullPtr = i; }, options);

  options.defaultChunking = dispenso::ParForOptions::kStatic;

  dispenso::parallel_for(
      0, 0, [myNullPtr](int i) { *myNullPtr = i; }, options);
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

  dispenso::ParForOptions options;
  options.wait = false;

  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumA](int y) { simpleInner(w, y, image, sumA); }, options);

  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumB](int y) { simpleInner(w, y, image, sumB); });

  EXPECT_EQ(sumA.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_EQ(sumB.load(std::memory_order_relaxed), w * h * 7);
}

void concurrentLoop(
    dispenso::ConcurrentTaskSet& taskSet,
    int w,
    int h,
    const std::vector<int>& image,
    std::atomic<int64_t>& sum) {
  dispenso::ParForOptions options;
  options.wait = false;
  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sum](int y) { simpleInner(w, y, image, sum); }, options);
}

TEST(GreedyFor, CoordinatedConcurrentLoops) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sumA(0);
  std::atomic<int64_t> sumB(0);

  dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

  dispenso::ParForOptions options;
  options.wait = false;
  dispenso::parallel_for(
      taskSet, 0, h, [w, &image, &sumA](int y) { simpleInner(w, y, image, sumA); }, options);

  std::thread thread(
      [&taskSet, w, h, &image, &sumB]() { concurrentLoop(taskSet, w, h, image, sumB); });
  thread.join();
  taskSet.wait();

  EXPECT_EQ(sumA.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_EQ(sumB.load(std::memory_order_relaxed), w * h * 7);
}

inline int getTestTid() {
  static DISPENSO_THREAD_LOCAL int tid = -1;
  static std::atomic<int> nextTid(0);
  if (tid < 0) {
    tid = nextTid.fetch_add(1, std::memory_order_relaxed);
  }
  return tid;
}

TEST(GreedyFor, OptionsMaxThreads) {
  size_t usedTids[9] = {0};

  dispenso::ThreadPool pool(8);
  dispenso::TaskSet tasks(pool);

  dispenso::ParForOptions options;
  options.maxThreads = 4;
  options.wait = false;

  auto func = [&usedTids](size_t index) {
    std::this_thread::yield();
    usedTids[getTestTid()] += index;
  };

  dispenso::parallel_for(tasks, 0, 10000, func, options);

  // We didn't tell the parallel_for to wait, so we need to do it here to ensure the loop is
  // complete.
  tasks.wait();
  int total = 0;
  int numNonZero = 0;

  for (int i = 0; i < 9; ++i) {
    numNonZero += usedTids[i] > 0;
    total += usedTids[i];
  }

  EXPECT_LE(numNonZero, 4);
  EXPECT_EQ(total, 49995000);

  memset(usedTids, 0, sizeof(usedTids));
  options.wait = true;

  dispenso::parallel_for(tasks, 0, 10000, func, options);
  total = 0;
  numNonZero = 0;

  for (int i = 0; i < 9; ++i) {
    numNonZero += usedTids[i] > 0;
    total += usedTids[i];
  }

  // We can now expect the calling thread to also participate.
  EXPECT_LE(numNonZero, 5);
  EXPECT_EQ(total, 49995000);
}
