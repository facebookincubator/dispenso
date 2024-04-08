/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  options.defaultChunking = dispenso::ParForChunking::kAuto;

  dispenso::parallel_for(
      0, 0, [myNullPtr](int i) { *myNullPtr = i; }, options);

  options.defaultChunking = dispenso::ParForChunking::kStatic;

  dispenso::parallel_for(
      0, 0, [myNullPtr](int i) { *myNullPtr = i; }, options);
}

template <typename StateContainer>
void loopWithStateImpl(dispenso::ThreadPool& pool = dispenso::globalThreadPool()) {
  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  dispenso::TaskSet taskSet(pool);
  StateContainer state;
  dispenso::parallel_for(
      taskSet,
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

void testMaxThreads(
    size_t poolSize,
    uint32_t maxThreads,
    bool testStaticChunking,
    bool testWaitOption) {
  size_t numAvailableThreads = poolSize + testWaitOption;
  std::vector<int> threadLocalSums(numAvailableThreads, 0);
  dispenso::ThreadPool pool(poolSize);
  dispenso::TaskSet tasks(pool);

  dispenso::ParForOptions options;
  options.maxThreads = maxThreads;
  options.wait = testWaitOption;
  options.defaultChunking =
      testStaticChunking ? dispenso::ParForChunking::kStatic : dispenso::ParForChunking::kAuto;

  auto func = [&threadLocalSums](int index) {
    assert(index > 0); // for correctness of numNonZero
    std::this_thread::yield();
    threadLocalSums[getTestTid()] += index;
  };

  dispenso::parallel_for(tasks, 1, 10000, func, options);

  if (!testWaitOption) {
    // We didn't tell the parallel_for to wait, so we need to do it here to ensure the loop is
    // complete.
    tasks.wait();
  }

  int total = 0;
  int numNonZero = 0;

  for (size_t i = 0; i < numAvailableThreads; ++i) {
    numNonZero += threadLocalSums[i] > 0;
    total += threadLocalSums[i];
  }

  // 0 indicates serial execution per API spec
  size_t translatedMaxThreads = maxThreads == 0 ? 1 : maxThreads;
  EXPECT_LE(numNonZero, std::min((size_t)translatedMaxThreads, numAvailableThreads));
  EXPECT_EQ(total, 49995000);
}

TEST(GreedyFor, OptionsMaxThreadsBigPoolStaticChunkingBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = true;
  testMaxThreads(8, 4, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsBigPoolStaticChunkingNonBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = false;
  testMaxThreads(8, 4, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsBigPoolAutoChunkingBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = true;
  testMaxThreads(8, 4, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsBigPoolAutoChunkingNonBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = false;
  testMaxThreads(8, 4, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSmallPoolStaticChunkingBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = true;
  testMaxThreads(4, 8, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSmallPoolStaticChunkingNonBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = false;
  testMaxThreads(4, 8, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSmallPoolAutoChunkingBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = true;
  testMaxThreads(4, 8, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSmallPoolAutoChunkingNonBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = false;
  testMaxThreads(4, 8, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSerialStaticChunkingBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = true;
  testMaxThreads(8, 0, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSerialStaticChunkingNonBlocking) {
  constexpr bool staticChunking = true;
  constexpr bool waitOption = false;
  testMaxThreads(8, 0, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSerialAutoChunkingBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = true;
  testMaxThreads(8, 0, staticChunking, waitOption);
}

TEST(GreedyFor, OptionsMaxThreadsSerialAutoChunkingNonBlocking) {
  constexpr bool staticChunking = false;
  constexpr bool waitOption = false;
  testMaxThreads(8, 0, staticChunking, waitOption);
}

TEST(GreedyFor, NegativeRangeLength) {
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::parallel_for(taskSet, 2, -2, [](auto /*index*/) {
    EXPECT_FALSE(true) << "Shouldn't enter this function at all";
  });
}

TEST(GreedyFor, NegativeRangeLengthBig) {
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::parallel_for(taskSet, 2147483647, -2147483647, [](auto /*index*/) {
    EXPECT_FALSE(true) << "Shouldn't enter this function at all";
  });
}

TEST(GreedyFor, ZeroLength2) {
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::parallel_for(taskSet, -77, -77, [](auto /*index*/) {
    EXPECT_FALSE(true) << "Shouldn't enter this function at all";
  });
}

TEST(GreedyFor, AvoidOverflow1) {
  std::atomic<uint32_t> count(0);
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::parallel_for(
      taskSet,
      std::numeric_limits<int16_t>::min(),
      std::numeric_limits<int16_t>::max(),
      [&count](auto /*index*/) { count.fetch_add(1, std::memory_order_relaxed); });

  EXPECT_EQ(count.load(), std::numeric_limits<uint16_t>::max());
}

TEST(GreedyFor, AvoidOverflow2) {
  dispenso::ThreadPool pool(8);
  dispenso::TaskSet taskSet(pool);
  dispenso::ParForOptions options;
  options.wait = false;
  options.defaultChunking = dispenso::ParForChunking::kAuto;

  std::vector<dispenso::CacheAligned<uint32_t>> vals;
  dispenso::parallel_for(
      taskSet,
      vals,
      []() { return uint32_t{0}; },
      std::numeric_limits<int32_t>::min() / 2 - 1,
      std::numeric_limits<int32_t>::max() / 2 + 1,
      [](auto& count, auto /*index*/) { ++count; },
      options);
  taskSet.wait();

  uint32_t count = 0;
  for (auto& v : vals) {
    count += v;
  }

  EXPECT_EQ(count, std::numeric_limits<uint32_t>::max() / 2 + 2);
}

TEST(GreedyFor, EmptyLoopsWaitIfToldTo) {
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::ParForOptions options;
  options.wait = false;

  std::atomic<int> count(0);

  dispenso::parallel_for(
      taskSet,
      0,
      1000,
      [&count](int /*index*/) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        count.fetch_add(1, std::memory_order_acq_rel);
      },
      options);

  dispenso::ParForOptions waitOptions;
  dispenso::parallel_for(
      taskSet,
      0,
      0,
      [](int /*index*/) { EXPECT_FALSE(true) << "Should not reach this lambda"; },
      waitOptions);

  EXPECT_EQ(count.load(), 1000);
}

TEST(GreedyFor, SingleLoopWaitIfToldTo) {
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::ParForOptions options;
  options.wait = false;

  std::atomic<int> count(0);

  dispenso::parallel_for(
      taskSet,
      0,
      1000,
      [&count](int /*index*/) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        count.fetch_add(1, std::memory_order_acq_rel);
      },
      options);

  dispenso::ParForOptions waitOptions;
  dispenso::parallel_for(
      taskSet, 0, 1, [](int i) { EXPECT_EQ(i, 0); }, waitOptions);

  EXPECT_EQ(count.load(), 1000);
}

TEST(GreedyFor, ZeroThreads) {
  // Using a threadpool with 0 threads should run via the calling thread.
  dispenso::ThreadPool pool(0);
  dispenso::TaskSet tasks(pool);

  int w = 1000;
  int h = 1000;
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sum(0);

  dispenso::parallel_for(tasks, 0, h, [w, &image, &sum](int y) { simpleInner(w, y, image, sum); });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
}

TEST(GreedyFor, ZeroThreadsWithState) {
  // Using a threadpool with 0 threads should run via the calling thread.
  dispenso::ThreadPool pool(0);
  loopWithStateImpl<std::vector<int64_t>>(pool);
}

TEST(GreedyFor, SimpleLoopFewerItemsThanThreads) {
  int w = 1000;
  int h = 3;
  dispenso::ThreadPool pool(5);
  std::vector<int> image(w * h, 7);

  std::atomic<int64_t> sum(0);

  dispenso::TaskSet tasks(pool);
  dispenso::parallel_for(tasks, 0, h, [w, &image, &sum](int y) { simpleInner(w, y, image, sum); });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
}
