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

TEST(ChunkedFor, SimpleLoop) {
  int w = 1024;
  int h = 1024;
  std::vector<int> image(static_cast<size_t>(w * h), 7);

  std::atomic<int64_t> sum(0);

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, h, 8), [w, &image, &sum](int ystart, int yend) {
        EXPECT_EQ(yend - ystart, 8);
        int64_t s = 0;
        for (int y = ystart; y < yend; ++y) {
          int* row = image.data() + y * w;
          for (int i = 0; i < w; ++i) {
            s += row[i];
          }
        }
        sum.fetch_add(s, std::memory_order_relaxed);
      });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
}

TEST(ChunkedFor, ShouldNotInvokeIfEmptyRange) {
  int* myNullPtr = nullptr;

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, 0, dispenso::ParForChunking::kAuto),
      [myNullPtr](int s, int e) { *myNullPtr = s + e; });

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, 0, dispenso::ParForChunking::kStatic),
      [myNullPtr](int s, int e) { *myNullPtr = s + e; });
}

TEST(ChunkedFor, SimpleLoopStatic) {
  int w = 1024;
  int h = 1024;
  std::vector<int> image(static_cast<size_t>(w * h), 7);

  std::atomic<int64_t> sum(0);
  std::atomic<int> numCalls(0);

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, h, dispenso::ParForChunking::kStatic),
      [w, &image, &sum, &numCalls](int ystart, int yend) {
        numCalls.fetch_add(1, std::memory_order_relaxed);
        int64_t s = 0;
        for (int y = ystart; y < yend; ++y) {
          int* row = image.data() + y * w;
          for (int i = 0; i < w; ++i) {
            s += row[i];
          }
        }
        sum.fetch_add(s, std::memory_order_relaxed);
      });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_LE(
      numCalls.load(std::memory_order_relaxed),
      static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(ChunkedFor, SimpleLoopAuto) {
  int w = 1024;
  int h = 1024;
  std::vector<int> image(static_cast<size_t>(w * h), 7);

  std::atomic<int64_t> sum(0);
  std::atomic<int> numCalls(0);
  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, h, dispenso::ParForChunking::kAuto),
      [w, &image, &sum, &numCalls](int ystart, int yend) {
        numCalls.fetch_add(1, std::memory_order_relaxed);
        int64_t s = 0;
        for (int y = ystart; y < yend; ++y) {
          int* row = image.data() + y * w;
          for (int i = 0; i < w; ++i) {
            s += row[i];
          }
        }
        sum.fetch_add(s, std::memory_order_relaxed);
      });

  EXPECT_EQ(sum.load(std::memory_order_relaxed), w * h * 7);
  EXPECT_GT(
      numCalls.load(std::memory_order_relaxed),
      static_cast<int>(std::thread::hardware_concurrency()));
  EXPECT_LE(numCalls.load(std::memory_order_relaxed), 1024);
}

template <typename StateContainer>
void loopWithStateImpl() {
  int w = 1024;
  int h = 1024;
  std::vector<int> image(static_cast<size_t>(w * h), 7);

  StateContainer state;
  dispenso::parallel_for(
      state,
      []() { return int64_t{0}; },
      dispenso::makeChunkedRange(0, h, 16),
      [w, &image](int64_t& sum, int ystart, int yend) {
        EXPECT_EQ(yend - ystart, 16);
        int64_t s = 0;
        for (int y = ystart; y < yend; ++y) {
          int* row = image.data() + y * w;
          for (int i = 0; i < w; ++i) {
            s += row[i];
          }
        }
        sum += s;
      });

  int64_t sum = 0;
  for (int64_t s : state) {
    sum += s;
  }

  EXPECT_EQ(sum, w * h * 7);
}

TEST(ChunkedFor, LoopWithDequeState) {
  loopWithStateImpl<std::deque<int64_t>>();
}
TEST(ChunkedFor, LoopWithVectorState) {
  loopWithStateImpl<std::vector<int64_t>>();
}
TEST(ChunkedFor, LoopWithListState) {
  loopWithStateImpl<std::list<int64_t>>();
}
