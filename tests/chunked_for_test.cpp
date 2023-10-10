/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <list>
#include <vector>

#include <dispenso/concurrent_vector.h>
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

TEST(ChunkedFor, SimpleLoopSmallRangeAtLargeValues) {
  std::atomic<uint64_t> numCalls(0);

  dispenso::ThreadPool pool(6);
  dispenso::TaskSet tasks(pool);

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(
          std::numeric_limits<uint64_t>::max() / 2 - 100,
          std::numeric_limits<uint64_t>::max() / 2 + 1000,
          dispenso::ParForChunking::kAuto),
      [&numCalls](auto ystart, auto yend) {
        numCalls.fetch_add(yend - ystart, std::memory_order_relaxed);
      });

  EXPECT_EQ(numCalls.load(std::memory_order_relaxed), 1100);
}

TEST(ChunkedFor, SimpleLoopSmallRange) {
  std::atomic<int> numCalls(0);

  dispenso::ThreadPool pool(6);
  dispenso::TaskSet tasks(pool);

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(
          std::numeric_limits<int16_t>::min(),
          std::numeric_limits<int16_t>::max(),
          dispenso::ParForChunking::kAuto),
      [&numCalls](auto ystart, auto yend) {
        numCalls.fetch_add(yend - ystart, std::memory_order_relaxed);
      });

  EXPECT_EQ(numCalls.load(std::memory_order_relaxed), (1 << 16) - 1);
}

TEST(ChunkedFor, LoopSmallRangeWithState) {
  std::atomic<int> numCalls(0);

  dispenso::ThreadPool pool(6);
  dispenso::TaskSet tasks(pool);

  std::vector<int> state;

  dispenso::parallel_for(
      tasks,
      state,
      []() { return 0; },
      dispenso::makeChunkedRange(
          std::numeric_limits<int16_t>::min(),
          std::numeric_limits<int16_t>::max(),
          dispenso::ParForChunking::kAuto),
      [&numCalls](auto& s, auto ystart, auto yend) {
        numCalls.fetch_add(yend - ystart, std::memory_order_relaxed);
        s += (yend - ystart);
      });

  EXPECT_EQ(numCalls.load(std::memory_order_relaxed), (1 << 16) - 1);
  int total = 0;
  for (int s : state) {
    total += s;
  }
  EXPECT_EQ(total, (1 << 16) - 1);
}

TEST(ChunkedFor, SimpleLoopSmallRangeExternalWait) {
  std::atomic<int> numCalls(0);

  dispenso::ThreadPool pool(6);
  dispenso::TaskSet tasks(pool);

  dispenso::ParForOptions options;
  options.wait = false;

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(
          std::numeric_limits<int16_t>::min(),
          std::numeric_limits<int16_t>::max(),
          dispenso::ParForChunking::kAuto),
      [&numCalls](auto ystart, auto yend) {
        numCalls.fetch_add(yend - ystart, std::memory_order_relaxed);
      },
      options);
  tasks.wait();

  EXPECT_EQ(numCalls.load(std::memory_order_relaxed), (1 << 16) - 1);
}

TEST(ChunkedFor, LoopSmallRangeWithStateWithExternalWait) {
  std::atomic<int> numCalls(0);

  dispenso::ThreadPool pool(6);
  dispenso::TaskSet tasks(pool);

  std::vector<int> state;
  dispenso::ParForOptions options;
  options.wait = false;

  dispenso::parallel_for(
      tasks,
      state,
      []() { return 0; },
      dispenso::makeChunkedRange(
          std::numeric_limits<int16_t>::min(),
          std::numeric_limits<int16_t>::max(),
          dispenso::ParForChunking::kAuto),
      [&numCalls](auto& s, auto ystart, auto yend) {
        numCalls.fetch_add(yend - ystart, std::memory_order_relaxed);
        s += (yend - ystart);
      },
      options);

  tasks.wait();

  EXPECT_EQ(numCalls.load(std::memory_order_relaxed), (1 << 16) - 1);
  int total = 0;
  for (int s : state) {
    total += s;
  }
  EXPECT_EQ(total, (1 << 16) - 1);
}

void minChunkSize(dispenso::ParForChunking choice, int start, int end, int minSize) {
  dispenso::ConcurrentVector<std::pair<int, int>> ranges;

  dispenso::ThreadPool pool(16);

  dispenso::ParForOptions options;
  options.minItemsPerChunk = minSize;

  dispenso::parallel_for(
      dispenso::makeChunkedRange(start, end, choice),
      [&ranges](int ystart, int yend) {
        ranges.push_back({ystart, yend});
      },
      options);

  EXPECT_GE(ranges.size(), 1);

  for (auto& r : ranges) {
    if (r.second != end) {
      EXPECT_LE(minSize, r.second - r.first);
    }
  }
}

TEST(ChunkedFor, MinChunkSizeLoopAuto) {
  minChunkSize(dispenso::ParForChunking::kAuto, 0, 1000000, 200);
  minChunkSize(dispenso::ParForChunking::kAuto, 0, 100, 200);
  minChunkSize(dispenso::ParForChunking::kAuto, 10000, 10020, 200);
  minChunkSize(dispenso::ParForChunking::kAuto, 1000000, 10000000, 20000);
  minChunkSize(dispenso::ParForChunking::kAuto, -10000000, -1000000, 20000);
}

TEST(ChunkedFor, MinChunkSizeLoopStatic) {
  minChunkSize(dispenso::ParForChunking::kStatic, 0, 1000000, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 0, 100, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 10000, 10020, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 1000000, 10000000, 20000);
  minChunkSize(dispenso::ParForChunking::kStatic, -10000000, -1000000, 20000);
}
