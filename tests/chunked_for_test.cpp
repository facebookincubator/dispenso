/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <future>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include <dispenso/concurrent_vector.h>
#include <dispenso/parallel_for.h>
#include <dispenso/task_set.h>
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
      dispenso::makeChunkedRange(0, 0, dispenso::ParForChunking::kAdaptive),
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
      dispenso::makeChunkedRange(0, h, dispenso::ParForChunking::kAdaptive),
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
          dispenso::ParForChunking::kAdaptive),
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
          dispenso::ParForChunking::kAdaptive),
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
          dispenso::ParForChunking::kAdaptive),
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
          dispenso::ParForChunking::kAdaptive),
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
          dispenso::ParForChunking::kAdaptive),
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

static void minChunkSize(dispenso::ParForChunking choice, int start, int end, int minSize) {
  dispenso::ConcurrentVector<std::pair<int, int>> ranges;

  dispenso::ThreadPool pool(16);
  dispenso::TaskSet tasks(pool);

  dispenso::ParForOptions options;
  options.minItemsPerChunk = minSize;

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(start, end, choice),
      [&ranges](int ystart, int yend) { ranges.push_back({ystart, yend}); },
      options);

  EXPECT_GE(ranges.size(), 1);

  // For kStatic, minItemsPerChunk is a hard floor on chunk size (only the last
  // chunk may be smaller).
  // For kAdaptive, it is a soft hint: up to numWorkers "last per worker" chunks
  // may fall below minSize due to steal-induced remainders. We bound the count
  // by the worker count (pool size + caller = 17 in this test).
  if (choice == dispenso::ParForChunking::kStatic) {
    for (auto& r : ranges) {
      if (r.second != end) {
        EXPECT_LE(minSize, r.second - r.first);
      }
    }
  } else {
    constexpr size_t kMaxWorkers = 17; // pool(16) + caller
    size_t belowMin = 0;
    for (auto& r : ranges) {
      if ((r.second - r.first) < minSize) {
        ++belowMin;
      }
    }
    EXPECT_LE(belowMin, kMaxWorkers)
        << "Too many sub-minSize chunks for kAdaptive (worker count = " << kMaxWorkers << ")";
  }
}

TEST(ChunkedFor, MinChunkSizeLoopAuto) {
  minChunkSize(dispenso::ParForChunking::kAdaptive, 0, 1000000, 200);
  minChunkSize(dispenso::ParForChunking::kAdaptive, 0, 100, 200);
  minChunkSize(dispenso::ParForChunking::kAdaptive, 10000, 10020, 200);
  minChunkSize(dispenso::ParForChunking::kAdaptive, 1000000, 10000000, 20000);
  minChunkSize(dispenso::ParForChunking::kAdaptive, -10000000, -1000000, 20000);
}

TEST(ChunkedFor, MinChunkSizeLoopStatic) {
  minChunkSize(dispenso::ParForChunking::kStatic, 0, 1000000, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 0, 100, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 10000, 10020, 200);
  minChunkSize(dispenso::ParForChunking::kStatic, 1000000, 10000000, 20000);
  minChunkSize(dispenso::ParForChunking::kStatic, -10000000, -1000000, 20000);
}

template <typename StateContainer>
void loopWithStateImplReuseState() {
  int w = 1024;
  int h = 1024;
  std::vector<int> image(static_cast<size_t>(w * h), 7);

  StateContainer state;

  dispenso::ParForOptions options;
  options.reuseExistingState = true;

  for (size_t i = 0; i < 3; ++i) {
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
        },
        options);
  }

  int64_t sum = 0;
  for (int64_t s : state) {
    sum += s;
  }

  EXPECT_EQ(sum, 3 * w * h * 7);
}

TEST(ChunkedFor, LoopWithDequeStateReuse) {
  loopWithStateImplReuseState<std::deque<int64_t>>();
}
TEST(ChunkedFor, LoopWithVectorStateReuse) {
  loopWithStateImplReuseState<std::vector<int64_t>>();
}
TEST(ChunkedFor, LoopWithListStateReuse) {
  loopWithStateImplReuseState<std::list<int64_t>>();
}

// Regression test: nested TaskSet + parallel_for with ring scheduling.
// TaskSet::wait() must drain per-thread rings, not just the central queue.
// Without the fix, scheduleBulkToRings puts work into rings but wait()
// only checks the central queue — all workers block in wait() while ring
// tasks are stranded, causing deadlock.
TEST(ChunkedFor, NestedTaskSetParallelForRingDeadlock) {
  constexpr int kThreads = 8;
  constexpr int kInnerSize = kThreads;

  auto future = std::async(std::launch::async, [=]() {
    dispenso::ThreadPool pool(kThreads);
    dispenso::ConcurrentTaskSet outerSet(pool);

    std::atomic<int> completed{0};

    for (int t = 0; t < kThreads; ++t) {
      outerSet.schedule([&pool, &completed, kInnerSize]() {
        dispenso::ConcurrentTaskSet innerSet(pool);
        std::atomic<int> innerCount{0};

        dispenso::parallel_for(
            innerSet,
            dispenso::makeChunkedRange(0, kInnerSize, dispenso::ParForChunking::kStatic),
            [&innerCount](int i, int end) {
              for (; i < end; ++i) {
                innerCount.fetch_add(1, std::memory_order_relaxed);
              }
            });

        innerSet.wait();
        EXPECT_EQ(innerCount.load(), kInnerSize);
        completed.fetch_add(1, std::memory_order_relaxed);
      });
    }

    outerSet.wait();
    EXPECT_EQ(completed.load(), kThreads);
  });

  auto waitStatus = future.wait_for(std::chrono::seconds(10));
  EXPECT_EQ(waitStatus, std::future_status::ready)
      << "Deadlock: TaskSet::wait() does not drain per-thread rings";
}

// =============================================================================
// Granularity contract tests
// =============================================================================
//
// The contract: when ParForOptions::granularity > 1, every chunk passed to the
// user's lambda by parallel work has (end - begin) as a multiple of
// granularity, EXCEPT possibly the single "tail" invocation that processes the
// sub-granularity remainder of the range at the very end.

namespace {

// Run parallel_for with the given chunking/granularity and collect:
//  - all chunk extents in invocation order (best-effort: ordering is racy
//    across threads, but each chunk is recorded once).
//  - whether every element of [start, end) is covered exactly once.
struct GranularityResult {
  std::vector<std::pair<int, int>> chunks; // (begin, end) for each invocation
  std::vector<int> visits; // visit count per element
  int totalCovered = 0;
};

GranularityResult runGranularityFor(
    dispenso::ParForChunking chunking,
    int start,
    int end,
    uint32_t granularity,
    int numThreads = 8) {
  GranularityResult res;
  res.visits.assign(static_cast<size_t>(end - start), 0);

  std::mutex visitsMutex;

  dispenso::ThreadPool pool(static_cast<size_t>(numThreads));
  dispenso::TaskSet tasks(pool);

  dispenso::ParForOptions options;
  options.granularity = granularity;

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(start, end, chunking),
      [&res, &visitsMutex, start](int s, int e) {
        std::lock_guard<std::mutex> g(visitsMutex);
        res.chunks.emplace_back(s, e);
        for (int i = s; i < e; ++i) {
          ++res.visits[static_cast<size_t>(i - start)];
        }
      },
      options);

  for (int v : res.visits) {
    if (v >= 1) {
      ++res.totalCovered;
    }
  }
  return res;
}

// Verify the contract:
//  - Every element of [start, end) visited exactly once.
//  - At most one chunk with (end - begin) % granularity != 0 (the tail).
//  - If a sub-granularity chunk exists, it sits at the end of the range.
void verifyGranularityContract(
    const GranularityResult& res,
    int start,
    int end,
    uint32_t granularity) {
  // Every element covered exactly once.
  for (size_t i = 0; i < res.visits.size(); ++i) {
    ASSERT_EQ(res.visits[i], 1) << "Element " << (start + static_cast<int>(i)) << " visited "
                                << res.visits[i] << " times (expected 1)";
  }
  ASSERT_EQ(res.totalCovered, end - start);

  // Count tail chunks.
  int tailChunks = 0;
  int tailMaxEnd = std::numeric_limits<int>::min();
  for (auto& c : res.chunks) {
    int size = c.second - c.first;
    if (granularity > 1 && (size % static_cast<int>(granularity)) != 0) {
      ++tailChunks;
      tailMaxEnd = std::max(tailMaxEnd, c.second);
    }
  }
  EXPECT_LE(tailChunks, 1) << "Found " << tailChunks
                           << " sub-granularity chunks, expected at most 1 (the tail)";
  if (tailChunks == 1) {
    EXPECT_EQ(tailMaxEnd, end)
        << "Sub-granularity chunk found but it is not the trailing chunk of the range";
  }
}

void verifyGranularityContract(
    const std::vector<int>& visits,
    const std::vector<std::pair<int, int>>& chunks,
    int start,
    int end,
    uint32_t granularity) {
  GranularityResult res;
  res.visits = visits;
  res.chunks = chunks;
  res.totalCovered = end - start;
  verifyGranularityContract(res, start, end, granularity);
}

} // namespace

TEST(ChunkedFor, GranularityStaticExactMultiple) {
  // 1024 elements, granularity 8 → exact multiple, no tail.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 1024, 8);
  verifyGranularityContract(res, 0, 1024, 8);
}

TEST(ChunkedFor, GranularityStaticWithTail) {
  // 1000 elements, granularity 8 → 125 * 8 = 1000... actually exact. Use 1003.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 1003, 8);
  verifyGranularityContract(res, 0, 1003, 8);
}

TEST(ChunkedFor, GranularityStaticLargeWithTail) {
  // 999999 with granularity 16: 62499 * 16 = 999984, tail = 15.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 999999, 16);
  verifyGranularityContract(res, 0, 999999, 16);
}

TEST(ChunkedFor, GranularityAutoExactMultiple) {
  auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, 1024, 8);
  verifyGranularityContract(res, 0, 1024, 8);
}

TEST(ChunkedFor, GranularityAutoWithTail) {
  auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, 10003, 8);
  verifyGranularityContract(res, 0, 10003, 8);
}

TEST(ChunkedFor, GranularityAutoLargeWithTail) {
  // Larger range so the dynamic path actually exercises multiple chunks per
  // worker (size 1,000,003 with granularity 16 → tail = 3).
  auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, 1000003, 16);
  verifyGranularityContract(res, 0, 1000003, 16);
}

TEST(ChunkedFor, GranularityNonZeroStart) {
  // start=1000, end=1257, granularity=16: range size 257, tail 1 (257 % 16 = 1).
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 1000, 1257, 16);
  verifyGranularityContract(res, 1000, 1257, 16);
}

TEST(ChunkedFor, GranularityNegativeStart) {
  // Verify negative starts work — granularity rounding mustn't mishandle sign.
  auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, -1000, 17, 8);
  verifyGranularityContract(res, -1000, 17, 8);
}

TEST(ChunkedFor, GranularitySingleElementSubGranularity) {
  // Range smaller than granularity (5 elements, granularity 8): the whole
  // range is a single sub-granularity tail. Should execute as one inline call.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 5, 8);
  verifyGranularityContract(res, 0, 5, 8);
  // Should have exactly one chunk covering everything.
  EXPECT_EQ(res.chunks.size(), 1u);
}

TEST(ChunkedFor, GranularitySmallRangeAtGranularityBoundary) {
  // Range exactly equal to granularity. Should run as one full chunk, no tail.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 8, 8);
  verifyGranularityContract(res, 0, 8, 8);
}

TEST(ChunkedFor, GranularityIgnoredForExplicitChunkSize) {
  // When the user gives an explicit chunk size, granularity is ignored —
  // the user is already specifying chunk granularity.
  dispenso::ConcurrentVector<std::pair<int, int>> chunks;
  dispenso::ThreadPool pool(8);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.granularity = 100; // Should be ignored.
  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(0, 1000, 7), // Explicit chunk size of 7.
      [&chunks](int s, int e) { chunks.push_back({s, e}); },
      options);
  // With explicit chunk=7, we expect chunks of size 7 (except possibly the last).
  // No granularity enforcement.
  ASSERT_GE(chunks.size(), 1u);
  // Sum of all chunk sizes = 1000.
  int total = 0;
  for (auto& c : chunks) {
    total += (c.second - c.first);
  }
  EXPECT_EQ(total, 1000);
}

TEST(ChunkedFor, GranularityVariousValues) {
  // Sweep across granularity values 1, 2, 4, 8, 16, 32, 64.
  for (uint32_t g : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
    SCOPED_TRACE("granularity=" + std::to_string(g));
    int size = 12345; // Prime-ish; rarely a multiple of granularity.
    auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, size, g);
    verifyGranularityContract(res, 0, size, g);

    auto resAuto = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, size, g);
    verifyGranularityContract(resAuto, 0, size, g);
  }
}

// Stress test: ensure that with high granularity and many threads, we don't
// accidentally drop or double-count any elements.
TEST(ChunkedFor, GranularityStressManyThreads) {
  for (int run = 0; run < 8; ++run) {
    auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, 100007, 8, 64);
    verifyGranularityContract(res, 0, 100007, 8);
  }
}

// Verify the sub-granularity tail of an interior chunk doesn't sneak through.
// Specifically: if the parallel portion produces N+1 chunks where one is
// sub-granularity AND not the last, the contract is violated.
TEST(ChunkedFor, GranularityNoInteriorSubGranularityChunks) {
  // Use a range where the chunk math is awkward: 41 elements, granularity 8,
  // 4 threads. 41 / 8 = 5 granularity-units, tail of 1.
  auto res = runGranularityFor(dispenso::ParForChunking::kStatic, 0, 41, 8, 4);
  verifyGranularityContract(res, 0, 41, 8);
}

// No-wait dynamic path: the tail must still run, even though the caller
// doesn't wait inside parallel_for. The caller uses an external taskSet.wait().
TEST(ChunkedFor, GranularityNoWaitDynamicWithTail) {
  int start = 0;
  int end = 10003;
  uint32_t granularity = 8;

  std::vector<int> visits(static_cast<size_t>(end - start), 0);
  std::vector<std::pair<int, int>> chunks;
  std::mutex m;

  dispenso::ThreadPool pool(8);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.granularity = granularity;
  options.wait = false;

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(start, end, dispenso::ParForChunking::kAdaptive),
      [&visits, &chunks, &m, start](int s, int e) {
        std::lock_guard<std::mutex> g(m);
        chunks.emplace_back(s, e);
        for (int i = s; i < e; ++i) {
          ++visits[static_cast<size_t>(i - start)];
        }
      },
      options);
  tasks.wait();

  verifyGranularityContract(visits, chunks, start, end, granularity);
}

// Granularity must not break the StateContainer overload either: per-thread
// reduction must still aggregate the full range.
TEST(ChunkedFor, GranularityWithStateReducesCorrectly) {
  std::vector<int64_t> states;
  dispenso::ThreadPool pool(8);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.granularity = 16;

  constexpr int kSize = 100003; // size mod 16 = 3 (a tail exists)
  dispenso::parallel_for(
      tasks,
      states,
      []() { return int64_t{0}; },
      dispenso::makeChunkedRange(0, kSize, dispenso::ParForChunking::kAdaptive),
      [](int64_t& s, int begin, int end) {
        for (int i = begin; i < end; ++i) {
          s += i;
        }
      },
      options);

  int64_t total = 0;
  for (auto v : states) {
    total += v;
  }
  int64_t expected = (int64_t{kSize} * (kSize - 1)) / 2;
  EXPECT_EQ(total, expected);
}

// =============================================================================
// kAdaptive coverage tests
// =============================================================================
//
// These specifically target the per-worker steal-half implementation: each
// element must be visited exactly once even under heavy stealing contention.

namespace {

// Run kAdaptive parallel_for and verify exactly-once coverage. Returns the
// observed chunk extents (not necessarily in order).
struct AdaptiveCoverageResult {
  std::vector<std::pair<int, int>> chunks;
  std::vector<int> visits;
};

AdaptiveCoverageResult runAdaptiveCoverage(int start, int end, int numThreads) {
  AdaptiveCoverageResult res;
  res.visits.assign(static_cast<size_t>(end - start), 0);
  std::mutex m;

  dispenso::ThreadPool pool(static_cast<size_t>(numThreads));
  dispenso::TaskSet tasks(pool);

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(start, end, dispenso::ParForChunking::kAdaptive),
      [&res, &m, start](int s, int e) {
        std::lock_guard<std::mutex> g(m);
        res.chunks.emplace_back(s, e);
        for (int i = s; i < e; ++i) {
          ++res.visits[static_cast<size_t>(i - start)];
        }
      });
  return res;
}

void verifyAdaptiveCoverage(const AdaptiveCoverageResult& res, int start, int /*end*/) {
  for (size_t i = 0; i < res.visits.size(); ++i) {
    ASSERT_EQ(res.visits[i], 1) << "index " << (start + static_cast<int>(i)) << " visited "
                                << res.visits[i] << " times";
  }
}

} // namespace

TEST(ChunkedFor, AdaptiveExactlyOnceCoverage) {
  // Sweep various sizes and thread counts. Run each combination multiple
  // times to shake out steal-race bugs.
  for (int run = 0; run < 5; ++run) {
    for (int size : {17, 100, 1000, 10003, 100007}) {
      for (int nt : {1, 2, 4, 8, 16}) {
        SCOPED_TRACE(
            "size=" + std::to_string(size) + " nt=" + std::to_string(nt) +
            " run=" + std::to_string(run));
        auto res = runAdaptiveCoverage(0, size, nt);
        verifyAdaptiveCoverage(res, 0, size);
      }
    }
  }
}

TEST(ChunkedFor, AdaptiveCoverageWithLargeOffsets) {
  // Verify the adaptive impl handles non-zero starts (including very large
  // ones that exercise the int64 wide-arithmetic paths).
  for (int run = 0; run < 3; ++run) {
    auto res1 = runAdaptiveCoverage(1000000, 1000000 + 5003, 8);
    verifyAdaptiveCoverage(res1, 1000000, 1000000 + 5003);

    auto res2 = runAdaptiveCoverage(-50000, -50000 + 10007, 16);
    verifyAdaptiveCoverage(res2, -50000, -50000 + 10007);
  }
}

TEST(ChunkedFor, AdaptiveCoverageStressManyThreads) {
  // High thread count + medium size = lots of stealing. This is the test
  // most likely to expose owner-vs-stealer races.
  for (int run = 0; run < 20; ++run) {
    SCOPED_TRACE("run=" + std::to_string(run));
    auto res = runAdaptiveCoverage(0, 50003, 32);
    verifyAdaptiveCoverage(res, 0, 50003);
  }
}

TEST(ChunkedFor, AdaptiveCoverageSingleThread) {
  // Edge case: pool of 1 (only caller participates). No stealing should
  // happen, but the code path must still be correct.
  for (int run = 0; run < 3; ++run) {
    auto res = runAdaptiveCoverage(0, 10000, 1);
    verifyAdaptiveCoverage(res, 0, 10000);
  }
}

TEST(ChunkedFor, AdaptiveCoverageVeryFewItems) {
  // size < numWorkers: not every worker can have a non-empty range.
  // Ensures we handle workers whose initial range is empty.
  for (int size : {1, 2, 3, 5, 7}) {
    SCOPED_TRACE("size=" + std::to_string(size));
    auto res = runAdaptiveCoverage(0, size, 16);
    verifyAdaptiveCoverage(res, 0, size);
  }
}

// Adaptive + granularity contract: every chunk size must be a multiple of
// granularity, except possibly the single tail.
TEST(ChunkedFor, AdaptiveGranularityContract) {
  for (uint32_t g : {2u, 4u, 8u, 16u, 32u}) {
    SCOPED_TRACE("granularity=" + std::to_string(g));
    int size = 50003;
    auto res = runGranularityFor(dispenso::ParForChunking::kAdaptive, 0, size, g, 16);
    verifyGranularityContract(res, 0, size, g);
  }
}

// Adaptive should reduce to single-worker inline execution for tiny ranges
// (less than minItemsPerChunk).
TEST(ChunkedFor, AdaptiveTinyRangeInlinesCorrectly) {
  std::vector<std::pair<int, int>> chunks;
  std::mutex m;
  dispenso::ThreadPool pool(16);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.minItemsPerChunk = 1000;
  // Range of 5 is way smaller than minItemsPerChunk; expect single inline call.
  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(0, 5, dispenso::ParForChunking::kAdaptive),
      [&chunks, &m](int s, int e) {
        std::lock_guard<std::mutex> g(m);
        chunks.emplace_back(s, e);
      },
      options);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].first, 0);
  EXPECT_EQ(chunks[0].second, 5);
}

// Verify the kAuto alias maps to the same behavior as kAdaptive.
TEST(ChunkedFor, KAutoAliasesKAdaptive) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  static_assert(
      static_cast<int>(dispenso::ParForChunking::kAuto) ==
          static_cast<int>(dispenso::ParForChunking::kAdaptive),
      "kAuto must alias kAdaptive");
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
}

// Steal-split helper: ensures computeStealSplit honors the granularity floor.
TEST(ChunkedFor, AdaptiveStealsHonorGranularity) {
  // Pick a viewport where the work is wildly uneven: a fake bin-pack-style
  // workload where elements with i % 3 == 0 cost much more. With enough
  // imbalance, idle workers MUST steal from busy ones. Granularity contract
  // must hold under that stealing.
  std::vector<std::pair<int, int>> chunks;
  std::vector<int> visits(60003, 0);
  std::mutex m;

  dispenso::ThreadPool pool(16);
  dispenso::TaskSet tasks(pool);
  dispenso::ParForOptions options;
  options.granularity = 8;

  dispenso::parallel_for(
      tasks,
      dispenso::makeChunkedRange(0, 60003, dispenso::ParForChunking::kAdaptive),
      [&chunks, &visits, &m](int s, int e) {
        // Simulate work imbalance: long chunks take longer (gives idle
        // workers time to steal). Hold mutex only briefly to record.
        volatile int sink = 0;
        for (int i = s; i < e; ++i) {
          // Heavy work for indices divisible by 13
          int iters = (i % 13 == 0) ? 1000 : 10;
          for (int j = 0; j < iters; ++j) {
            sink = sink + j;
          }
        }
        std::lock_guard<std::mutex> g(m);
        chunks.emplace_back(s, e);
        for (int i = s; i < e; ++i) {
          ++visits[static_cast<size_t>(i)];
        }
        (void)sink;
      },
      options);

  verifyGranularityContract(visits, chunks, 0, 60003, 8);
}

// Nested parallel_for tests: outer kAdaptive that runs an inner parallel_for
// from each chunk. Verifies (1) correctness — every (outer, inner) pair
// executes exactly once — and (2) no deadlock. The current dispenso behavior
// for nested parallel_for is to detect recursion and run the inner inline
// on the calling worker (avoids oversubscription and deadlock), but the
// correctness contract is the same regardless of implementation strategy.

namespace {

// Helper: run an outer parallel_for over [0, outerN) of the given strategy,
// where each outer chunk runs an inner parallel_for over [0, innerN) of the
// given strategy that increments matrix[o][i]. Returns the resulting matrix.
template <typename OuterRange, typename InnerRange>
std::vector<std::atomic<int>>
nestedExecute(int outerN, int innerN, OuterRange outerR, InnerRange innerR) {
  std::vector<std::atomic<int>> matrix(static_cast<size_t>(outerN) * innerN);
  for (auto& a : matrix) {
    a.store(0, std::memory_order_relaxed);
  }
  dispenso::parallel_for(outerR, [&matrix, innerN, innerR](int o0, int o1) {
    for (int o = o0; o < o1; ++o) {
      dispenso::parallel_for(innerR, [&matrix, o, innerN](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
          matrix[static_cast<size_t>(o) * innerN + i].fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
  });
  return matrix;
}

void expectExactlyOnce(const std::vector<std::atomic<int>>& m) {
  for (size_t i = 0; i < m.size(); ++i) {
    int v = m[i].load(std::memory_order_relaxed);
    EXPECT_EQ(v, 1) << "matrix[" << i << "] visited " << v << " times (expected 1)";
  }
}

} // namespace

TEST(ChunkedFor, NestedAdaptiveAdaptive) {
  const int outerN = 64;
  const int innerN = 200;
  auto m = nestedExecute(
      outerN,
      innerN,
      dispenso::makeChunkedRange(0, outerN, dispenso::ParForChunking::kAdaptive),
      dispenso::makeChunkedRange(0, innerN, dispenso::ParForChunking::kAdaptive));
  expectExactlyOnce(m);
}

TEST(ChunkedFor, NestedStaticAdaptive) {
  const int outerN = 64;
  const int innerN = 200;
  auto m = nestedExecute(
      outerN,
      innerN,
      dispenso::makeChunkedRange(0, outerN, dispenso::ParForChunking::kStatic),
      dispenso::makeChunkedRange(0, innerN, dispenso::ParForChunking::kAdaptive));
  expectExactlyOnce(m);
}

TEST(ChunkedFor, NestedAdaptiveStatic) {
  const int outerN = 64;
  const int innerN = 200;
  auto m = nestedExecute(
      outerN,
      innerN,
      dispenso::makeChunkedRange(0, outerN, dispenso::ParForChunking::kAdaptive),
      dispenso::makeChunkedRange(0, innerN, dispenso::ParForChunking::kStatic));
  expectExactlyOnce(m);
}

// Stress version with deeper outer × inner and uneven inner sizes to
// exercise the stripe-claim path under recursion.
TEST(ChunkedFor, NestedAdaptiveAdaptiveStress) {
  const int outerN = 256;
  const int innerN = 1024;
  auto m = nestedExecute(
      outerN,
      innerN,
      dispenso::makeChunkedRange(0, outerN, dispenso::ParForChunking::kAdaptive),
      dispenso::makeChunkedRange(0, innerN, dispenso::ParForChunking::kAdaptive));
  expectExactlyOnce(m);
}

// Three-level deep nesting: outer kAdaptive → middle kAdaptive → inner
// kAdaptive. Verifies the recursion handling doesn't break at depth > 2.
TEST(ChunkedFor, ThreeLevelNestedAdaptive) {
  const int a = 8;
  const int b = 16;
  const int c = 32;
  std::vector<std::atomic<int>> matrix(static_cast<size_t>(a) * b * c);
  for (auto& v : matrix) {
    v.store(0, std::memory_order_relaxed);
  }
  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, a, dispenso::ParForChunking::kAdaptive),
      [&matrix, b, c](int a0, int a1) {
        for (int ai = a0; ai < a1; ++ai) {
          dispenso::parallel_for(
              dispenso::makeChunkedRange(0, b, dispenso::ParForChunking::kAdaptive),
              [&matrix, ai, b, c](int b0, int b1) {
                for (int bi = b0; bi < b1; ++bi) {
                  dispenso::parallel_for(
                      dispenso::makeChunkedRange(0, c, dispenso::ParForChunking::kAdaptive),
                      [&matrix, ai, bi, b, c](int c0, int c1) {
                        (void)b;
                        (void)c;
                        for (int ci = c0; ci < c1; ++ci) {
                          matrix
                              [(static_cast<size_t>(ai) * b + static_cast<size_t>(bi)) * c +
                               static_cast<size_t>(ci)]
                                  .fetch_add(1, std::memory_order_relaxed);
                        }
                      });
                }
              });
        }
      });
  expectExactlyOnce(matrix);
}
