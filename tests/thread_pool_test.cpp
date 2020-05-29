// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/thread_pool.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using testing::AnyOf;
using testing::Eq;

TEST(ThreadPool, SimpleCreationDestruction) {
  dispenso::ThreadPool pool(10);
  EXPECT_EQ(pool.numThreads(), 10);
}

TEST(ThreadPool, Resize) {
  dispenso::ThreadPool pool(10);
  EXPECT_EQ(10, pool.numThreads());
  pool.resize(8);
  EXPECT_EQ(8, pool.numThreads());
  pool.resize(12);
  EXPECT_EQ(12, pool.numThreads());
  pool.resize(12);
  EXPECT_EQ(12, pool.numThreads());
}

enum ScheduleType { kDefault, kForceQueue, kMixed };

class ThreadPoolTest : public testing::TestWithParam<ScheduleType> {
 protected:
  void initPool(size_t threads) {
    pool_ = std::make_unique<dispenso::ThreadPool>(threads);
  }

  template <typename F>
  void schedule(F&& f) {
    switch (GetParam()) {
      case kDefault:
        pool_->schedule(std::forward<F>(f));
        break;
      case kForceQueue:
        pool_->schedule(std::forward<F>(f), dispenso::ForceQueuingTag());
        break;
      case kMixed:
        if (count_++ & 1) {
          pool_->schedule(std::forward<F>(f));
        } else {
          pool_->schedule(std::forward<F>(f), dispenso::ForceQueuingTag());
        }
        break;
    }
  }

  void destroyPool() {
    pool_.reset();
  }

 private:
  std::unique_ptr<dispenso::ThreadPool> pool_;
  size_t count_ = 0;
};

INSTANTIATE_TEST_CASE_P(
    ThreadPoolTestParameters,
    ThreadPoolTest,
    testing::Values(kDefault, kForceQueue, kMixed));

TEST_P(ThreadPoolTest, SimpleWork) {
  constexpr int kWorkItems = 10000;
  std::vector<int> outputs(kWorkItems, 0);
  std::atomic<int> completed(0);
  {
    initPool(10);
    int i = 0;
    for (int& o : outputs) {
      schedule([i, &o, &completed]() {
        o = i * i;
        completed.fetch_add(1, std::memory_order_relaxed);
      });
      ++i;
    }
    destroyPool();
  }

  int i = 0;
  for (int o : outputs) {
    EXPECT_EQ(o, i * i);
    ++i;
  }
}

TEST_P(ThreadPoolTest, MixedWork) {
  constexpr int64_t kWorkItems = 10000;
  std::vector<int64_t> outputsA(kWorkItems, 0);
  std::vector<int64_t> outputsB(kWorkItems, 0);
  std::atomic<int> completed(0);
  {
    initPool(10);
    for (int64_t i = 0; i < kWorkItems; ++i) {
      auto& a = outputsA[i];
      auto& b = outputsB[i];
      schedule([i, &a, &completed]() {
        a = i * i;
        completed.fetch_add(1, std::memory_order_relaxed);
      });
      schedule([i, &b, &completed]() {
        b = i * i * i;
        completed.fetch_add(1, std::memory_order_relaxed);
      });
    }
    destroyPool();
  }

  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }
}

TEST(ThreadPool, ResizeConcurrent) {
  constexpr int kWorkItems = 10000;
  std::vector<int> outputs(kWorkItems, 0);
  std::atomic<int> completed(0);
  {
    dispenso::ThreadPool pool(10);
    int i = 0;
    for (int& o : outputs) {
      pool.schedule([i, &o, &completed]() {
        o = i * i;
        completed.fetch_add(1, std::memory_order_relaxed);
      });
      ++i;

      if ((i & 127) == 0) {
        if (pool.numThreads() == 1) {
          pool.resize(10);
        }
        pool.resize(pool.numThreads() - 1);
      }
    }
    EXPECT_EQ(4, pool.numThreads());
  }

  int i = 0;
  for (int o : outputs) {
    EXPECT_EQ(o, i * i);
    ++i;
  }
}

TEST(ThreadPool, ResizeMoreConcurrent) {
  constexpr int kWorkItems = 1000000;
  std::vector<int64_t> outputs(kWorkItems, 0);
  std::atomic<int> completed(0);
  {
    dispenso::ThreadPool pool(10);

    std::thread resizer0([&pool]() {
      for (int i = 0; i < 2000; ++i) {
        pool.resize(4);
      }
    });

    std::thread resizer1([&pool]() {
      for (int i = 0; i < 2000; ++i) {
        pool.resize(8);
      }
    });

    int64_t i = 0;
    for (int64_t& o : outputs) {
      pool.schedule([i, &o, &completed]() {
        o = i * i;
        completed.fetch_add(1, std::memory_order_relaxed);
      });
      ++i;
    }
    resizer0.join();
    resizer1.join();

    EXPECT_THAT(static_cast<int>(pool.numThreads()), AnyOf(Eq(4), Eq(8)));
  }

  int64_t i = 0;
  for (int64_t o : outputs) {
    EXPECT_EQ(o, i * i);
    ++i;
  }
}

TEST(ThreadPool, ResizeCheckApproxActualRunningThreads) {
  constexpr int kWorkItems = 1000000;
  std::vector<int64_t> outputs(kWorkItems, 0);
  std::atomic<int> completed(0);

  std::mutex mtx;
  std::set<std::thread::id> tidSet;

  dispenso::ThreadPool pool(1);

  pool.resize(8);

  int64_t i = 0;
  for (int64_t& o : outputs) {
    pool.schedule([i, &o, &completed, &mtx, &tidSet]() {
      o = i * i;
      completed.fetch_add(1, std::memory_order_release);
      std::lock_guard<std::mutex> lk(mtx);
      tidSet.insert(std::this_thread::get_id());
    });
    ++i;
  }

  while (completed.load(std::memory_order_relaxed) < kWorkItems) {
  }

  // We choose > 2 because there is 1 original thread, and one schedule thread (main thread). In
  // order to not have an occasional flake, we choose much lower than 8, though this test is
  // fundamentally flawed in that it cannot guarantee flake-free behavior.  Thus we turn this
  // particular check off when running in TSAN.
#if defined(__has_feature) 
#if !__has_feature(thread_sanitizer)
  EXPECT_GT(tidSet.size(), 2);
#endif
#endif // TSAN

  EXPECT_THAT(static_cast<int>(pool.numThreads()), AnyOf(Eq(4), Eq(8)));

  i = 0;
  for (int64_t o : outputs) {
    EXPECT_EQ(o, i * i) << " i = " << i;
    ++i;
  }
}
