/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/parallel_invoke.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

TEST(ParallelInvokeTest, TwoFunctors) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  int a = 0, b = 0;
  dispenso::parallel_invoke(tasks, [&a]() { a = 42; }, [&b]() { b = 99; });
  tasks.wait();
  EXPECT_EQ(a, 42);
  EXPECT_EQ(b, 99);
}

TEST(ParallelInvokeTest, ThreeFunctors) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  int a = 0, b = 0, c = 0;
  dispenso::parallel_invoke(tasks, [&a]() { a = 1; }, [&b]() { b = 2; }, [&c]() { c = 3; });
  tasks.wait();
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}

TEST(ParallelInvokeTest, SingleFunctor) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  int a = 0;
  dispenso::parallel_invoke(tasks, [&a]() { a = 7; });
  tasks.wait();
  EXPECT_EQ(a, 7);
}

TEST(ParallelInvokeTest, LastRunsInline) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  auto callerThread = std::this_thread::get_id();
  std::thread::id lastThread;
  dispenso::parallel_invoke(
      tasks, []() {}, [&lastThread]() { lastThread = std::this_thread::get_id(); });
  tasks.wait();
  EXPECT_EQ(lastThread, callerThread);
}

TEST(ParallelInvokeTest, ManyFunctors) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  std::atomic<int> sum(0);
  dispenso::parallel_invoke(
      tasks,
      [&sum]() { sum.fetch_add(1, std::memory_order_relaxed); },
      [&sum]() { sum.fetch_add(2, std::memory_order_relaxed); },
      [&sum]() { sum.fetch_add(4, std::memory_order_relaxed); },
      [&sum]() { sum.fetch_add(8, std::memory_order_relaxed); },
      [&sum]() { sum.fetch_add(16, std::memory_order_relaxed); });
  tasks.wait();
  EXPECT_EQ(sum.load(), 31);
}

TEST(ParallelInvokeTest, RecursiveDivideAndConquer) {
  std::vector<int> data(1000);
  std::iota(data.begin(), data.end(), 0);
  std::vector<int> sorted_copy = data;
  std::sort(sorted_copy.begin(), sorted_copy.end());

  std::shuffle(data.begin(), data.end(), std::mt19937(42));

  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  size_t mid = data.size() / 2;
  dispenso::parallel_invoke(
      tasks,
      [&data, mid]() { std::sort(data.begin(), data.begin() + mid); },
      [&data, mid]() { std::sort(data.begin() + mid, data.end()); });
  tasks.wait();
  std::inplace_merge(data.begin(), data.begin() + mid, data.end());
  EXPECT_EQ(data, sorted_copy);
}

TEST(ParallelInvokeTest, HeavyWorkload) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  // static so the lambdas below can use it without an explicit capture; MSVC
  // (unlike GCC/Clang) otherwise rejects the reference with C3493.
  static constexpr int kSize = 100000;
  std::vector<double> a(kSize), b(kSize);

  dispenso::parallel_invoke(
      tasks,
      [&a]() {
        for (int i = 0; i < kSize; ++i) {
          a[static_cast<size_t>(i)] = static_cast<double>(i) * 1.5;
        }
      },
      [&b]() {
        for (int i = 0; i < kSize; ++i) {
          b[static_cast<size_t>(i)] = static_cast<double>(i) * 2.5;
        }
      });
  tasks.wait();

  EXPECT_DOUBLE_EQ(a[0], 0.0);
  EXPECT_DOUBLE_EQ(a[kSize - 1], (kSize - 1) * 1.5);
  EXPECT_DOUBLE_EQ(b[0], 0.0);
  EXPECT_DOUBLE_EQ(b[kSize - 1], (kSize - 1) * 2.5);
}
