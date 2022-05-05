/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/rw_lock.h>

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include <gtest/gtest.h>

using namespace std::chrono_literals;

TEST(RWLock, SimpleUncontested) {
  dispenso::RWLock mtx;
  int foo = 0;
  std::unique_lock<dispenso::RWLock> lk(mtx);
  foo = 1;

  lk.unlock();

  EXPECT_EQ(foo, 1);
}

TEST(RWLock, BasicWriterTest) {
  int count = 0;
  dispenso::RWLock mtx;
  constexpr int kPerThreadTotal = 100000;

  auto toRun = [&]() {
    for (int i = 0; i < kPerThreadTotal; ++i) {
      std::unique_lock<dispenso::RWLock> lk(mtx);
      ++count;
    }
  };

  std::thread thread0(toRun);
  std::thread thread1(toRun);

  thread0.join();
  thread1.join();

  EXPECT_EQ(count, 2 * kPerThreadTotal);
}

TEST(RWLock, HighContentionReaderWriterTest) {
  int count = 0;
  dispenso::RWLock mtx;
  constexpr int kPerThreadTotal = 100000;

  auto toRunWriter = [&]() {
    for (int i = 0; i < kPerThreadTotal; ++i) {
      std::unique_lock<dispenso::RWLock> lk(mtx);
      ++count;
    }
  };

  int64_t someVal = 0;

  auto toRunReader = [&]() {
    for (int i = 0; i < kPerThreadTotal; ++i) {
      std::shared_lock<dispenso::RWLock> lk(mtx);
      someVal += count;
    }
  };

  std::thread thread0(toRunWriter);
  std::thread thread1(toRunReader);

  thread0.join();
  thread1.join();

  EXPECT_EQ(count, kPerThreadTotal);
  EXPECT_GE(someVal, 0);
}

TEST(RWLock, ReaderWriterTest) {
  int guardedCount = 0;
  dispenso::RWLock mtx;
  constexpr int kWriterTotal = 100;
  constexpr int kReaderTotal = 100000;

  auto toRunWriter = [&]() {
    for (int i = 0; i < kWriterTotal; ++i) {
      std::unique_lock<dispenso::RWLock> lk(mtx);
      ++guardedCount;
      lk.unlock();
      // Just hang out for a while til we write again.
      std::this_thread::sleep_for(1ms);
    }
  };

  int64_t sum = 0;

  auto toRunReader = [&]() {
    for (int i = 0; i < kReaderTotal; ++i) {
      std::shared_lock<dispenso::RWLock> lk(mtx);
      sum += guardedCount;
    }
  };

  std::thread thread0(toRunWriter);
  std::thread thread1(toRunReader);

  thread0.join();
  thread1.join();

  EXPECT_EQ(guardedCount, kWriterTotal);
  EXPECT_GE(sum, 0);
}

TEST(RWLock, TestAlignment) {
  static_assert(
      alignof(dispenso::RWLock) >= dispenso::kCacheLineSize,
      "Somehow RWLock not aligned to avoid false sharing");
  static_assert(
      alignof(dispenso::UnalignedRWLock) < dispenso::kCacheLineSize,
      "UnalignedRWLock is overaligned");
}
