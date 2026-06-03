/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/rw_lock.h>

#include <atomic>
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

// Regression: try_lock() (write) must fail while a reader is held, and must
// leave the lock in a clean, re-lockable state. The previous implementation set
// the writer bit and returned true even with active readers, handing out a
// non-exclusive "exclusive" lock.
TEST(RWLock, TryLockFailsWithActiveReader) {
  dispenso::RWLock mtx;

  // Hold a read lock on the main thread.
  mtx.lock_shared();

  // A write try_lock from another thread must not succeed while the reader is
  // held. Use a separate thread because reader->writer on one thread is UB.
  std::atomic<bool> gotWriteWhileRead{true};
  std::thread reader([&]() { gotWriteWhileRead.store(mtx.try_lock()); });
  reader.join();
  EXPECT_FALSE(gotWriteWhileRead.load());

  // Once the reader drains, the lock must be acquirable again -- proving the
  // failed try_lock rolled its writer bit back rather than leaking it.
  mtx.unlock_shared();

  std::atomic<bool> gotWriteAfter{false};
  std::thread writer([&]() {
    if (mtx.try_lock()) {
      gotWriteAfter.store(true);
      mtx.unlock();
    }
  });
  writer.join();
  EXPECT_TRUE(gotWriteAfter.load());
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
