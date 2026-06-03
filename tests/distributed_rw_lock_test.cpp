/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/distributed_rw_lock.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// DistributedRWLock (public wrapper with cached TLS slot)
// ---------------------------------------------------------------------------

TEST(DistributedRWLock, UncontestedReadWrite) {
  dispenso::DistributedRWLock<> mtx;
  int value = 0;

  {
    std::unique_lock<dispenso::DistributedRWLock<>> lk(mtx);
    value = 42;
  }

  {
    std::shared_lock<dispenso::DistributedRWLock<>> lk(mtx);
    EXPECT_EQ(value, 42);
  }
}

TEST(DistributedRWLock, MultipleReaders) {
  dispenso::DistributedRWLock<> mtx;
  const int sharedData = 123;
  std::atomic<int> readCount{0};
  constexpr int kNumThreads = 8;
  constexpr int kReadsPerThread = 10000;

  std::vector<std::thread> readers;
  for (int t = 0; t < kNumThreads; ++t) {
    readers.emplace_back([&]() {
      for (int i = 0; i < kReadsPerThread; ++i) {
        std::shared_lock<dispenso::DistributedRWLock<>> lk(mtx);
        EXPECT_EQ(sharedData, 123);
        readCount.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto& t : readers) {
    t.join();
  }

  EXPECT_EQ(readCount.load(), kNumThreads * kReadsPerThread);
}

TEST(DistributedRWLock, WriterExclusion) {
  dispenso::DistributedRWLock<> mtx;
  int count = 0;
  constexpr int kPerThreadTotal = 50000;

  auto writerFn = [&]() {
    for (int i = 0; i < kPerThreadTotal; ++i) {
      std::unique_lock<dispenso::DistributedRWLock<>> lk(mtx);
      ++count;
    }
  };

  std::thread t0(writerFn);
  std::thread t1(writerFn);
  t0.join();
  t1.join();

  EXPECT_EQ(count, 2 * kPerThreadTotal);
}

TEST(DistributedRWLock, ReaderWriterContention) {
  dispenso::DistributedRWLock<> mtx;
  int guardedCount = 0;
  constexpr int kWriterTotal = 10000;
  constexpr int kReaderTotal = 100000;

  auto writerFn = [&]() {
    for (int i = 0; i < kWriterTotal; ++i) {
      std::unique_lock<dispenso::DistributedRWLock<>> lk(mtx);
      ++guardedCount;
    }
  };

  auto readerFn = [&]() {
    for (int i = 0; i < kReaderTotal; ++i) {
      std::shared_lock<dispenso::DistributedRWLock<>> lk(mtx);
      // Under the shared lock the writer cannot be mid-update, so each read
      // observes a committed value in [0, kWriterTotal]; a torn read from a
      // broken lock would fall outside that range.
      int val = guardedCount;
      EXPECT_GE(val, 0);
      EXPECT_LE(val, kWriterTotal);
    }
  };

  std::thread writer(writerFn);
  std::thread reader(readerFn);
  writer.join();
  reader.join();

  EXPECT_EQ(guardedCount, kWriterTotal);
}

TEST(DistributedRWLock, TryLockShared) {
  dispenso::DistributedRWLock<> mtx;

  // Should succeed when no writer holds the lock
  EXPECT_TRUE(mtx.try_lock_shared());
  mtx.unlock_shared();

  // Should fail when writer holds the lock
  mtx.lock();
  // try_lock_shared from the same thread while holding write lock
  // is UB for most mutexes, but we can test try_lock from another thread
  std::atomic<bool> gotLock{false};
  std::thread t([&]() { gotLock.store(mtx.try_lock_shared()); });
  t.join();
  EXPECT_FALSE(gotLock.load());
  mtx.unlock();
}

TEST(DistributedRWLock, TryLock) {
  dispenso::DistributedRWLock<> mtx;

  // Should succeed when unlocked
  EXPECT_TRUE(mtx.try_lock());
  mtx.unlock();

  // Should fail when another writer holds it
  mtx.lock();
  std::atomic<bool> gotLock{false};
  std::thread t([&]() { gotLock.store(mtx.try_lock()); });
  t.join();
  EXPECT_FALSE(gotLock.load());
  mtx.unlock();

  // The failed try_lock must have rolled back its partial writer-bit acquisition
  // cleanly — the lock should still be acquirable.
  EXPECT_TRUE(mtx.try_lock());
  mtx.unlock();
}

// ---------------------------------------------------------------------------
// DistributedRWLockImpl (intrusive, explicit index)
// ---------------------------------------------------------------------------

TEST(DistributedRWLockImpl, ExplicitIndex) {
  dispenso::detail::DistributedRWLockImpl<16> lock;

  // Readers on distinct slots are mutually compatible and can be held together.
  lock.lock_shared(0);
  lock.lock_shared(7);
  lock.lock_shared(15);
  // A reader on yet another distinct slot still succeeds while the others are held.
  EXPECT_TRUE(lock.try_lock_shared(3));
  // The same slot also permits multiple concurrent shared holders.
  EXPECT_TRUE(lock.try_lock_shared(0));
  lock.unlock_shared(0); // release the extra slot-0 holder
  lock.unlock_shared(3);
  lock.unlock_shared(15);
  lock.unlock_shared(7);
  lock.unlock_shared(0);

  // With all readers released, try_lock() (writer) succeeds without blocking
  // (this is the Impl try_lock happy path; WriterBlocksReaders only covers
  // the blocking lock()).
  ASSERT_TRUE(lock.try_lock());
  // ...and now excludes readers on every slot.
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_FALSE(lock.try_lock_shared(i));
  }
  lock.unlock();

  // After release, a reader can acquire again.
  EXPECT_TRUE(lock.try_lock_shared(5));
  lock.unlock_shared(5);
}

TEST(DistributedRWLockImpl, WriterBlocksReaders) {
  dispenso::detail::DistributedRWLockImpl<8> lock;

  lock.lock(); // Acquire exclusive

  // All try_lock_shared attempts should fail while writer holds
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_FALSE(lock.try_lock_shared(i));
  }

  lock.unlock();

  // Now they should all succeed
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(lock.try_lock_shared(i));
  }
  for (size_t i = 0; i < 8; ++i) {
    lock.unlock_shared(i);
  }
}

TEST(DistributedRWLockImpl, ConcurrentReadersExplicitIndex) {
  dispenso::detail::DistributedRWLockImpl<16> lock;
  std::atomic<int> readCount{0};
  constexpr int kNumThreads = 8;
  constexpr int kReadsPerThread = 10000;

  std::vector<std::thread> readers;
  for (int t = 0; t < kNumThreads; ++t) {
    readers.emplace_back([&, t]() {
      size_t slot = static_cast<size_t>(t);
      for (int i = 0; i < kReadsPerThread; ++i) {
        lock.lock_shared(slot);
        readCount.fetch_add(1, std::memory_order_relaxed);
        lock.unlock_shared(slot);
      }
    });
  }

  for (auto& th : readers) {
    th.join();
  }

  EXPECT_EQ(readCount.load(), kNumThreads * kReadsPerThread);
}

TEST(DistributedRWLockImpl, SmallN) {
  // Verify the smallest useful N works correctly
  dispenso::detail::DistributedRWLockImpl<2> lock;

  // Index masking: index 5 maps to slot 1 (5 & 1 == 1)
  lock.lock_shared(0);
  lock.lock_shared(5); // maps to slot 1
  lock.unlock_shared(5);
  lock.unlock_shared(0);

  lock.lock();
  EXPECT_FALSE(lock.try_lock_shared(0));
  EXPECT_FALSE(lock.try_lock_shared(1));
  lock.unlock();
}

TEST(DistributedRWLock, Alignment) {
  static_assert(
      alignof(dispenso::DistributedRWLock<>) >= dispenso::kCacheLineSize,
      "DistributedRWLock must be cache-line aligned");
}

TEST(DistributedRWLock, SharedLockCompat) {
  // Verify std::shared_lock<DistributedRWLock> compiles and works
  dispenso::DistributedRWLock<> mtx;
  std::shared_lock<dispenso::DistributedRWLock<>> lk(mtx);
  lk.unlock();
  lk.lock();
}

TEST(DistributedRWLock, UniqueLockCompat) {
  // Verify std::unique_lock<DistributedRWLock> compiles and works
  dispenso::DistributedRWLock<> mtx;
  std::unique_lock<dispenso::DistributedRWLock<>> lk(mtx);
  lk.unlock();
  lk.lock();
}

// ---------------------------------------------------------------------------
// Stress tests — exercise contention paths at scale
// ---------------------------------------------------------------------------

// Stress test with many readers and writers to exercise:
// - lock_shared backoff loop (reader arrives while writer holds)
// - readerRelease waking a sleeping writer (last reader drains)
// - setWriteBit spin (writer-writer contention)
// - waitForReaderDrain via OS wait (writer waits for active readers)
TEST(DistributedRWLockImpl, StressManyReadersWriters) {
  dispenso::detail::DistributedRWLockImpl<8> lock;
  // Plain (non-atomic) so a broken lock races/loses updates instead of the
  // self-synchronizing atomic RMW masking the defect.
  int64_t guardedValue{0};
  std::atomic<int64_t> writeCount{0};

  constexpr int kNumWriters = 4;
  constexpr int kNumReaders = 12;
  constexpr int kWriteIters = 20000;
  constexpr int kReadIters = 50000;

  std::vector<std::thread> threads;

  // Writers: increment guardedValue under exclusive lock
  for (int w = 0; w < kNumWriters; ++w) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kWriteIters; ++i) {
        lock.lock();
        ++guardedValue;
        lock.unlock();
      }
      writeCount.fetch_add(kWriteIters, std::memory_order_relaxed);
    });
  }

  // Readers: every read under the shared lock must observe a committed value in
  // [0, kNumWriters*kWriteIters]; a torn read from a broken lock would not.
  for (int r = 0; r < kNumReaders; ++r) {
    threads.emplace_back([&, r]() {
      size_t slot = static_cast<size_t>(r);
      for (int i = 0; i < kReadIters; ++i) {
        lock.lock_shared(slot);
        int64_t val = guardedValue;
        EXPECT_GE(val, 0);
        EXPECT_LE(val, static_cast<int64_t>(kNumWriters) * kWriteIters);
        lock.unlock_shared(slot);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(guardedValue, static_cast<int64_t>(kNumWriters) * kWriteIters);
  EXPECT_EQ(writeCount.load(), static_cast<int64_t>(kNumWriters) * kWriteIters);
}

// Stress test mixing try_lock and try_lock_shared with blocking lock variants
// to exercise rollback paths and contention between try/blocking callers.
TEST(DistributedRWLockImpl, StressTryLockContention) {
  dispenso::detail::DistributedRWLockImpl<4> lock;
  // Plain (non-atomic) so a broken lock races/loses updates instead of the
  // self-synchronizing atomic RMW masking the defect.
  int guardedCount{0};
  std::atomic<int> tryWriteSuccesses{0};
  std::atomic<int> tryReadSuccesses{0};

  constexpr int kNumThreads = 8;
  constexpr int kIters = 30000;

  std::vector<std::thread> threads;

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      size_t slot = static_cast<size_t>(t);
      for (int i = 0; i < kIters; ++i) {
        // Alternate between try and blocking variants
        if (i % 4 == 0) {
          // Blocking write
          lock.lock();
          ++guardedCount;
          lock.unlock();
        } else if (i % 4 == 1) {
          // Try write — may fail due to contention (exercises rollback)
          if (lock.try_lock()) {
            ++guardedCount;
            tryWriteSuccesses.fetch_add(1, std::memory_order_relaxed);
            lock.unlock();
          }
        } else if (i % 4 == 2) {
          // Blocking read: the value seen under the shared lock must be a
          // committed count within range (never torn/garbage).
          lock.lock_shared(slot);
          int val = guardedCount;
          EXPECT_GE(val, 0);
          EXPECT_LE(val, kNumThreads * kIters);
          lock.unlock_shared(slot);
        } else {
          // Try read — may fail if a writer holds the lock
          if (lock.try_lock_shared(slot)) {
            int val = guardedCount;
            EXPECT_GE(val, 0);
            EXPECT_LE(val, kNumThreads * kIters);
            tryReadSuccesses.fetch_add(1, std::memory_order_relaxed);
            lock.unlock_shared(slot);
          }
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Verify: blocking writes always succeed, try writes succeed at least sometimes
  int expectedBlockingWrites = kNumThreads * (kIters / 4);
  EXPECT_EQ(guardedCount, expectedBlockingWrites + tryWriteSuccesses.load());
  EXPECT_GT(tryWriteSuccesses.load(), 0);
  EXPECT_GT(tryReadSuccesses.load(), 0);
}

// Stress the DistributedRWLock public wrapper under high thread count to
// exercise the TLS slot caching and ensure no slot collisions cause correctness
// issues.
TEST(DistributedRWLock, StressHighThreadCount) {
  dispenso::DistributedRWLock<16> mtx;
  int64_t guardedValue = 0;

  constexpr int kNumWriters = 4;
  constexpr int kNumReaders = 28;
  constexpr int kWriteIters = 10000;
  constexpr int kReadIters = 50000;

  std::vector<std::thread> threads;

  for (int w = 0; w < kNumWriters; ++w) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kWriteIters; ++i) {
        std::unique_lock<dispenso::DistributedRWLock<16>> lk(mtx);
        ++guardedValue;
      }
    });
  }

  for (int r = 0; r < kNumReaders; ++r) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kReadIters; ++i) {
        std::shared_lock<dispenso::DistributedRWLock<16>> lk(mtx);
        // Each read observes a committed value in [0, kNumWriters*kWriteIters].
        int64_t val = guardedValue;
        EXPECT_GE(val, 0);
        EXPECT_LE(val, static_cast<int64_t>(kNumWriters) * kWriteIters);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(guardedValue, static_cast<int64_t>(kNumWriters) * kWriteIters);
}
