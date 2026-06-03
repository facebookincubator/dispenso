/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file distributed_rw_lock.h
 * @ingroup group_sync
 * A high-throughput distributed reader/writer lock for read-mostly workloads.
 *
 * DistributedRWLock provides the same interface as std::shared_mutex and is
 * fully compatible with std::shared_lock, std::unique_lock, and std::lock_guard.
 *
 * Reads are near-contention-free: each thread hashes to one of N cache-line-
 * aligned sub-locks, so concurrent readers rarely touch the same cache line.
 * Writes are O(N) — a writer must lock all N sub-locks in sequence — and the
 * lock is a spin lock (no OS backoff), so write contention scales badly.
 *
 * The template parameter N controls the read-side fan-out vs. write cost
 * trade-off. Writer cost is O(N), so N effectively encodes "how rare are
 * writes": the larger N, the lower the write:read ratio must be for this
 * lock to win. N must be a power of 2.
 *
 * ## When to choose this lock
 *
 * Shared/reader-writer locks pay extra bookkeeping per reader; that cost
 * only amortizes when reads vastly outnumber writes. If writes are
 * anywhere near reads (≥10% of ops), prefer std::mutex (or a basic spin
 * lock for very fast critical sections) — both this lock and
 * dispenso::RWLock spin without OS backoff and lose badly to a single-
 * owner mutex once writers contend. std::shared_mutex isn't the right
 * fallback there either: its per-reader bookkeeping costs more than
 * just locking exclusively when writes are common.
 *
 * Within the read-mostly regime:
 *
 * | Workload                                       | Recommended primitive  |
 * |------------------------------------------------|------------------------|
 * | 1-2 contending threads                         | dispenso::RWLock       |
 * | 4+ contending threads, very fast critical sect | DistributedRWLock<N>   |
 * | Slow critical sections (allocation, IO, sleep) | std::shared_mutex      |
 *
 * Choosing N: the default N=16 covers most read-mostly cases. Drop to
 * N=8 if you have only a handful of contending readers, or if writes are
 * slightly less rare — the smaller writer cost is worth more than the
 * extra reader fan-out at low concurrency. N=128 is almost never optimal:
 * reader parallelism saturates well before writer cost does, so the
 * larger N just makes writers worse.
 *
 * **Catastrophic miss case:** at 32 threads with ~50% writes, RWLock is
 * ~10× slower than std::shared_mutex on Linux and DistributedRWLock<16>
 * is ~35× slower on Windows. A plain std::mutex would beat either —
 * shared locks are simply the wrong tool when writes are common.
 *
 * Ideal use case: a data structure resized only on rare events (e.g. a
 * thread pool's per-thread ring array, which only changes on pool resize).
 **/

#pragma once

#include <dispenso/detail/distributed_rw_lock_impl.h>
#include <dispenso/thread_id.h>

namespace dispenso {

/**
 * A distributed reader/writer lock compatible with std::shared_mutex.
 *
 * Uses dispenso::threadId() to pick a sub-lock slot. threadId() is a fast
 * thread-local counter (constant-initialized, lazy-filled on first call) that
 * monotonically distributes threads as 0, 1, 2, ... — masked by N this gives
 * uniform distribution across sub-locks with a single TEB read per call. The
 * lazy-init pattern matters on MSVC, where a `static thread_local T x = ...`
 * with a non-trivial initializer would generate a per-call TLS init guard.
 *
 * @tparam N Number of sub-locks. Must be a power of 2. Higher N = better read
 *           scalability under high thread counts, but O(N) write cost.
 *           Default 16 balances read distribution vs. write overhead.
 *
 * @note Readers and the writer-bit acquisition pure-spin (no OS backoff); only a
 *       writer's reader-drain phase blocks in the OS (futex/WaitOnAddress/condvar)
 *       to avoid burning CPU while it waits for active readers to finish. Best
 *       suited for guarding fast operations with high read traffic and very rare
 *       writes.
 **/
template <size_t N = 16>
class alignas(kCacheLineSize) DistributedRWLock {
 public:
  /** Acquire shared (read) access. */
  void lock_shared() {
    impl_.lock_shared(threadId());
  }

  /** Release shared (read) access. */
  void unlock_shared() {
    impl_.unlock_shared(threadId());
  }

  /** Try to acquire shared (read) access.
   * @return true if acquired, false if a writer holds the lock. */
  bool try_lock_shared() {
    return impl_.try_lock_shared(threadId());
  }

  /** Acquire exclusive (write) access. */
  void lock() {
    impl_.lock();
  }

  /** Try to acquire exclusive (write) access.
   * @return true if acquired, false otherwise. */
  bool try_lock() {
    return impl_.try_lock();
  }

  /** Release exclusive (write) access. */
  void unlock() {
    impl_.unlock();
  }

 private:
  detail::DistributedRWLockImpl<N> impl_;
};

} // namespace dispenso
