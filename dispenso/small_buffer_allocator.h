// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <vector>

#include <dispenso/platform.h>

#include <concurrentqueue.h>

namespace dispenso {

/**
 * A class for allocating small chunks of memory quickly.  The class is built on concepts of
 * thread-local pools of buffers of specific sizes.  It is best to limit the distinct number of
 * sizes of chunk sizes used in one program or there is a good chance that there may be a lot of
 * unused memory allocated in the system.
 *
 * <code>SmallBufferAllocator</code> is completely thread-safe.
 **/
template <size_t kChunkSize>
class SmallBufferAllocator {
 private:
  static constexpr size_t log2(size_t n) {
    return ((n < 2) ? 1 : 1 + log2(n / 2));
  }

  static constexpr size_t kLogFactor = log2(kChunkSize);

  static constexpr size_t kMallocBytes = (1 << 18) * kLogFactor;
  static constexpr size_t kIdealTLCacheBytes = (1 << 16) * kLogFactor;
  static constexpr size_t kIdealNumTLBuffers = kIdealTLCacheBytes / kChunkSize;
  static constexpr size_t kMaxNumTLBuffers = 2 * kIdealNumTLBuffers;
  static constexpr size_t kBuffersPerMalloc = kMallocBytes / kChunkSize;

  static_assert(kIdealNumTLBuffers > 0);

 public:
  /**
   * Allocate a buffer of <code>kChunkSize</code> bytes.
   *
   * @return a pointer to the buffer.
   **/
  static char* alloc() {
    if (!tlCount_) {
      tlCount_ = grabFromCentralStore(tlBuffers_);
    }
    return tlBuffers_[--tlCount_];
  }

  /**
   * Deallocate a buffer previously allocated via <code>alloc</code>
   *
   * @param buffer The buffer to deallocate.
   **/
  static void dealloc(char* buffer) {
    tlBuffers_[tlCount_++] = buffer;
    if (tlCount_ == kMaxNumTLBuffers) {
      recycleToCentralStore(tlBuffers_ + kIdealNumTLBuffers, kIdealNumTLBuffers);
      tlCount_ -= kIdealNumTLBuffers;
    }
  }

  /**
   * Get the approximate number of underlying bytes allocated by the allocator.  This is mostly for
   * testing and debugging.
   *
   * @return The approximate number of bytes currently allocated.
   **/
  static size_t bytesAllocated() {
    uint32_t allocId = 0;
    while (!backingStoreLock_.compare_exchange_weak(allocId, 1, std::memory_order_acquire)) {
    }
    size_t bytes = kMallocBytes * backingStore().size();
    backingStoreLock_.store(0, std::memory_order_release);
    return bytes;
  }

 private:
  static moodycamel::ConcurrentQueue<char*>& centralStore() {
    // Controlled leak to avoid static destruction order fiasco.
    static moodycamel::ConcurrentQueue<char*>* queue = new moodycamel::ConcurrentQueue<char*>();
    return *queue;
  }
  static std::vector<char*>& backingStore() {
    // Controlled leak to avoid static destruction order fiasco.
    static std::vector<char*>* buffers = new std::vector<char*>();
    return *buffers;
  }

  static moodycamel::ProducerToken& pToken() {
    static thread_local moodycamel::ProducerToken token(centralStore());
    return token;
  }
  static moodycamel::ConsumerToken& cToken() {
    static thread_local moodycamel::ConsumerToken token(centralStore());
    return token;
  }

  static size_t grabFromCentralStore(char** buffers) {
    static thread_local struct ThreadCleanup {
      ~ThreadCleanup() {
        SmallBufferAllocator<kChunkSize>::centralStore().enqueue_bulk(buffers, count);
      }
      char** buffers;
      size_t& count;
    } cleanup{tlBuffers_, tlCount_};

    auto& cstore = centralStore();
    while (true) {
      size_t grabbed = cstore.try_dequeue_bulk(cToken(), buffers, kIdealNumTLBuffers);
      if (grabbed) {
        return grabbed;
      }
      uint32_t allocId = backingStoreLock_.fetch_add(1, std::memory_order_acquire);
      if (allocId == 0) {
        char* buffer = reinterpret_cast<char*>(detail::alignedMalloc(kMallocBytes, kChunkSize));
        backingStore().push_back(buffer);

        constexpr size_t kNumToPush = kBuffersPerMalloc - kIdealNumTLBuffers;
        char* topush[kNumToPush];
        for (size_t i = 0; i < kNumToPush; ++i, buffer += kChunkSize) {
          topush[i] = buffer;
        }
        cstore.enqueue_bulk(pToken(), topush, kNumToPush);
        backingStoreLock_.store(0, std::memory_order_release);
        for (size_t i = 0; i < kIdealNumTLBuffers; ++i, buffer += kChunkSize) {
          buffers[i] = buffer;
        }
        return kIdealNumTLBuffers;
      } else {
        while (backingStoreLock_.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
      }
    }
  }

  static void recycleToCentralStore(char** buffers, size_t numToRecycle) {
    centralStore().enqueue_bulk(pToken(), buffers, numToRecycle);
    // TODO(bbudge): consider whether we need to do any garbage collection and return memory to
    // the system.
  }

 private:
  // Prefer "dumb" thread local to thread_local when possible to avoid overheads.
  static DISPENSO_THREAD_LOCAL char* tlBuffers_[kMaxNumTLBuffers];
  static DISPENSO_THREAD_LOCAL size_t tlCount_;
  static std::atomic<uint32_t> backingStoreLock_;
};

template <size_t kChunkSize>
DISPENSO_THREAD_LOCAL char* SmallBufferAllocator<
    kChunkSize>::tlBuffers_[SmallBufferAllocator<kChunkSize>::kMaxNumTLBuffers];
template <size_t kChunkSize>
DISPENSO_THREAD_LOCAL size_t SmallBufferAllocator<kChunkSize>::tlCount_ = 0;

template <size_t kChunkSize>
std::atomic<uint32_t> SmallBufferAllocator<kChunkSize>::backingStoreLock_(0);
} // namespace dispenso
