// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

/**
 * @file small_buffer_allocator.h
 * A file providing SmallBufferAllocator.  This allocator can allocate and deallocate chunks of a
 * set size in a way that is efficient and scales quite well across many threads.
 **/

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

  // TODO(T88183021): Make these factors compile-time configurable.  For example, the current values
  // can lead to megabytes of data being allocated, even if the alloctor is only used for one or two
  // allocations.  Likely we can reduce these sizes by a decent factor without affecting benchmarks,
  // and then reduce them even further as an option.
  static constexpr size_t kMallocBytes = (1 << 18) * kLogFactor;
  static constexpr size_t kIdealTLCacheBytes = (1 << 16) * kLogFactor;
  static constexpr size_t kIdealNumTLBuffers = kIdealTLCacheBytes / kChunkSize;
  static constexpr size_t kMaxNumTLBuffers = 2 * kIdealNumTLBuffers;
  static constexpr size_t kBuffersPerMalloc = kMallocBytes / kChunkSize;

  static_assert(kIdealNumTLBuffers > 0, "Must have a positive number of buffers to work with");

 public:
  /**
   * Allocate a buffer of <code>kChunkSize</code> bytes.
   *
   * @return a pointer to the buffer.
   **/
  static char* alloc() {
    if (!tlCount_) {
      // We only need to register (at least) once when we grab from the central store; without going
      // to the central store at least once (or calling dealloc first), we cannot have buffers to
      // return.
      registerCleanup();
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
    // We need to register at least once for any call to dealloc, because we may dealloc on this
    // thread without having allocated on the same thread, and if we don't register, any memory in
    // the thread-local buffers will not be returned to the central store on thread destruction.
    registerCleanup();
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
  struct PerThreadQueuingData {
    PerThreadQueuingData(
        moodycamel::ConcurrentQueue<char*>& cstore,
        char** tlBuffers,
        size_t& tlCount)
        : cstore_(cstore), ptoken_(cstore), ctoken_(cstore), buffers_(tlBuffers), count_(tlCount) {}

    ~PerThreadQueuingData() {
      enqueue_bulk(buffers_, count_);
    }

    void enqueue_bulk(char** buffers, size_t count) {
      cstore_.enqueue_bulk(ptoken_, buffers, count);
    }

    size_t try_dequeue_bulk(char** buffers, size_t count) {
      return cstore_.try_dequeue_bulk(ctoken_, buffers, count);
    }

   private:
    moodycamel::ConcurrentQueue<char*>& cstore_;
    moodycamel::ProducerToken ptoken_;
    moodycamel::ConsumerToken ctoken_;
    char** buffers_;
    size_t& count_;
  };

  static PerThreadQueuingData& getThreadQueuingData() {
    static thread_local PerThreadQueuingData data(centralStore(), tlBuffers_, tlCount_);
    return data;
  }

  static void registerCleanup() {
    // Note that this would be better/cheaper as a static thread_local member; however, there are
    // currently bugs in multiple compilers that prevent the destructor from being called properly
    // in that context.  A workaround that appears to work more portably is to put this here, and
    // ensure registerCleanup is called for any alloc or dealloc, even though this may have a small
    // runtime cost.
    (void)getThreadQueuingData();
  }

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

  static size_t grabFromCentralStore(char** buffers) {
    auto& queue = getThreadQueuingData();
    while (true) {
      size_t grabbed = queue.try_dequeue_bulk(buffers, kIdealNumTLBuffers);
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
        queue.enqueue_bulk(topush, kNumToPush);
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
    getThreadQueuingData().enqueue_bulk(buffers, numToRecycle);
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

namespace detail {
// Set a standard for the maximum chunk size for use within dispenso.  Diminishing returns after a
// certain size, and each new pool has it's own memory overhead.
constexpr size_t kMaxSmallBufferSize = 256;

// Within dispenso, enable the user to disable use of SmallBufferAllocator.  Some small embedded
// applications only submit a few closures concurrently, and SmallBufferAllocator is overkill.
// Additionally, SmallBufferAllocator has large memory overhead in such environments (TODO to make
// that better, but it will never be as lean as malloc/free).
template <size_t kBlockSize>
void* allocSmallBuffer() {
#if defined(DISPENSO_NO_SMALL_BUFFER_ALLOCATOR)
  return alignedMalloc(kBlockSize, kBlockSize);
#else
  return SmallBufferAllocator<kBlockSize>::alloc();
#endif // DISPENSO_NO_SMALL_BUFFER_ALLOCATOR
}

template <size_t kBlockSize>
void deallocSmallBuffer(void* buf) {
#if defined(DISPENSO_NO_SMALL_BUFFER_ALLOCATOR)
  alignedFree(buf);
#else
  SmallBufferAllocator<kBlockSize>::dealloc(reinterpret_cast<char*>(buf));
#endif // DISPENSO_NO_SMALL_BUFFER_ALLOCATOR
}
} // namespace detail

} // namespace dispenso
