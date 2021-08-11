// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <vector>

#include <dispenso/detail/math.h>
#include <dispenso/platform.h>
#include <dispenso/tsan_annotations.h>

#include <concurrentqueue.h>

namespace dispenso {
namespace detail {

struct SmallBufferGlobals {
  moodycamel::ConcurrentQueue<char*> centralStore;
  std::vector<char*> backingStore;
  std::atomic<uint32_t> backingStoreLock{0};

  ~SmallBufferGlobals() {
    for (char* b : backingStore) {
      alignedFree(b);
    }
  }
};

template <size_t kChunkSize>
DISPENSO_DLL_ACCESS SmallBufferGlobals& getSmallBufferGlobals();

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
  static constexpr size_t kLogFactor = log2const(kChunkSize | 1);

  // TODO(T88183021): Make these factors compile-time configurable.  For example, the current values
  // can lead to megabytes of data being allocated, even if the alloctor is only used for one or two
  // allocations.  Likely we can reduce these sizes by a decent factor without affecting benchmarks,
  // and then reduce them even further as an option.
  static constexpr size_t kMallocBytes = (1 << 12) * kLogFactor;
  static constexpr size_t kIdealTLCacheBytes = kMallocBytes / 4;
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
    auto bnc = buffersAndCount();
    char** tlBuffers = std::get<0>(bnc);
    size_t& tlCount = std::get<1>(bnc);
    if (!tlCount) {
      // We only need to register (at least) once when we grab from the central store; without going
      // to the central store at least once (or calling dealloc first), we cannot have buffers to
      // return.
      registerCleanup();
      tlCount = grabFromCentralStore(tlBuffers);
    }
    return tlBuffers[--tlCount];
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
    auto bnc = buffersAndCount();
    char** tlBuffers = std::get<0>(bnc);
    size_t& tlCount = std::get<1>(bnc);
    registerCleanup();
    tlBuffers[tlCount++] = buffer;
    if (tlCount == kMaxNumTLBuffers) {
      recycleToCentralStore(tlBuffers + kIdealNumTLBuffers, kIdealNumTLBuffers);
      tlCount -= kIdealNumTLBuffers;
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
    auto& globals = getSmallBufferGlobals<kChunkSize>();
    auto& lock = globals.backingStoreLock;
    while (!lock.compare_exchange_weak(allocId, 1, std::memory_order_acquire)) {
    }
    size_t bytes = kMallocBytes * globals.backingStore.size();
    lock.store(0, std::memory_order_release);
    return bytes;
  }

 private:
  struct PerThreadQueuingData {
    PerThreadQueuingData(
        moodycamel::ConcurrentQueue<char*>& cstore,
        std::tuple<char**, size_t&> buffersAndCount)
        : cstore_(cstore),
          ptoken_(cstore),
          ctoken_(cstore),
          buffers_(std::get<0>(buffersAndCount)),
          count_(std::get<1>(buffersAndCount)) {}

    ~PerThreadQueuingData() {
      enqueue_bulk(buffers_, count_);
    }

    void enqueue_bulk(char** buffers, size_t count) {
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
      cstore_.enqueue_bulk(ptoken_, buffers, count);
      DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
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

  static void registerCleanup() {
    // Note that this would be better/cheaper as a static thread_local member; however, there are
    // currently bugs in multiple compilers that prevent the destructor from being called properly
    // in that context.  A workaround that appears to work more portably is to put this here, and
    // ensure registerCleanup is called for any alloc or dealloc, even though this may have a small
    // runtime cost.
    (void)getThreadQueuingData();
  }

  static size_t grabFromCentralStore(char** buffers) {
    auto& queue = getThreadQueuingData();
    auto& globals = getSmallBufferGlobals<kChunkSize>();
    auto& lock = globals.backingStoreLock;
    auto& backingStore = globals.backingStore;
    while (true) {
      size_t grabbed = queue.try_dequeue_bulk(buffers, kIdealNumTLBuffers);
      if (grabbed) {
        return grabbed;
      }
      uint32_t allocId = lock.fetch_add(1, std::memory_order_acquire);
      if (allocId == 0) {
        char* buffer = reinterpret_cast<char*>(detail::alignedMalloc(kMallocBytes, kChunkSize));
        backingStore.push_back(buffer);

        constexpr size_t kNumToPush = kBuffersPerMalloc - kIdealNumTLBuffers;
        char* topush[kNumToPush];
        for (size_t i = 0; i < kNumToPush; ++i, buffer += kChunkSize) {
          topush[i] = buffer;
        }
        queue.enqueue_bulk(topush, kNumToPush);
        lock.store(0, std::memory_order_release);
        for (size_t i = 0; i < kIdealNumTLBuffers; ++i, buffer += kChunkSize) {
          buffers[i] = buffer;
        }
        return kIdealNumTLBuffers;
      } else {
        while (lock.load(std::memory_order_relaxed)) {
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
  DISPENSO_DLL_ACCESS static std::tuple<char**, size_t&> buffersAndCount() {
    static DISPENSO_THREAD_LOCAL char* tlBuffers[kMaxNumTLBuffers];
    static DISPENSO_THREAD_LOCAL size_t tlCount = 0;
    return {tlBuffers, tlCount};
  };
  DISPENSO_DLL_ACCESS static PerThreadQueuingData& getThreadQueuingData() {
    static thread_local PerThreadQueuingData data(
        getSmallBufferGlobals<kChunkSize>().centralStore, buffersAndCount());
    return data;
  }
};

} // namespace detail
} // namespace dispenso
