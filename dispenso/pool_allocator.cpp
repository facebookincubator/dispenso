/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/pool_allocator.h>

namespace dispenso {

template <bool kThreadSafe>
PoolAllocatorT<kThreadSafe>::PoolAllocatorT(
    size_t chunkSize,
    size_t allocSize,
    std::function<void*(size_t)> allocFunc,
    std::function<void(void*)> deallocFunc)
    : chunkSize_(chunkSize),
      allocSize_(allocSize),
      chunksPerAlloc_(allocSize / chunkSize),
      allocFunc_(std::move(allocFunc)),
      deallocFunc_(std::move(deallocFunc)) {
  // Start off with at least enough space to store at least one set of chunks.
  chunks_.reserve(chunksPerAlloc_);
}

template <bool kThreadSafe>
char* PoolAllocatorT<kThreadSafe>::alloc() {
  while (true) {
    uint32_t allocId = 0;
    if (kThreadSafe) {
      allocId = backingAllocLock_.fetch_or(1, std::memory_order_acquire);
    }

    if (allocId == 0) {
      if (chunks_.empty()) {
        char* buffer;
        if (backingAllocs2_.empty()) {
          buffer = reinterpret_cast<char*>(allocFunc_(allocSize_));
        } else {
          buffer = backingAllocs2_.back();
          backingAllocs2_.pop_back();
        }
        backingAllocs_.push_back(buffer);
        // Push n-1 values into the chunks_ buffer, and then return the nth.
        for (size_t i = 0; i < chunksPerAlloc_ - 1; ++i) {
          chunks_.push_back(buffer);
          buffer += chunkSize_;
        }
        if (kThreadSafe) {
          backingAllocLock_.store(0, std::memory_order_release);
        }
        return buffer;
      }
      char* back = chunks_.back();
      chunks_.pop_back();
      if (kThreadSafe) {
        backingAllocLock_.store(0, std::memory_order_release);
      }
      return back;
    } else {
      std::this_thread::yield();
    }
  }
}

template <bool kThreadSafe>
void PoolAllocatorT<kThreadSafe>::dealloc(char* ptr) {
  // For now do not release any memory back to the deallocFunc until destruction.
  // TODO(bbudge): Consider cases where we haven't gotten below some threshold of ready chunks
  // in a while.  In that case, we could begin tracking allocations, and try to assemble entire
  // starting allocations, possibly deferring a small amount to each alloc call.  This would be
  // slower, but would ensure we don't get into a situation where we need a bunch of memory up
  // front, and then never again.

  while (true) {
    uint32_t allocId = 0;
    if (kThreadSafe) {
      allocId = backingAllocLock_.fetch_or(1, std::memory_order_acquire);
    }
    if (allocId == 0) {
      chunks_.push_back(ptr);
      if (kThreadSafe) {
        backingAllocLock_.store(0, std::memory_order_release);
      }
      break;
    }
  }
}

template <bool kThreadSafe>
void PoolAllocatorT<kThreadSafe>::clear() {
  chunks_.clear();
  if (backingAllocs2_.size() < backingAllocs_.size()) {
    std::swap(backingAllocs2_, backingAllocs_);
  }
  for (char* ba : backingAllocs_) {
    backingAllocs2_.push_back(ba);
  }
  backingAllocs_.clear();
}

template <bool kThreadSafe>
PoolAllocatorT<kThreadSafe>::~PoolAllocatorT() {
  for (char* backing : backingAllocs_) {
    deallocFunc_(backing);
  }
  for (char* backing : backingAllocs2_) {
    deallocFunc_(backing);
  }
}

template class PoolAllocatorT<false>;
template class PoolAllocatorT<true>;

} // namespace dispenso
