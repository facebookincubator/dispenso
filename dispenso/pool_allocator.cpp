/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/pool_allocator.h>

namespace dispenso {

PoolAllocator::PoolAllocator(
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

char* PoolAllocator::alloc() {
  while (true) {
    uint32_t allocId = backingAllocLock_.fetch_add(1, std::memory_order_acquire);

    if (allocId == 0) {
      if (chunks_.empty()) {
        char* buffer = reinterpret_cast<char*>(allocFunc_(allocSize_));
        backingAllocs_.push_back(buffer);
        // Push n-1 values into the chunks_ buffer, and then return the nth.
        for (size_t i = 0; i < chunksPerAlloc_ - 1; ++i) {
          chunks_.push_back(buffer);
          buffer += chunkSize_;
        }
        backingAllocLock_.store(0, std::memory_order_release);
        return buffer;
      }
      char* back = chunks_.back();
      chunks_.pop_back();
      backingAllocLock_.store(0, std::memory_order_release);
      return back;
    } else {
      std::this_thread::yield();
    }
  }
}

void PoolAllocator::dealloc(char* ptr) {
  // For now do not release any memory back to the deallocFunc until destruction.
  // TODO(bbudge): Consider cases where we haven't gotten below some threshold of ready chunks
  // in a while.  In that case, we could begin tracking allocations, and try to assemble entire
  // starting allocations, possibly deferring a small amount to each alloc call.  This would be
  // slower, but would ensure we don't get into a situation where we need a bunch of memory up
  // front, and then never again.

  while (true) {
    uint32_t allocId = backingAllocLock_.fetch_add(1, std::memory_order_acquire);
    if (allocId == 0) {
      chunks_.push_back(ptr);
      backingAllocLock_.store(0, std::memory_order_release);
      break;
    }
  }
}

PoolAllocator::~PoolAllocator() {
  for (char* backing : backingAllocs_) {
    deallocFunc_(backing);
  }
}

} // namespace dispenso
