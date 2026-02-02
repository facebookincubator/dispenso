/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file pool_allocator.h
 * @ingroup group_alloc
 * A pool allocator to help reduce calls to the underlying allocation and deallocation functions
 * that can be provided custom backing allocation and deallocation functions, e.g. cudaMalloc,
 * cudaFree.
 **/

#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include <dispenso/platform.h>

namespace dispenso {

/**
 * A pool allocator to help reduce calls to the underlying allocation and deallocation functions.
 **/
template <bool kThreadSafe>
class PoolAllocatorT {
 public:
  /**
   * Construct a PoolAllocator.
   *
   * @param chunkSize The chunk size for each pool allocation
   * @param allocSize The size of underlying slabs to be chunked
   * @param allocFunc The underlying allocation function for allocating slabs
   * @param deallocFunc The underlying deallocation function.  Currently only called on destruction.
   **/
  DISPENSO_DLL_ACCESS PoolAllocatorT(
      size_t chunkSize,
      size_t allocSize,
      std::function<void*(size_t)> allocFunc,
      std::function<void(void*)> deallocFunc);

  /**
   * Allocate a chunk from a slab
   *
   * @return The pointer to a buffer of chunkSize bytes
   **/
  DISPENSO_DLL_ACCESS char* alloc();

  /**
   * Deallocate a previously allocated chunk
   *
   * @param ptr The chunk to return to the available pool
   **/
  DISPENSO_DLL_ACCESS void dealloc(char* ptr);

  /**
   * Effectively dealloc all previously allocated chunks.  Useful for arenas.
   * This function is not thread safe, and no previously allocated chunks may be dealloc'd after
   * clear.
   **/
  DISPENSO_DLL_ACCESS void clear();

  /**
   * Get the total capicity allocated in chunks (how many alloc() could be called without triggering
   * allocFunc() if all chunks were available)
   **/
  size_t totalChunkCapacity() const {
    return (backingAllocs2_.size() + backingAllocs_.size()) * chunksPerAlloc_;
  }
  /**
   * Destruct a PoolAllocator
   **/
  DISPENSO_DLL_ACCESS ~PoolAllocatorT();

 private:
  const size_t chunkSize_;
  const size_t allocSize_;
  const size_t chunksPerAlloc_;

  std::function<void*(size_t)> allocFunc_;
  std::function<void(void*)> deallocFunc_;

  // Use of a spin lock was found to be faster than std::mutex in benchmarks.
  alignas(kCacheLineSize) std::atomic<uint32_t> backingAllocLock_{0};
  std::vector<char*> backingAllocs_;
  std::vector<char*> backingAllocs2_;

  std::vector<char*> chunks_;
};

/** Thread-safe pool allocator with internal locking. */
using PoolAllocator = PoolAllocatorT<true>;
/** Pool allocator without locking, for single-threaded use or external synchronization. */
using NoLockPoolAllocator = PoolAllocatorT<false>;

} // namespace dispenso
