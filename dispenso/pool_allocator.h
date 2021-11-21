// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include <dispenso/platform.h>

namespace dispenso {

/**
 * A pool allocator to help reduce calls to the underlying allocation and deallocation functions.
 **/
class PoolAllocator {
 public:
  /**
   * Construct a PoolAllocator.
   *
   * @param chunkSize The chunk size for each pool allocation
   * @param allocSize The size of underlying slabs to be chunked
   * @param allocFunc The underlying allocation function for allocating slabs
   * @param deallocFunc The underlying deallocation function.  Currently only called on destruction.
   **/
  PoolAllocator(
      size_t chunkSize,
      size_t allocSize,
      std::function<void*(size_t)> allocFunc,
      std::function<void(void*)> deallocFunc);

  /**
   * Allocate a chunk from a slab
   *
   * @return The pointer to a buffer of chunkSize bytes
   **/
  char* alloc();

  /**
   * Deallocate a previously allocated chunk
   *
   * @param ptr The chunk to return to the available pool
   **/
  void dealloc(char* ptr);

  /**
   * Destruct a PoolAllocator
   **/
  ~PoolAllocator();

 private:
  const size_t chunkSize_;
  const size_t allocSize_;
  const size_t chunksPerAlloc_;

  std::function<void*(size_t)> allocFunc_;
  std::function<void(void*)> deallocFunc_;

  // Use of a spin lock was found to be faster than std::mutex in benchmarks.
  alignas(kCacheLineSize) std::atomic<uint32_t> backingAllocLock_{0};
  std::vector<char*> backingAllocs_;

  std::vector<char*> chunks_;
};

} // namespace dispenso
