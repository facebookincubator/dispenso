/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file small_buffer_allocator.h
 * A file providing SmallBufferAllocator.  This allocator can allocate and deallocate chunks of a
 * set size in a way that is efficient and scales quite well across many threads.
 **/

#pragma once

#include <dispenso/detail/math.h>
#include <dispenso/platform.h>

namespace dispenso {

/**
 * Set a standard for the maximum chunk size for use within dispenso.  The reason for this limit is
 * that there are diminishing returns after a certain size, and each new pool has it's own memory
 * overhead.
 **/
constexpr size_t kMaxSmallBufferSize = 256;

namespace detail {

DISPENSO_DLL_ACCESS char* allocSmallBufferImpl(size_t ordinal);
DISPENSO_DLL_ACCESS void deallocSmallBufferImpl(size_t ordinal, void* buf);

DISPENSO_DLL_ACCESS size_t approxBytesAllocatedSmallBufferImpl(size_t ordinal);

// This has the effect of selecting actual block sizes starting with 4 bytes.  Smaller requests
// (e.g. 1 byte, 2 bytes) will still utilize 4-byte blocks.  Choice of 4 bytes as the smallest
// mainly aligns to sizeof(ptr) on 32-bit platforms, where we'd expect most common use cases to be
// no smaller than one pointer.  Retaining 4-byte buckets on 64-bit platforms doesn't cost much
// (tiny startup/teardown cost, and trivial amount of memory) when not using 4-byte or smaller
// allocations, and makes the code simpler.
constexpr size_t getOrdinal(size_t blockSize) {
  return std::max<ssize_t>(0, log2const(blockSize) - 2);
}

template <size_t kBlockSize>
inline std::enable_if_t<(kBlockSize <= kMaxSmallBufferSize), char*> allocSmallOrLarge() {
#if defined(DISPENSO_NO_SMALL_BUFFER_ALLOCATOR)
  return reinterpret_cast<char*>(alignedMalloc(kBlockSize, kBlockSize));
#else
  return allocSmallBufferImpl(getOrdinal(kBlockSize));
  ;
#endif // DISPENSO_NO_SMALL_BUFFER_ALLOCATOR
}

template <size_t kBlockSize>
inline std::enable_if_t<(kBlockSize > kMaxSmallBufferSize), char*> allocSmallOrLarge() {
  return reinterpret_cast<char*>(alignedMalloc(kBlockSize, kBlockSize));
}

template <size_t kBlockSize>
inline std::enable_if_t<(kBlockSize <= kMaxSmallBufferSize), void> deallocSmallOrLarge(void* buf) {
#if defined(DISPENSO_NO_SMALL_BUFFER_ALLOCATOR)
  alignedFree(buf);
#else
  deallocSmallBufferImpl(getOrdinal(kBlockSize), buf);
#endif // DISPENSO_NO_SMALL_BUFFER_ALLOCATOR
}

template <size_t kBlockSize>
inline std::enable_if_t<(kBlockSize > kMaxSmallBufferSize), void> deallocSmallOrLarge(void* buf) {
  alignedFree(buf);
}

static struct SchwarzSmallBufferInit {
  DISPENSO_DLL_ACCESS SchwarzSmallBufferInit();
  DISPENSO_DLL_ACCESS ~SchwarzSmallBufferInit();
} smallBufferInit;

} // namespace detail

/**
 * Allocate a small buffer from a small buffer pool.
 *
 * @tparam kBlockSize The size of the block to allocate.  Must be a power of two, and must be less
 * than or equal to kMaxSmallBufferSize.
 * @return The pointer to the allocated block of memory.
 * @note: The returned buffer must be returned to the pool via deallocSmallBuffer templatized on the
 * same block size.  If kBlockSize > kMaxSmallBufferSize, this function falls back on alignedMalloc.
 * If DISPENSO_NO_SMALL_BUFFER_ALLOCATOR is defined, we will always fall back on
 * alignedMalloc/alignedFree.
 **/
template <size_t kBlockSize>
inline char* allocSmallBuffer() {
  return detail::allocSmallOrLarge<kBlockSize>();
}
/**
 * Free a small buffer from a small buffer pool.
 *
 * @tparam kBlockSize The size of the block to allocate.  Must be a power of two, and must be less
 * than or equal to kMaxSmallBufferSize.
 * @param buf the pointer to block of memory to return to the pool.  Must have been allocated with
 * allocSmallBuffer templatized on the same block size.
 * @note: If kBlockSize > kMaxSmallBufferSize, this function falls back on alignedFree.
 **/
template <size_t kBlockSize>
inline void deallocSmallBuffer(void* buf) {
  detail::deallocSmallOrLarge<kBlockSize>(buf);
}

/**
 * Get the approximate bytes allocated for a single small buffer pool (associated with
 *kBlockSize). This function is not highly performant and locks, and should only be used for
 *diagnostics (e.g. tests).
 *
 * @tparam kBlockSize The block size for the pool to query.
 **/
template <size_t kBlockSize>
size_t approxBytesAllocatedSmallBuffer() {
  return detail::approxBytesAllocatedSmallBufferImpl(detail::getOrdinal(kBlockSize));
}

} // namespace dispenso
