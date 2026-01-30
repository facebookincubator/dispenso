/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file util.h
 * @ingroup group_util
 * A collection of utility functions and types for memory alignment, bit manipulation,
 * and performance optimization.
 *
 * @note The constant `kCacheLineSize` used by several utilities here is defined in
 *       `<dispenso/platform.h>`, which is included by this header.
 **/

#pragma once

#include <dispenso/detail/math.h>
#include <dispenso/detail/op_result.h>
#include <dispenso/platform.h>

namespace dispenso {

/**
 * @brief Allocate memory with a specified alignment.
 *
 * This function allocates memory aligned to the specified boundary. The alignment
 * must be a power of 2 and at least sizeof(uintptr_t).
 *
 * @param bytes The number of bytes to allocate
 * @param alignment The alignment requirement in bytes (must be power of 2)
 * @return Pointer to aligned memory, or nullptr on allocation failure
 *
 * @note Memory allocated with alignedMalloc must be freed with alignedFree
 * @see alignedFree
 *
 * Example:
 * @code
 * void* ptr = dispenso::alignedMalloc(1024, 64);  // 64-byte aligned
 * // ... use ptr ...
 * dispenso::alignedFree(ptr);
 * @endcode
 */
inline void* alignedMalloc(size_t bytes, size_t alignment) {
  return detail::alignedMalloc(bytes, alignment);
}

/**
 * @brief Allocate memory aligned to cache line size.
 *
 * This is a convenience overload that aligns to kCacheLineSize (typically 64 bytes),
 * which helps avoid false sharing in concurrent data structures.
 *
 * @param bytes The number of bytes to allocate
 * @return Pointer to cache-line aligned memory, or nullptr on allocation failure
 *
 * @note Memory allocated with alignedMalloc must be freed with alignedFree
 * @see alignedFree, kCacheLineSize
 */
inline void* alignedMalloc(size_t bytes) {
  return detail::alignedMalloc(bytes);
}

/**
 * @brief Free memory allocated by alignedMalloc.
 *
 * @param ptr Pointer to memory allocated by alignedMalloc (can be nullptr)
 *
 * @see alignedMalloc
 */
inline void alignedFree(void* ptr) {
  detail::alignedFree(ptr);
}

/**
 * @brief Deleter for smart pointers that use aligned memory allocation.
 *
 * This deleter calls the destructor and frees memory allocated with alignedMalloc.
 * It can be used with std::unique_ptr and std::shared_ptr.
 *
 * @tparam T The type being deleted
 *
 * Example:
 * @code
 * using AlignedPtr = std::unique_ptr<MyType, dispenso::AlignedDeleter<MyType>>;
 * void* mem = dispenso::alignedMalloc(sizeof(MyType), 64);
 * AlignedPtr ptr(new (mem) MyType(), dispenso::AlignedDeleter<MyType>());
 * @endcode
 */
template <typename T>
using AlignedDeleter = detail::AlignedFreeDeleter<T>;

/**
 * @brief Align a value up to the next cache line boundary.
 *
 * Rounds up the input value to the next multiple of kCacheLineSize. Useful for
 * manual memory layout to avoid false sharing.
 *
 * @param val The value to align
 * @return Value aligned to next cache line boundary
 *
 * @see kCacheLineSize
 *
 * Example:
 * @code
 * size_t offset = dispenso::alignToCacheLine(37);  // Returns 64
 * @endcode
 */
inline constexpr uintptr_t alignToCacheLine(uintptr_t val) {
  return detail::alignToCacheLine(val);
}

/**
 * @brief CPU relaxation hint for spin loops.
 *
 * Emits a platform-specific instruction (PAUSE on x86, YIELD on ARM) to improve
 * spin loop performance and reduce power consumption. Use this in busy-wait loops
 * to be friendlier to hyper-threading and the CPU pipeline.
 *
 * @note This is a no-op on platforms without a specific relax instruction
 *
 * Example:
 * @code
 * while (!flag.load(std::memory_order_acquire)) {
 *   dispenso::cpuRelax();  // Be nice to the CPU
 * }
 * @endcode
 */
inline void cpuRelax() {
  detail::cpuRelax();
}

/**
 * @brief Round up to the next power of 2.
 *
 * Computes the smallest power of 2 that is greater than or equal to the input value.
 *
 * @param v Input value
 * @return Next power of 2 (or v if v is already a power of 2)
 *
 * @note Returns 0 if v is 0
 *
 * Example:
 * @code
 * static_assert(dispenso::nextPow2(17) == 32);
 * static_assert(dispenso::nextPow2(64) == 64);
 * @endcode
 */
constexpr uint64_t nextPow2(uint64_t v) {
  return detail::nextPow2(v);
}

/**
 * @brief Compute log base 2 of a value (compile-time).
 *
 * Computes floor(log2(v)) at compile time. Useful for template metaprogramming
 * and constexpr contexts.
 *
 * @param v Input value (must be > 0)
 * @return floor(log2(v))
 *
 * @note Behavior is undefined if v is 0
 *
 * Example:
 * @code
 * static_assert(dispenso::log2const(64) == 6);
 * static_assert(dispenso::log2const(100) == 6);
 * @endcode
 */
constexpr inline uint32_t log2const(uint64_t v) {
  return detail::log2const(v);
}

/**
 * @brief Compute log base 2 of a value (runtime).
 *
 * Computes floor(log2(v)) using platform-specific intrinsics for optimal performance.
 * On x86/x64, uses __builtin_clzll or __lzcnt64. Falls back to constexpr version
 * on other platforms.
 *
 * @param v Input value (must be > 0)
 * @return floor(log2(v))
 *
 * @note Behavior is undefined if v is 0
 *
 * Example:
 * @code
 * uint32_t power = dispenso::log2(size);  // Fast bit scan
 * @endcode
 */
inline uint32_t log2(uint64_t v) {
  return detail::log2(v);
}

/**
 * @brief Buffer with proper alignment for type T.
 *
 * Provides uninitialized storage with proper alignment for type T. Useful for
 * manual object lifetime management or placement new scenarios.
 *
 * @tparam T The type to provide storage for
 *
 * Example:
 * @code
 * dispenso::AlignedBuffer<MyType> buf;
 * MyType* obj = new (buf.b) MyType();
 * obj->~MyType();
 * @endcode
 */
template <typename T>
using AlignedBuffer = detail::AlignedBuffer<T>;

/**
 * @brief Cache-line aligned atomic pointer.
 *
 * An atomic pointer aligned to cache line boundary to avoid false sharing.
 * Inherits from std::atomic<T*>.
 *
 * @tparam T The pointed-to type
 *
 * Example:
 * @code
 * dispenso::AlignedAtomic<int> ptr;
 * ptr.store(new int(42));
 * @endcode
 */
template <typename T>
using AlignedAtomic = detail::AlignedAtomic<T>;

/**
 * @brief Optional-like storage with in-place construction (C++14 compatible).
 *
 * Provides similar functionality to std::optional but works in C++14 codebases.
 * Stores the value in-place and tracks whether a value is present.
 *
 * @tparam T The type to store
 *
 * @note For small types, this has ~2x the size overhead of std::optional (pointer vs bool flag).
 *       For larger types, the overhead is similar. Prefer std::optional in C++17 and later.
 *
 * Example:
 * @code
 * dispenso::OpResult<int> result;
 * if (success) {
 *   result.emplace(42);
 * }
 * if (result) {
 *   use(result.value());
 * }
 * @endcode
 */
template <typename T>
using OpResult = detail::OpResult<T>;

/**
 * @brief Information for statically chunking a range across threads.
 *
 * When dividing work into static chunks, using a simple chunk size plus remainder
 * can lead to poor load balancing. This struct provides the optimal chunking strategy
 * where some tasks get ceil(items/chunks) work and others get floor(items/chunks).
 */
using StaticChunking = detail::StaticChunking;

/**
 * @brief Compute optimal static chunking for load balancing.
 *
 * Divides items into chunks such that the work is distributed as evenly as possible.
 * Returns chunking info where some tasks get ceil(items/chunks) and others get
 * floor(items/chunks).
 *
 * @param items Total number of items to process
 * @param chunks Number of chunks to divide into (must be > 0)
 * @return StaticChunking information for distributing the work
 *
 * Example:
 * @code
 * auto chunking = dispenso::staticChunkSize(100, 8);
 * // First 4 threads get 13 items each, last 4 get 12 items each
 * // 4*13 + 4*12 = 52 + 48 = 100
 * @endcode
 */
inline StaticChunking staticChunkSize(ssize_t items, ssize_t chunks) {
  return detail::staticChunkSize(items, chunks);
}

} // namespace dispenso
