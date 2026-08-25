/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file platform.h
 * @ingroup group_util
 * Platform constants and common utilities.
 **/

#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <thread>
#include <type_traits>

#if defined(_MSC_VER) && \
    (defined(_M_AMD64) || defined(_M_IX86) || defined(_M_ARM64) || defined(_M_ARM))
#include <intrin.h>
#endif

namespace dispenso {

#define DISPENSO_MAJOR_VERSION 1
#define DISPENSO_MINOR_VERSION 6
#define DISPENSO_PATCH_VERSION 2

// C++20 concepts support detection
#if __cplusplus >= 202002L && defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define DISPENSO_HAS_CONCEPTS 1
#include <concepts>
#else
#define DISPENSO_HAS_CONCEPTS 0
#endif

/**
 * @def DISPENSO_REQUIRES
 * @brief Macro for conditionally applying C++20 concept constraints.
 *
 * On C++20 with concepts support, this expands to a requires clause.
 * On C++14/17, this expands to nothing, maintaining backward compatibility.
 *
 * Example usage:
 * @code
 * template <typename F>
 * DISPENSO_REQUIRES(std::invocable<F>)
 * void schedule(F&& f);
 * @endcode
 **/
#if DISPENSO_HAS_CONCEPTS
#define DISPENSO_REQUIRES(...) requires(__VA_ARGS__)
#else
#define DISPENSO_REQUIRES(...)
#endif

/**
 * @def DISPENSO_DEPRECATED
 * @brief Macro that expands to <code>[[deprecated(msg)]]</code> on C++17+ and to nothing on C++14.
 *
 * Use on enumerators, functions, types, etc. to mark them as deprecated. The C++14 fallback is a
 * no-op so deprecation messages disappear silently on older toolchains — newer toolchains will see
 * the warning. (The standard attribute on enumerators only became valid in C++17, hence the
 * version gate.)
 *
 * Example: <code>kAuto DISPENSO_DEPRECATED("Use kAdaptive") = kAdaptive,</code>
 **/
#if __cplusplus >= 201703L
#define DISPENSO_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define DISPENSO_DEPRECATED(msg)
#endif

#if defined(DISPENSO_SHARED_LIB)
#if defined _WIN32

#if defined(DISPENSO_LIB_EXPORT)
#define DISPENSO_DLL_ACCESS __declspec(dllexport)
#else
#define DISPENSO_DLL_ACCESS __declspec(dllimport)
#endif // DISPENSO_LIB_EXPORT

#elif defined(__clang__) || defined(__GNUC__)
#define DISPENSO_DLL_ACCESS __attribute__((visibility("default")))
#endif // PLATFORM
#endif // DISPENSO_SHARED_LIB

#if !defined(DISPENSO_DLL_ACCESS)
#define DISPENSO_DLL_ACCESS
#endif // DISPENSO_DLL_ACCESS

// Suppresses Clang thread-safety-analysis warnings for a single function.
// Expands to the attribute on Clang; a no-op on all other compilers (MSVC, GCC, etc.)
// that do not support thread-safety analysis.
#if defined(__clang__)
#define DISPENSO_NO_THREAD_SAFETY_ANALYSIS __attribute__((no_thread_safety_analysis))
#else
#define DISPENSO_NO_THREAD_SAFETY_ANALYSIS
#endif

using ssize_t = std::make_signed<std::size_t>::type;

#if defined(__CUDACC__)
#define DISPENSO_INLINE __host__ __device__ __forceinline__
#elif defined(__clang__) || defined(__GNUC__)
#define DISPENSO_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define DISPENSO_INLINE __forceinline
#else
#define DISPENSO_INLINE inline
#endif // PLATFORM

/**
 * @var constexpr size_t kCacheLineSize
 * @brief A constant that defines a safe number of bytes+alignment to avoid false sharing.
 **/
#if defined(__APPLE__) && defined(__arm64__)
constexpr size_t kCacheLineSize = 128;
#else
constexpr size_t kCacheLineSize = 64;
#endif

/**
 * @def DISPENSO_CACHELINE_ALIGNED
 * @brief Cache-line alignment specifier for use on class/struct definitions.
 *
 * Expands to `alignas(kCacheLineSize)`. Prefer this over a bare
 * `class alignas(kCacheLineSize) Foo` on type definitions: cyclomatic-complexity
 * analyzers (lizard/abacus) misparse `class alignas(...) Foo` as a function named
 * `alignas` and fold the entire class body into one bogus high-CCN "function".
 * Routing through a macro hides the `alignas(...)` token so the type parses
 * normally. (Bare `alignas(kCacheLineSize)` on member variables is fine -- the
 * misparse only happens on type definitions.)
 **/
#define DISPENSO_CACHELINE_ALIGNED alignas(kCacheLineSize)

/**
 * @def DISPENSO_THREAD_LOCAL
 * @brief A macro that can be used when declaring a lightweight thread-local variable.
 **/

// TODO(bbudge): Non-gcc/clang/msvc platforms.
#if defined(_MSC_VER)
#define DISPENSO_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define DISPENSO_THREAD_LOCAL __thread
#else
#error Supply lightweight thread-locals for this compiler.  Can define to thread_local if lightweight not available
#endif

#if (defined(__GNUC__) || defined(__clang__))
#define DISPENSO_EXPECT(a, b) __builtin_expect(a, b)
#else
#define DISPENSO_EXPECT(a, b) a
#endif

// clang-format off
#if (defined(__GNUC__) || defined(__clang__))
#define DO_PRAGMA(X) _Pragma(#X)
#define DISPENSO_DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISPENSO_DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISPENSO_DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored #warningName)
#if !defined(__clang__)
#define DISPENSO_DISABLE_WARNING_ZERO_VARIADIC_MACRO_ARGUMENTS
#define DISPENSO_DISABLE_WARNING_GLOBAL_CONSTRUCTORS
#define DISPENSO_DISABLE_WARNING_FREE_NONHEAP_OBJECT \
  DISPENSO_DISABLE_WARNING(-Wfree-nonheap-object)
#else
#define DISPENSO_DISABLE_WARNING_ZERO_VARIADIC_MACRO_ARGUMENTS \
  DISPENSO_DISABLE_WARNING(-Wgnu-zero-variadic-macro-arguments)
#define DISPENSO_DISABLE_WARNING_GLOBAL_CONSTRUCTORS \
  DISPENSO_DISABLE_WARNING(-Wglobal-constructors)
#define DISPENSO_DISABLE_WARNING_FREE_NONHEAP_OBJECT
#endif
#elif defined(_MSC_VER)
#define DISPENSO_DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISPENSO_DISABLE_WARNING_POP __pragma(warning(pop))
#define DISPENSO_DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))
#define DISPENSO_DISABLE_WARNING_ZERO_VARIADIC_MACRO_ARGUMENTS
#define DISPENSO_DISABLE_WARNING_GLOBAL_CONSTRUCTORS
#define DISPENSO_DISABLE_WARNING_FREE_NONHEAP_OBJECT
#else
#define DISPENSO_DISABLE_WARNING_PUSH
#define DISPENSO_DISABLE_WARNING_POP
#define DISPENSO_DISABLE_WARNING_ZERO_VARIADIC_MACRO_ARGUMENTS
#define DISPENSO_DISABLE_WARNING_GLOBAL_CONSTRUCTORS
#define DISPENSO_DISABLE_WARNING_FREE_NONHEAP_OBJECT
#endif
// clang-format on

/**
 * A wrapper that aligns the contained value to cache line boundaries.
 *
 * Useful for avoiding false sharing in concurrent data structures.
 *
 * @tparam T The type to wrap with cache line alignment.
 */
template <typename T>
class CacheAligned {
 public:
  CacheAligned() = default;
  /** Construct from a value. @param t The value to wrap. */
  CacheAligned(T t) : t_(t) {}
  operator T&() {
    return t_;
  }

  operator const T&() const {
    return t_;
  }

 private:
  alignas(kCacheLineSize) T t_;
};

namespace detail {

template <typename T>
struct AlignedBuffer {
  alignas(alignof(T)) char b[sizeof(T)];
};

template <typename T>
struct alignas(kCacheLineSize) AlignedAtomic : public std::atomic<T*> {};

inline void* alignedMalloc(size_t bytes, size_t alignment) {
  alignment = std::max(alignment, sizeof(uintptr_t));
  char* ptr = reinterpret_cast<char*>(::malloc(bytes + alignment));
  uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t oldBase = base;
  uintptr_t mask = alignment - 1;
  base += alignment;
  base &= ~mask;

  uintptr_t* recovery = reinterpret_cast<uintptr_t*>(base - sizeof(uintptr_t));
  *recovery = oldBase;
  return reinterpret_cast<void*>(base);
}

inline void* alignedMalloc(size_t bytes) {
  return alignedMalloc(bytes, kCacheLineSize);
}

inline void alignedFree(void* ptr) {
  if (!ptr) {
    return;
  }
  char* p = reinterpret_cast<char*>(ptr);
  uintptr_t recovered = *reinterpret_cast<uintptr_t*>(p - sizeof(uintptr_t));
  ::free(reinterpret_cast<void*>(recovered));
}

template <typename T>
struct AlignedFreeDeleter {
  void operator()(T* ptr) {
    ptr->~T();
    detail::alignedFree(ptr);
  }
};
template <>
struct AlignedFreeDeleter<void> {
  void operator()(void* ptr) {
    detail::alignedFree(ptr);
  }
};

// Array deleter for aligned allocations. Destructor loop is elided by
// the compiler for trivially destructible types.
template <typename T>
struct AlignedArrayFreeDeleter {
  size_t count;
  void operator()(T* ptr) {
    for (size_t i = 0; i < count; ++i) {
      ptr[i].~T();
    }
    detail::alignedFree(ptr);
  }
};

// Allocate a value-initialized array of T with alignof(T) alignment.
// Constructor/destructor loops are elided for trivial types.
template <typename T>
std::unique_ptr<T[], AlignedArrayFreeDeleter<T>> makeAlignedArray(size_t n) {
  void* raw = detail::alignedMalloc(sizeof(T) * n, alignof(T));
  T* arr = static_cast<T*>(raw);
  for (size_t i = 0; i < n; ++i) {
    new (&arr[i]) T();
  }
  return std::unique_ptr<T[], AlignedArrayFreeDeleter<T>>(arr, AlignedArrayFreeDeleter<T>{n});
}

// Allocate a single object of T with alignof(T) alignment.
template <typename T, class... Args>
std::unique_ptr<T, AlignedFreeDeleter<T>> makeAligned(Args&&... args) {
  void* raw = detail::alignedMalloc(sizeof(T), alignof(T));
  T* obj = new (raw) T(std::forward<Args>(args)...);
  return std::unique_ptr<T, AlignedFreeDeleter<T>>(obj);
}

template <typename T, class... Args>
std::shared_ptr<T> make_shared(Args&&... args) {
  void* tv = alignedMalloc(sizeof(T), alignof(T));
  T* t = new (tv) T(std::forward<Args>(args)...);
  return std::shared_ptr<T>(t, AlignedFreeDeleter<T>());
}

inline constexpr uintptr_t alignToCacheLine(uintptr_t val) {
  constexpr uintptr_t kMask = kCacheLineSize - 1;
  val += kMask;
  val &= ~kMask;
  return val;
}

#if defined __x86_64__ || defined __i386__
inline void cpuRelax() {
  asm volatile("pause" ::: "memory");
}
#elif defined _MSC_VER && (defined _M_AMD64 || defined _M_IX86)
inline void cpuRelax() {
  _mm_pause();
}
#elif defined __arm64__ || defined __aarch64__
inline void cpuRelax() {
  asm volatile("yield" ::: "memory");
}
#elif defined _MSC_VER && (defined _M_ARM64 || defined _M_ARM)
inline void cpuRelax() {
  __yield();
}
#elif defined __powerpc__ || defined __POWERPC__
#if defined __APPLE__
inline void cpuRelax() {
  asm volatile("or r27,r27,r27" ::: "memory");
}
#else
inline void cpuRelax() {
  asm volatile("or 27,27,27" ::: "memory");
}
#endif // APPLE
#else
// TODO: provide reasonable relax on other archs.
inline void cpuRelax() {}
#endif // ARCH

// When statically chunking a range, it is generally not possible to use a single chunk size plus
// remainder and get a good load distribution.  By estimating too high, we can have idle threads. By
// estimating too low, the remainder can be several times as large as the chunk for other threads.
// Instead, we compute the chunk size that is the ceil of the fractional chunk size.  That can be
// used for the first transitionIndex values, while the remaining (chunks - transitionTaskIndex)
// values will be ceilChunkSize - 1.
struct StaticChunking {
  ssize_t transitionTaskIndex;
  ssize_t ceilChunkSize;
};

inline StaticChunking staticChunkSize(ssize_t items, ssize_t chunks) {
  assert(chunks > 0);
  StaticChunking chunking;
  chunking.ceilChunkSize = (items + chunks - 1) / chunks;
  ssize_t numLeft = chunking.ceilChunkSize * chunks - items;
  chunking.transitionTaskIndex = chunks - numLeft;
  return chunking;
}

// Granularity-aware variant: ceilChunkSize is rounded UP to a multiple of
// `granularity`, so each "ceil" chunk is granularity-aligned. The "floor"
// chunks (those at index >= transitionTaskIndex) are ceilChunkSize - granularity,
// also granularity-aligned. Caller must have already trimmed `items` to a
// multiple of `granularity` so that all chunks (not just intermediate ones)
// are granularity-multiples.
inline StaticChunking staticChunkSizeGranular(ssize_t items, ssize_t chunks, uint32_t granularity) {
  assert(chunks > 0);
  assert(granularity >= 1);
  if (granularity <= 1) {
    return staticChunkSize(items, chunks);
  }
  assert(items % static_cast<ssize_t>(granularity) == 0);
  StaticChunking chunking;
  // Items measured in "granularity units".
  ssize_t gUnits = items / static_cast<ssize_t>(granularity);
  ssize_t ceilG = (gUnits + chunks - 1) / chunks;
  ssize_t numLeft = ceilG * chunks - gUnits;
  chunking.ceilChunkSize = ceilG * static_cast<ssize_t>(granularity);
  chunking.transitionTaskIndex = chunks - numLeft;
  return chunking;
}

} // namespace detail
} // namespace dispenso
