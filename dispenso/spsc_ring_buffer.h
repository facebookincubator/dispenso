/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file spsc_ring_buffer.h
 * @ingroup group_util
 * A lock-free single-producer single-consumer (SPSC) ring buffer.
 *
 * This buffer is designed for high-performance communication between exactly one producer
 * thread and one consumer thread. It uses a fixed-capacity circular buffer with atomic
 * head and tail pointers, providing wait-free operations in the common case.
 *
 * @note This implementation is NOT thread-safe for multiple producers or multiple consumers.
 *       Use only with exactly one producer thread and one consumer thread.
 **/

#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include <dispenso/platform.h>
#include <dispenso/util.h>

namespace dispenso {

// TODO(bbudge): Add kDynamicCapacity specialization that heap-allocates storage
// with runtime-determined capacity.

/**
 * @class SPSCRingBuffer
 * @brief A lock-free single-producer single-consumer ring buffer with fixed capacity.
 *
 * This class implements a bounded, lock-free ring buffer optimized for the case where
 * there is exactly one producer thread and exactly one consumer thread. It uses relaxed
 * memory ordering where possible and acquire/release semantics only where necessary
 * for correctness.
 *
 * The buffer stores elements in a contiguous array, avoiding dynamic memory allocation
 * after construction. The capacity is fixed at compile time via a template parameter.
 *
 * Unlike std::array-based implementations, this buffer does NOT require the element type
 * to be default-constructible. Elements are constructed in-place when pushed and destroyed
 * when popped.
 *
 * @tparam T The type of elements stored in the buffer. Must be move-constructible.
 * @tparam Capacity The minimum number of elements the buffer can hold. Must be at least 1.
 *                  Defaults to 16, which provides good cache locality.
 * @tparam RoundUpToPowerOfTwo If true (default), rounds up the internal buffer size to the
 *                             next power of two for faster index wrap-around using bitwise
 *                             AND instead of modulo. This may result in actual capacity being
 *                             larger than requested. Set to false to use exactly the requested
 *                             capacity.
 *
 * ## Thread Safety
 *
 * This class is designed for exactly one producer thread and one consumer thread:
 * - Only one thread may call `try_push()` or `try_emplace()` at any time
 * - Only one thread may call `try_pop()` at any time
 * - The producer and consumer may be different threads
 * - `empty()`, `full()`, and `size()` may be called from any thread, but provide
 *   only a snapshot that may be immediately stale
 *
 * ## Memory Ordering
 *
 * The implementation uses:
 * - `memory_order_relaxed` for local reads of head/tail
 * - `memory_order_acquire` when reading the "other" index (consumer reads head, producer
 * reads tail)
 * - `memory_order_release` when updating head/tail after successful push/pop
 *
 * ## Performance Characteristics
 *
 * - Push: O(1), wait-free when buffer is not full
 * - Pop: O(1), wait-free when buffer is not empty
 * - Memory: sizeof(T) * (actual capacity + 1) + 2 cache lines for head/tail indices
 * - When RoundUpToPowerOfTwo is true (default), index wrap-around uses fast bitwise AND
 *
 * ## Example Usage
 *
 * @code
 * // Default: rounds up to power of two for performance
 * dispenso::SPSCRingBuffer<int, 100> buffer;  // actual capacity is 127 (128 - 1)
 *
 * // Exact capacity mode (no rounding)
 * dispenso::SPSCRingBuffer<int, 100, false> exact;  // actual capacity is 100
 *
 * // Producer thread
 * if (buffer.try_push(42)) {
 *     // Success
 * }
 *
 * // Consumer thread
 * int value;
 * if (buffer.try_pop(value)) {
 *     // Use value
 * }
 * @endcode
 *
 * @see https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
 *      for background on lock-free queue algorithms
 */
template <typename T, size_t Capacity = 16, bool RoundUpToPowerOfTwo = true>
class SPSCRingBuffer {
  static_assert(Capacity >= 1, "SPSCRingBuffer capacity must be at least 1");
  static_assert(
      std::is_move_constructible<T>::value,
      "SPSCRingBuffer element type must be move-constructible");

 public:
  /**
   * @brief The type of elements stored in this buffer.
   */
  using value_type = T;

  /**
   * @brief The size type used for indices and counts.
   */
  using size_type = size_t;

  /**
   * @brief Constructs an empty ring buffer.
   *
   * The buffer is initialized with no elements. The internal storage is
   * uninitialized - elements are only constructed when pushed.
   *
   * @note Construction is not thread-safe. Ensure the buffer is fully
   *       constructed before any thread accesses it.
   */
  SPSCRingBuffer() = default;

  /**
   * @brief Ring buffers are not copyable.
   *
   * Copying a concurrent data structure would require synchronization
   * and could lead to subtle bugs. Use move semantics if you need to
   * transfer ownership.
   */
  SPSCRingBuffer(const SPSCRingBuffer&) = delete;

  /**
   * @brief Ring buffers are not copy-assignable.
   * @see SPSCRingBuffer(const SPSCRingBuffer&)
   */
  SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

  /**
   * @brief Ring buffers are not movable.
   *
   * Moving a ring buffer while producer/consumer threads are active
   * would be unsafe. If you need to transfer ownership, ensure all
   * threads have stopped first.
   */
  SPSCRingBuffer(SPSCRingBuffer&&) = delete;

  /**
   * @brief Ring buffers are not move-assignable.
   * @see SPSCRingBuffer(SPSCRingBuffer&&)
   */
  SPSCRingBuffer& operator=(SPSCRingBuffer&&) = delete;

  /**
   * @brief Destroys the ring buffer.
   *
   * All elements remaining in the buffer are destroyed. Ensure no
   * producer or consumer threads are accessing the buffer when it
   * is destroyed.
   *
   * @note Destruction is not thread-safe.
   */
  ~SPSCRingBuffer() {
    // Destroy any remaining elements
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_relaxed);
    while (head != tail) {
      elementAt(head)->~T();
      head = increment(head);
    }
  }

  /**
   * @brief Attempts to push an element into the buffer by moving.
   *
   * If the buffer has space, the element is moved into the buffer and
   * the function returns true. If the buffer is full, the function
   * returns false and the element is unchanged.
   *
   * @param item The element to push (will be moved from on success).
   * @return true if the element was successfully pushed, false if the buffer was full.
   *
   * @note Only one thread may call this function at any time (single producer).
   * @note This operation is wait-free.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<std::string, 4> buffer;
   * std::string msg = "hello";
   * if (buffer.try_push(std::move(msg))) {
   *     // msg is now empty (moved from)
   * } else {
   *     // msg is unchanged, buffer was full
   * }
   * @endcode
   */
  bool try_push(T&& item) {
    const size_t currentTail = tail_.load(std::memory_order_relaxed);
    const size_t nextTail = increment(currentTail);

    // Check if buffer is full
    if (nextTail == head_.load(std::memory_order_acquire)) {
      return false;
    }

    // Construct element in-place
    new (elementAt(currentTail)) T(std::move(item));
    tail_.store(nextTail, std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to push an element into the buffer by copying.
   *
   * If the buffer has space, the element is copied into the buffer and
   * the function returns true. If the buffer is full, the function
   * returns false.
   *
   * @param item The element to push (will be copied).
   * @return true if the element was successfully pushed, false if the buffer was full.
   *
   * @note Only one thread may call this function at any time (single producer).
   * @note This operation is wait-free.
   * @note Prefer try_push(T&&) when the source element is no longer needed.
   */
  bool try_push(const T& item) {
    const size_t currentTail = tail_.load(std::memory_order_relaxed);
    const size_t nextTail = increment(currentTail);

    // Check if buffer is full
    if (nextTail == head_.load(std::memory_order_acquire)) {
      return false;
    }

    // Construct element in-place via copy
    new (elementAt(currentTail)) T(item);
    tail_.store(nextTail, std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to construct an element in-place in the buffer.
   *
   * If the buffer has space, constructs an element directly in the buffer
   * storage using the provided arguments, avoiding any copy or move operations.
   *
   * @tparam Args The types of arguments to forward to T's constructor.
   * @param args The arguments to forward to the element constructor.
   * @return true if the element was successfully emplaced, false if the buffer was full.
   *
   * @note Only one thread may call this function at any time (single producer).
   * @note This operation is wait-free.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<std::pair<int, std::string>, 4> buffer;
   * if (buffer.try_emplace(42, "hello")) {
   *     // Element constructed in-place
   * }
   * @endcode
   */
  template <typename... Args>
  bool try_emplace(Args&&... args) {
    const size_t currentTail = tail_.load(std::memory_order_relaxed);
    const size_t nextTail = increment(currentTail);

    // Check if buffer is full
    if (nextTail == head_.load(std::memory_order_acquire)) {
      return false;
    }

    // Construct element in-place
    new (elementAt(currentTail)) T(std::forward<Args>(args)...);
    tail_.store(nextTail, std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to pop an element from the buffer.
   *
   * If the buffer has elements, moves the front element into the output
   * parameter and returns true. If the buffer is empty, returns false
   * and leaves the output parameter unchanged.
   *
   * @param[out] item The location to move the popped element to.
   * @return true if an element was successfully popped, false if the buffer was empty.
   *
   * @note Only one thread may call this function at any time (single consumer).
   * @note This operation is wait-free.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<int, 4> buffer;
   * buffer.try_push(42);
   *
   * int value;
   * if (buffer.try_pop(value)) {
   *     assert(value == 42);
   * }
   * @endcode
   */
  bool try_pop(T& item) {
    const size_t currentHead = head_.load(std::memory_order_relaxed);

    // Check if buffer is empty
    if (currentHead == tail_.load(std::memory_order_acquire)) {
      return false;
    }

    T* elem = elementAt(currentHead);
    item = std::move(*elem);
    elem->~T();
    head_.store(increment(currentHead), std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to pop an element from the buffer, returning an optional.
   *
   * If the buffer has elements, moves the front element into an OpResult
   * and returns it. If the buffer is empty, returns an empty OpResult.
   *
   * This provides a cleaner API than try_pop(T&) when default-constructibility
   * of T is not guaranteed or when a more functional style is preferred.
   *
   * @return An OpResult containing the popped element, or an empty OpResult
   *         if the buffer was empty.
   *
   * @note Only one thread may call this function at any time (single consumer).
   * @note This operation is wait-free.
   * @note In C++17 and beyond, you may prefer to use std::optional directly.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<std::string, 4> buffer;
   * buffer.try_push("hello");
   *
   * if (auto result = buffer.try_pop()) {
   *     std::cout << result.value() << std::endl;
   * }
   * @endcode
   */
  OpResult<T> try_pop() {
    const size_t currentHead = head_.load(std::memory_order_relaxed);

    // Check if buffer is empty
    if (currentHead == tail_.load(std::memory_order_acquire)) {
      return {};
    }

    T* elem = elementAt(currentHead);
    OpResult<T> result(std::move(*elem));
    elem->~T();
    head_.store(increment(currentHead), std::memory_order_release);
    return result;
  }

  /**
   * @brief Attempts to pop an element into uninitialized storage.
   *
   * Similar to try_pop, but uses placement new to construct the element
   * into the provided storage. This is useful when T is not default-constructible.
   *
   * @param[out] storage Pointer to uninitialized storage where the element will be
   *                     move-constructed. Must have proper alignment for T.
   * @return true if an element was successfully popped, false if the buffer was empty.
   *
   * @note Only one thread may call this function at any time (single consumer).
   * @note The caller is responsible for eventually destroying the constructed object.
   * @note This operation is wait-free.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<NonDefaultConstructible, 4> buffer;
   * alignas(NonDefaultConstructible) char storage[sizeof(NonDefaultConstructible)];
   * if (buffer.try_pop_into(reinterpret_cast<NonDefaultConstructible*>(storage))) {
   *     auto* ptr = reinterpret_cast<NonDefaultConstructible*>(storage);
   *     // use *ptr
   *     ptr->~NonDefaultConstructible();
   * }
   * @endcode
   */
  bool try_pop_into(T* storage) {
    const size_t currentHead = head_.load(std::memory_order_relaxed);

    // Check if buffer is empty
    if (currentHead == tail_.load(std::memory_order_acquire)) {
      return false;
    }

    T* elem = elementAt(currentHead);
    new (storage) T(std::move(*elem));
    elem->~T();
    head_.store(increment(currentHead), std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to push multiple elements into the buffer.
   *
   * Pushes as many elements as possible from the range [first, last) into the buffer
   * in a single atomic tail update. This reduces atomic operation overhead when
   * pushing multiple items.
   *
   * @tparam InputIt Input iterator type. Must dereference to a type convertible to T.
   * @param first Iterator to the first element to push.
   * @param last Iterator past the last element to push.
   * @return The number of elements successfully pushed. May be less than the
   *         range size if the buffer becomes full.
   *
   * @note Only one thread may call this function at any time (single producer).
   * @note Elements are moved from the input range.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<int, 8> buffer;
   * std::vector<int> items = {1, 2, 3, 4, 5};
   * size_t pushed = buffer.try_push_batch(items.begin(), items.end());
   * // pushed contains the number of items successfully added
   * @endcode
   */
  template <typename InputIt>
  size_type try_push_batch(InputIt first, InputIt last) {
    const size_t currentTail = tail_.load(std::memory_order_relaxed);
    const size_t currentHead = head_.load(std::memory_order_acquire);

    // Calculate available space (actual capacity is kBufferSize - 1)
    size_t available;
    if (currentTail >= currentHead) {
      // Tail is ahead of or at head: available = capacity - (tail - head)
      available = (kBufferSize - 1) - (currentTail - currentHead);
    } else {
      // Tail has wrapped: available = head - tail - 1
      available = currentHead - currentTail - 1;
    }

    if (available == 0) {
      return 0;
    }

    // Push as many items as possible
    size_t count = 0;
    size_t tailPos = currentTail;
    for (; first != last && count < available; ++first, ++count) {
      new (elementAt(tailPos)) T(std::move(*first));
      tailPos = increment(tailPos);
    }

    if (count > 0) {
      tail_.store(tailPos, std::memory_order_release);
    }
    return count;
  }

  /**
   * @brief Attempts to pop multiple elements from the buffer.
   *
   * Pops up to maxCount elements from the buffer into the output iterator
   * in a single atomic head update. This reduces atomic operation overhead
   * when consuming multiple items.
   *
   * @tparam OutputIt Output iterator type. Must be assignable from T.
   * @param dest Output iterator to write popped elements to.
   * @param maxCount Maximum number of elements to pop.
   * @return The number of elements successfully popped. May be less than
   *         maxCount if the buffer has fewer elements.
   *
   * @note Only one thread may call this function at any time (single consumer).
   * @note Elements are moved to the output range.
   *
   * ## Example
   * @code
   * SPSCRingBuffer<int, 8> buffer;
   * // ... push some items ...
   * std::vector<int> items(4);
   * size_t popped = buffer.try_pop_batch(items.begin(), 4);
   * // First 'popped' elements of items contain the popped values
   * @endcode
   */
  template <typename OutputIt>
  size_type try_pop_batch(OutputIt dest, size_type maxCount) {
    const size_t currentHead = head_.load(std::memory_order_relaxed);
    const size_t currentTail = tail_.load(std::memory_order_acquire);

    // Calculate available items
    size_t available;
    if (currentTail >= currentHead) {
      available = currentTail - currentHead;
    } else {
      available = kBufferSize - currentHead + currentTail;
    }

    if (available == 0) {
      return 0;
    }

    // Pop as many items as requested and available
    size_t count = std::min(available, maxCount);
    size_t headPos = currentHead;
    for (size_t i = 0; i < count; ++i, ++dest) {
      T* elem = elementAt(headPos);
      *dest = std::move(*elem);
      elem->~T();
      headPos = increment(headPos);
    }

    if (count > 0) {
      head_.store(headPos, std::memory_order_release);
    }
    return count;
  }

  /**
   * @brief Checks if the buffer is empty.
   *
   * @return true if the buffer contains no elements, false otherwise.
   *
   * @note This function provides only a snapshot of the buffer state.
   *       The result may be stale by the time it is used, as another
   *       thread may have pushed or popped an element.
   *
   * @note Safe to call from any thread, but the result is only a hint.
   */
  bool empty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  /**
   * @brief Checks if the buffer is full.
   *
   * @return true if the buffer has no space for additional elements, false otherwise.
   *
   * @note This function provides only a snapshot of the buffer state.
   *       The result may be stale by the time it is used, as another
   *       thread may have popped an element.
   *
   * @note Safe to call from any thread, but the result is only a hint.
   */
  bool full() const {
    return increment(tail_.load(std::memory_order_acquire)) ==
        head_.load(std::memory_order_acquire);
  }

  /**
   * @brief Returns the current number of elements in the buffer.
   *
   * @return The number of elements currently in the buffer.
   *
   * @note This function provides only a snapshot of the buffer state.
   *       The result may be stale by the time it is used.
   *
   * @note Safe to call from any thread, but the result is only a hint.
   *
   * @note The implementation handles wrap-around correctly using modular
   *       arithmetic.
   */
  size_type size() const {
    const size_t head = head_.load(std::memory_order_acquire);
    const size_t tail = tail_.load(std::memory_order_acquire);
    // Handle wrap-around: if tail < head, we've wrapped
    return (tail >= head) ? (tail - head) : (kBufferSize - head + tail);
  }

  /**
   * @brief Returns the maximum number of elements the buffer can hold.
   *
   * When RoundUpToPowerOfTwo is true (default), this may be larger than the
   * requested Capacity template parameter, as the internal buffer is rounded
   * up to the next power of two for performance.
   *
   * @return The actual maximum capacity of the buffer (kBufferSize - 1).
   *
   * @note The actual storage uses capacity() + 1 slots internally to distinguish
   *       between empty and full states.
   */
  static constexpr size_type capacity() noexcept {
    return kBufferSize - 1;
  }

 private:
  /**
   * @brief Computes the internal buffer size at compile time.
   *
   * When RoundUpToPowerOfTwo is true, rounds (Capacity + 1) up to the next power of two.
   * Otherwise, uses exactly (Capacity + 1).
   */
  static constexpr size_t computeBufferSize() noexcept {
    if constexpr (RoundUpToPowerOfTwo) {
      return static_cast<size_t>(detail::nextPow2(Capacity + 1));
    } else {
      return Capacity + 1;
    }
  }

  /**
   * @brief Internal buffer size (includes one extra slot to distinguish empty from full).
   *
   * The ring buffer uses one slot as a sentinel to distinguish between the
   * empty state (head == tail) and the full state (next(tail) == head).
   * When RoundUpToPowerOfTwo is true, this is rounded up to a power of two.
   */
  static constexpr size_t kBufferSize = computeBufferSize();

  /**
   * @brief Compile-time check if kBufferSize is a power of two.
   *
   * When true, we can use faster bitwise AND instead of modulo for wrap-around.
   */
  static constexpr bool kIsPowerOfTwo = (kBufferSize & (kBufferSize - 1)) == 0;

  /**
   * @brief Mask for power-of-two wrap-around (kBufferSize - 1).
   *
   * Only valid when kIsPowerOfTwo is true.
   */
  static constexpr size_t kMask = kBufferSize - 1;

  /**
   * @brief Increments an index with wrap-around.
   *
   * Uses bitwise AND when buffer size is a power of two (faster),
   * otherwise falls back to modulo.
   *
   * @param index The current index.
   * @return The next index, wrapping to 0 if necessary.
   */
  static constexpr size_t increment(size_t index) noexcept {
    if constexpr (kIsPowerOfTwo) {
      return (index + 1) & kMask;
    } else {
      return (index + 1) % kBufferSize;
    }
  }

  /**
   * @brief Returns a pointer to the element at the given index.
   */
  T* elementAt(size_t index) {
    return reinterpret_cast<T*>(&storage_[index * sizeof(T)]);
  }

  const T* elementAt(size_t index) const {
    return reinterpret_cast<const T*>(&storage_[index * sizeof(T)]);
  }

  /// Head index (consumer reads from here). Cache-line aligned to avoid false sharing.
  alignas(kCacheLineSize) std::atomic<size_t> head_{0};

  /// Tail index (producer writes here). Cache-line aligned to avoid false sharing.
  alignas(kCacheLineSize) std::atomic<size_t> tail_{0};

  /// Uninitialized storage for elements. Uses Capacity + 1 slots to distinguish empty from full.
  /// Elements are constructed via placement new when pushed and explicitly destroyed when popped.
  alignas(T) char storage_[sizeof(T) * kBufferSize];
};

} // namespace dispenso
