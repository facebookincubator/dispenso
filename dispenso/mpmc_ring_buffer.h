/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file mpmc_ring_buffer.h
 * @ingroup group_util
 * A lock-free multi-producer multi-consumer (MPMC) bounded ring buffer.
 *
 * This buffer implements the Vyukov bounded MPMC queue algorithm, which uses per-slot sequence
 * numbers for synchronization. It provides O(1) fail-fast push and pop operations with no CAS
 * retry loops on the uncontended fast path.
 *
 * The primary use case is per-thread work queues for fork-join scheduling, where a scheduler
 * pushes targeted work to a specific thread's ring and the thread (or a work-stealing neighbor)
 * pops from it. Bulk push is supported for efficient batch scheduling.
 *
 * @note This implementation is thread-safe for multiple producers AND multiple consumers.
 *
 * @see docs/development/architecture/three_tier_scheduling.md for the design context.
 * @see https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
 **/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#include <dispenso/platform.h>
#include <dispenso/util.h>

namespace dispenso {

/**
 * @class MpmcRingBuffer
 * @brief A lock-free multi-producer multi-consumer ring buffer with fixed capacity.
 *
 * This class implements a bounded, lock-free ring buffer that supports concurrent access from
 * multiple producer threads and multiple consumer threads simultaneously. It uses the Vyukov
 * bounded MPMC queue algorithm with per-slot sequence numbers.
 *
 * Each slot contains a sequence number that tracks its state:
 * - When `seq == pos`: the slot is available for writing (empty)
 * - When `seq == pos + 1`: the slot contains data and is available for reading
 * - Other values: the slot is being written to or read from by another thread
 *
 * Unlike std::array-based implementations, this buffer does NOT require the element type
 * to be default-constructible. Elements are constructed in-place when pushed and destroyed
 * when popped.
 *
 * @tparam T The type of elements stored in the buffer. Must be move-constructible.
 * @tparam Capacity The number of elements the buffer can hold. Must be at least 2.
 *                  Defaults to 16, which provides a good balance of capacity and cache
 *                  footprint for per-thread work queues.
 * @tparam RoundUpToPowerOfTwo If true (default), rounds up the internal buffer size to the
 *                             next power of two for faster index wrap-around using bitwise
 *                             AND instead of modulo. This may result in actual capacity being
 *                             larger than requested. Set to false to use exactly the requested
 *                             capacity.
 *
 * ## Thread Safety
 *
 * This class is fully thread-safe:
 * - Multiple threads may call `try_push()`, `try_emplace()`, or `try_push_batch()` concurrently
 * - Multiple threads may call `try_pop()` concurrently
 * - `empty()`, `full()`, and `size()` may be called from any thread, but provide
 *   only a snapshot that may be immediately stale
 *
 * ## Memory Ordering
 *
 * The implementation uses:
 * - `memory_order_relaxed` for loading head/tail in the fast path
 * - `memory_order_acquire` for reading per-slot sequence numbers (ensures data visibility)
 * - `memory_order_release` for writing per-slot sequence numbers (publishes data)
 * - `compare_exchange_strong` on head/tail to claim a position
 *
 * ## Correctness & ABA-freedom
 *
 * `head_` and `tail_` are **monotonically increasing 64-bit counters** — never reset, and never
 * wrapping in any realistic runtime (2^64 push/pop operations). The slot index is `counter & mask`,
 * but the CAS always targets the full counter, never the wrapped index. Because a monotonic counter
 * can never present the same value twice, the classic ABA problem cannot arise on `head_`/`tail_`:
 * a successful CAS proves that no other producer (resp. consumer) advanced the counter between our
 * relaxed load and the CAS, so the position we just claimed is exclusively ours. A full lap around
 * the ring advances the counter by `capacity`, so it cannot alias an earlier value.
 *
 * The per-slot sequence numbers carry the other half of the Vyukov invariant and make each
 * (slot, lap) pair distinguishable: a slot accepts a write at position `pos` only when `seq ==
 * pos`, and yields a read only when `seq == pos + 1`. A producer publishes with `seq = pos + 1`; a
 * consumer releases with `seq = pos + capacity` (exactly the next counter value that maps to this
 * slot). So a stale observer can never mistake one lap's slot state for another's, and the value
 * written into `seq` is itself monotonic per slot — there is no recycled sentinel to confuse a CAS.
 *
 * **Deviation from canonical Vyukov:** the single-element operations are *fail-fast* — one
 * `compare_exchange_strong`, no retry loop. The canonical algorithm reloads the counter and loops
 * on CAS failure. Here, contention (a peer claimed the same position first) simply returns
 * false/empty even when the buffer is not actually full/empty. This is a deliberate
 * spurious-failure trade-off for bounded fail-fast scheduling (callers retry or fall back to
 * another tier; see fork_join_scheduling.md). It never corrupts state, double-uses a slot, or loses
 * an element — the monotonic-counter + per-slot-sequence invariants above hold regardless of how
 * the single CAS resolves. (`try_push_batch` claims a contiguous run `[tail, tail+available)` with
 * one CAS; the same monotonic-counter argument makes that whole reserved run exclusively the
 * caller's.)
 *
 * ## Memory Layout
 *
 * Each slot is padded up to a multiple of the cache line size to avoid false sharing between
 * adjacent slots when multiple threads operate on neighboring elements. The element buffer is
 * placed *before* the sequence number so that any alignment padding `T` requires does not land
 * between `seq` and `data` -- such interior padding could otherwise push a slot that would have
 * fit in a single cache line over the boundary:
 *
 * ```
 * Slot 0: [data (T) | seq (atomic) | padding to cache-line multiple]
 * Slot 1: [data (T) | seq (atomic) | padding to cache-line multiple]
 * ...
 * ```
 *
 * The per-slot size therefore depends on `sizeof(T)` (and `alignof(T)`), not just the cache line
 * size. A slot fits in a single cache line only when `sizeof(T) + sizeof(std::atomic<size_t>)`
 * does -- roughly `sizeof(T) <= 56` for a 64-byte cache line (or `<= 120` on platforms with a
 * 128-byte cache line, such as Apple ARM). Larger elements round each slot up to the next cache-
 * line multiple; e.g. a 64-byte element yields 128-byte slots.
 *
 * With a small `T`, 16 slots x 64 bytes = 1 KB per ring -- compact enough for per-thread
 * allocation even at high thread counts (256 threads = 256 KB total). For larger elements, scale
 * this figure by the actual per-slot size (`sizeof(Slot)`).
 *
 * ## Performance Characteristics
 *
 * - Push: O(1), fail-fast when full (no blocking, no retry loop)
 * - Pop: O(1), fail-fast when empty (no blocking, no retry loop)
 * - Bulk push: O(K) with single CAS for K items
 * - Memory: sizeof(Slot) * capacity + 2 cache lines for head/tail, where sizeof(Slot) is
 *   sizeof(T) plus the sequence number rounded up to a cache-line multiple (one cache line for
 *   small T, more for large T)
 *
 * ## Example Usage
 *
 * @code
 * dispenso::MpmcRingBuffer<int, 16> ring;
 *
 * // Producer thread(s)
 * if (ring.try_push(42)) {
 *     // Success
 * }
 *
 * // Consumer thread(s)
 * int value;
 * if (ring.try_pop(value)) {
 *     // Use value
 * }
 *
 * // Bulk push (returns number actually pushed)
 * std::array<int, 4> items = {1, 2, 3, 4};
 * size_t pushed = ring.try_push_batch(items.data(), items.size());
 * // pushed <= 4; caller handles overflow for remaining items
 * @endcode
 */
template <typename T, size_t Capacity = 16, bool RoundUpToPowerOfTwo = true>
#if DISPENSO_HAS_CONCEPTS
  requires std::move_constructible<T> && std::is_nothrow_move_constructible_v<T>
#endif
class MpmcRingBuffer {
  static_assert(Capacity >= 2, "MpmcRingBuffer capacity must be at least 2");
#if !DISPENSO_HAS_CONCEPTS
  static_assert(
      std::is_move_constructible<T>::value,
      "MpmcRingBuffer element type must be move-constructible");
  static_assert(
      std::is_nothrow_move_constructible<T>::value,
      "MpmcRingBuffer element type must be nothrow-move-constructible");
#endif

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
   * Initializes all slot sequence numbers to their position index, indicating
   * that all slots are available for writing.
   *
   * @note Construction is not thread-safe. Ensure the buffer is fully
   *       constructed before any thread accesses it.
   */
  MpmcRingBuffer() {
    for (size_t i = 0; i < kBufferSize; ++i) {
      slots_[i].seq.store(i, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Ring buffers are not copyable.
   */
  MpmcRingBuffer(const MpmcRingBuffer&) = delete;

  /**
   * @brief Ring buffers are not copy-assignable.
   */
  MpmcRingBuffer& operator=(const MpmcRingBuffer&) = delete;

  /**
   * @brief Ring buffers are not movable.
   */
  MpmcRingBuffer(MpmcRingBuffer&&) = delete;

  /**
   * @brief Ring buffers are not move-assignable.
   */
  MpmcRingBuffer& operator=(MpmcRingBuffer&&) = delete;

  /**
   * @brief Destroys the ring buffer.
   *
   * All elements remaining in the buffer are destroyed. Ensure no
   * threads are accessing the buffer when it is destroyed.
   *
   * @note Destruction is not thread-safe.
   */
  ~MpmcRingBuffer() {
    // Drain and destroy any remaining elements.
    // At destruction time there must be no concurrent access, so we can
    // read head/tail relaxed and walk forward, destroying elements whose
    // sequence numbers indicate they contain data.
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_relaxed);
    while (head != tail) {
      size_t pos = wrapIndex(head);
      dataPtr(slots_[pos])->~T();
      ++head;
    }
  }

  /**
   * @brief Attempts to push an element into the buffer by moving.
   *
   * If the buffer has space, the element is moved into the buffer and
   * the function returns true. If the buffer is full or another thread
   * is contending for the same slot, returns false immediately.
   *
   * @param item The element to push (will be moved from on success).
   * @return true if the element was successfully pushed, false if the buffer
   *         was full or contended.
   *
   * @note This operation is lock-free and fail-fast (no retry loop).
   *
   * ## Example
   * @code
   * MpmcRingBuffer<std::string, 4> buffer;
   * std::string msg = "hello";
   * if (buffer.try_push(std::move(msg))) {
   *     // msg is now empty (moved from)
   * }
   * @endcode
   */
  bool try_push(T&& item) {
    return emplaceImpl(std::move(item));
  }

  /**
   * @brief Attempts to push an element into the buffer by copying.
   *
   * @param item The element to push (will be copied).
   * @return true if the element was successfully pushed, false if the buffer
   *         was full or contended.
   *
   * @note This operation is lock-free and fail-fast (no retry loop).
   * @note Prefer try_push(T&&) when the source element is no longer needed.
   */
#if DISPENSO_HAS_CONCEPTS
  bool try_push(const T& item)
    requires std::is_nothrow_copy_constructible_v<T>
  {
    return emplaceImpl(item);
  }
#else
  template <typename U = T, std::enable_if_t<std::is_nothrow_copy_constructible<U>::value, int> = 0>
  bool try_push(const T& item) {
    return emplaceImpl(item);
  }
#endif

  /**
   * @brief Attempts to construct an element in-place in the buffer.
   *
   * If the buffer has space, constructs an element directly in the buffer
   * storage using the provided arguments, avoiding any copy or move operations.
   *
   * @tparam Args The types of arguments to forward to T's constructor.
   * @param args The arguments to forward to the element constructor.
   * @return true if the element was successfully emplaced, false if the buffer
   *         was full or contended.
   *
   * @note This operation is lock-free and fail-fast (no retry loop).
   *
   * ## Example
   * @code
   * MpmcRingBuffer<std::pair<int, std::string>, 4> buffer;
   * if (buffer.try_emplace(42, "hello")) {
   *     // Element constructed in-place
   * }
   * @endcode
   */
#if DISPENSO_HAS_CONCEPTS
  template <typename... Args>
    requires std::is_nothrow_constructible_v<T, Args...>
  bool try_emplace(Args&&... args) {
    return emplaceImpl(std::forward<Args>(args)...);
  }
#else
  template <
      typename... Args,
      std::enable_if_t<std::is_nothrow_constructible<T, Args...>::value, int> = 0>
  bool try_emplace(Args&&... args) {
    return emplaceImpl(std::forward<Args>(args)...);
  }
#endif

  /**
   * @brief Attempts to pop an element from the buffer.
   *
   * If the buffer has elements, moves the front element into the output
   * parameter and returns true. If the buffer is empty or another thread
   * is contending for the same slot, returns false immediately.
   *
   * @param[out] item The location to move the popped element to.
   * @return true if an element was successfully popped, false if the buffer
   *         was empty or contended.
   *
   * @note This operation is lock-free and fail-fast (no retry loop).
   *
   * ## Example
   * @code
   * MpmcRingBuffer<int, 4> buffer;
   * buffer.try_push(42);
   *
   * int value;
   * if (buffer.try_pop(value)) {
   *     assert(value == 42);
   * }
   * @endcode
   */
  bool try_pop(T& item) {
    // This overload move-*assigns* into the caller's object. If that throws after
    // the head_ CAS below, the slot would be left unreleased (seq stuck at
    // head+1) and its element leaked, since the destructor only walks [head,
    // tail). Require nothrow move-assignment so the slot-release path can never
    // be derailed. Types that are only nothrow-move-constructible can still use
    // try_pop() (OpResult) or try_pop_into(), which move-construct.
    static_assert(
        std::is_nothrow_move_assignable<T>::value,
        "MpmcRingBuffer::try_pop(T&) requires a nothrow-move-assignable T; "
        "use try_pop() or try_pop_into() for nothrow-move-constructible-only types");
    size_t head = head_.load(std::memory_order_relaxed);
    // Fast empty-check: a relaxed tail load is much cheaper than the acquire
    // slot.seq load below (esp. on weak-memory architectures: ldr vs ldar).
    // Callers that poll many sources for work hit this path constantly.
    if (head == tail_.load(std::memory_order_relaxed)) {
      return false;
    }
    Slot& slot = slots_[wrapIndex(head)];
    size_t seq = slot.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head + 1);
    if (diff == 0) {
      if (head_.compare_exchange_strong(head, head + 1, std::memory_order_relaxed)) {
        T* elem = dataPtr(slot);
        item = std::move(*elem);
        elem->~T();
        slot.seq.store(head + kBufferSize, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Attempts to pop an element from the buffer, returning an optional.
   *
   * If the buffer has elements, moves the front element into an OpResult
   * and returns it. If the buffer is empty, returns an empty OpResult.
   *
   * @return An OpResult containing the popped element, or an empty OpResult
   *         if the buffer was empty or contended.
   *
   * @note This operation is lock-free and fail-fast (no retry loop).
   *
   * ## Example
   * @code
   * MpmcRingBuffer<std::string, 4> buffer;
   * buffer.try_push(std::string("hello"));
   *
   * if (auto result = buffer.try_pop()) {
   *     std::cout << result.value() << std::endl;
   * }
   * @endcode
   */
  OpResult<T> try_pop() {
    size_t head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_relaxed)) {
      return {};
    }
    Slot& slot = slots_[wrapIndex(head)];
    size_t seq = slot.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head + 1);
    if (diff == 0) {
      if (head_.compare_exchange_strong(head, head + 1, std::memory_order_relaxed)) {
        T* elem = dataPtr(slot);
        OpResult<T> result(std::move(*elem));
        elem->~T();
        slot.seq.store(head + kBufferSize, std::memory_order_release);
        return result;
      }
    }
    return {};
  }

  /**
   * @brief Attempts to pop an element into uninitialized storage.
   *
   * Similar to try_pop(T&), but uses placement new to move-construct the
   * element into the provided storage. This is useful when T is not
   * default-constructible.
   *
   * @param[out] storage Pointer to uninitialized storage where the element will be
   *                     move-constructed. Must have proper alignment for T.
   * @return true if an element was successfully popped, false if the buffer
   *         was empty or contended.
   *
   * @note The caller is responsible for eventually destroying the constructed object.
   * @note This operation is lock-free and fail-fast (no retry loop).
   */
  bool try_pop_into(T* storage) {
    size_t head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_relaxed)) {
      return false;
    }
    Slot& slot = slots_[wrapIndex(head)];
    size_t seq = slot.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head + 1);
    if (diff == 0) {
      if (head_.compare_exchange_strong(head, head + 1, std::memory_order_relaxed)) {
        T* elem = dataPtr(slot);
        new (storage) T(std::move(*elem));
        elem->~T();
        slot.seq.store(head + kBufferSize, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Attempts to push multiple elements into the buffer.
   *
   * Pushes up to `count` elements from the array into the buffer in a single
   * atomic tail reservation. Returns the number of elements actually pushed,
   * which may be less than `count` if the buffer doesn't have enough space
   * or if there is contention from other producers.
   *
   * Validates each slot's sequence number to determine how many consecutive
   * slots are available, then reserves them with a single CAS on the tail.
   *
   * This design enables natural overflow handling:
   * @code
   * size_t pushed = ring.try_push_batch(items, count);
   * if (pushed < count) {
   *     // Push remainder to central queue or other fallback
   *     centralQueue.push(items + pushed, count - pushed);
   * }
   * @endcode
   *
   * @param items Pointer to the array of items to push. Items are moved from.
   * @param count Number of items to attempt to push.
   * @return The number of items actually pushed (0 to count).
   *
   * @note After a successful push, each slot's sequence number is published
   *       independently, so consumers can start popping slot 0 before slot
   *       count-1 is written.
   */
  size_type try_push_batch(T* items, size_type count) {
    if (count == 0) {
      return 0;
    }
    if (count > kBufferSize) {
      count = kBufferSize;
    }

    size_t tail = tail_.load(std::memory_order_relaxed);

    // Validate each slot in the reservation range.
    size_t available = 0;
    for (size_t i = 0; i < count; ++i) {
      Slot& slot = slots_[wrapIndex(tail + i)];
      size_t seq = slot.seq.load(std::memory_order_acquire);
      intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail + i);
      if (diff != 0) {
        break;
      }
      ++available;
    }
    if (available == 0) {
      return 0;
    }

    if (tail_.compare_exchange_strong(tail, tail + available, std::memory_order_relaxed)) {
      for (size_t i = 0; i < available; ++i) {
        Slot& slot = slots_[wrapIndex(tail + i)];
        new (dataPtr(slot)) T(std::move(items[i]));
        slot.seq.store(tail + i + 1, std::memory_order_release);
      }
      return available;
    }

    return 0;
  }

  /**
   * @brief Checks if the buffer is empty.
   *
   * @return true if the buffer appears to contain no elements.
   *
   * @note This is a snapshot that may be immediately stale. Safe to call
   *       from any thread, but the result is only a hint.
   */
  bool empty() const {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_relaxed);
    return head == tail;
  }

  /**
   * @brief Checks if the buffer is full.
   *
   * @return true if the buffer appears to have no space for additional elements.
   *
   * @note This is a snapshot that may be immediately stale. Safe to call
   *       from any thread, but the result is only a hint.
   */
  bool full() const {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_relaxed);
    return (tail - head) >= kBufferSize;
  }

  /**
   * @brief Returns the approximate number of elements in the buffer.
   *
   * @return An estimate of the number of elements currently in the buffer.
   *
   * @note This is a snapshot that may be immediately stale. Safe to call
   *       from any thread, but the result is only a hint. The value may
   *       momentarily exceed capacity() under concurrent modification.
   */
  size_type size() const {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_relaxed);
    return tail - head;
  }

  /**
   * @brief Returns the maximum number of elements the buffer can hold.
   *
   * When RoundUpToPowerOfTwo is true (default), this may be larger than the
   * requested Capacity template parameter.
   *
   * @return The actual capacity of the buffer.
   */
  static constexpr size_type capacity() noexcept {
    return kBufferSize;
  }

 private:
  static constexpr size_t computeBufferSize() noexcept {
    return RoundUpToPowerOfTwo ? static_cast<size_t>(detail::nextPow2(Capacity)) : Capacity;
  }

  static constexpr size_t kBufferSize = computeBufferSize();
  static_assert(
      (kBufferSize & (kBufferSize - 1)) == 0 || !RoundUpToPowerOfTwo,
      "Internal error: kBufferSize must be power of two when RoundUpToPowerOfTwo is true");
  static constexpr bool kIsPow2 = (kBufferSize & (kBufferSize - 1)) == 0;
  static constexpr size_t kMask = kBufferSize - 1;

  static size_t wrapIndex(size_t i) {
    return kIsPow2 ? (i & kMask) : (i % kBufferSize);
  }

  // Shared single-slot fast path for try_push() and try_emplace(): reserve one slot with a single
  // CAS on the tail, then construct T in place from the forwarded arguments. Intentionally
  // unconstrained -- the public overloads carry the nothrow-construction constraints; this just
  // centralizes the (otherwise identical) algorithm so it lives in one place. Fully inlined, so
  // the forwarding adds no runtime cost on the hot path.
  template <typename... Args>
  bool emplaceImpl(Args&&... args) {
    size_t tail = tail_.load(std::memory_order_relaxed);
    Slot& slot = slots_[wrapIndex(tail)];
    size_t seq = slot.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail);
    if (diff == 0) {
      // ABA-free: tail_ is a monotonic 64-bit counter, so a successful CAS proves no other
      // producer claimed this position since the load (see "Correctness & ABA-freedom" above).
      // Fail-fast: a single attempt, no retry loop -- contention returns false, not corruption.
      if (tail_.compare_exchange_strong(tail, tail + 1, std::memory_order_relaxed)) {
        new (dataPtr(slot)) T(std::forward<Args>(args)...);
        slot.seq.store(tail + 1, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  // The element buffer comes first: placing the over-aligned `data` ahead of `seq` keeps any
  // padding T's alignment requires from landing *between* the two members, which can otherwise
  // tip a slot that would have fit in one cache line over the boundary. The trailing
  // alignas(kCacheLineSize) still rounds the whole slot up to a cache-line multiple to prevent
  // false sharing between neighbors.
  struct alignas(kCacheLineSize) Slot {
    alignas(T) char data[sizeof(T)];
    std::atomic<size_t> seq;
  };

  T* dataPtr(Slot& slot) {
    return reinterpret_cast<T*>(slot.data);
  }

  const T* dataPtr(const Slot& slot) const {
    return reinterpret_cast<const T*>(slot.data);
  }

  /// Head index (consumers CAS this forward). Cache-line aligned to avoid false sharing with tail.
  alignas(kCacheLineSize) std::atomic<size_t> head_{0};

  /// Tail index (producers CAS this forward). Cache-line aligned to avoid false sharing with head.
  alignas(kCacheLineSize) std::atomic<size_t> tail_{0};

  /// Per-slot storage with sequence numbers. Each slot is cache-line aligned.
  Slot slots_[kBufferSize];
};

} // namespace dispenso
