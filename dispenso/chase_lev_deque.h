/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file chase_lev_deque.h
 * @ingroup group_util
 * A lock-free single-producer / multi-consumer (SPMC) work-stealing deque.
 *
 * This is the bounded variant of the Chase-Lev deque (Chase & Lev, 2005), the standard
 * data structure backing fork-join work-stealing schedulers (e.g., TBB, Cilk). The owning
 * thread pushes and pops at the "bottom" end with no atomic CAS on the fast path; other
 * threads steal from the "top" end via CAS. Owner pop is LIFO (newest-first) for cache
 * locality; steal is FIFO (oldest-first) to take broad chunks of work.
 *
 * @note Only the owning thread may push or pop. Any thread may steal.
 *
 * @see "Dynamic Circular Work-Stealing Deque", Chase & Lev, SPAA 2005.
 * @see "Correct and Efficient Work-Stealing for Weak Memory Models", Lê et al., PPoPP 2013.
 **/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include <dispenso/platform.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {

/**
 * @class ChaseLevDeque
 * @brief A lock-free SPMC bounded work-stealing deque.
 *
 * The owning thread pushes/pops at the bottom end (LIFO), and any thread can steal from
 * the top end (FIFO). The fast paths (push, non-contended pop) use no compare-and-swap;
 * only the contended last-element pop and steal require CAS. This makes the deque the
 * standard primitive for low-overhead recursive fork-join work distribution.
 *
 * Storage is a fixed-size circular buffer indexed by monotonically-advancing top and
 * bottom counters. Capacity is fixed at compile time and must be a power of two.
 *
 * @tparam T The type of elements stored. Must be trivially copyable. The Chase-Lev
 *           steal protocol performs a tentative read of the slot before the CAS that
 *           claims it; a losing stealer simply discards the read. This is sound only
 *           when the read is a value copy that leaves the slot intact, which the
 *           trivially-copyable constraint guarantees. For move-only payloads, store
 *           a raw pointer (or other small POD wrapper) in the deque.
 * @tparam Capacity Maximum number of in-flight elements. Must be a power of two and
 *                  at least 1. Defaults to 32, sized for typical recursive fork-join
 *                  depths while keeping the buffer in a few cache lines.
 *
 * ## Thread Safety
 *
 * - Exactly one thread (the owner) may call `try_push`, `try_pop`, and `try_pop_into`.
 * - Any thread (including the owner) may call `try_steal` and `try_steal_into`.
 * - `empty()`, `size()`, and `capacity()` may be called from any thread, but provide
 *   only a snapshot that may be immediately stale.
 *
 * ## Steal Contention
 *
 * `try_steal` returns false on both an empty deque and a lost CAS race. The two cases
 * are not distinguished in the API; callers that want to retry on contention may
 * compare `size()` before and after, or simply retry a small number of times.
 *
 * ## Performance Characteristics
 *
 * - Push: one relaxed load + one release store (of `bottom_`) + one acquire load. No CAS.
 * - Non-contended pop: one relaxed store + a seq_cst fence + one relaxed load. No CAS.
 * - Last-element pop and steal: one CAS on `top_`.
 *
 * ## Example Usage
 *
 * @code
 * dispenso::ChaseLevDeque<int, 64> deque;
 *
 * // Owner thread
 * deque.try_push(42);
 * int v;
 * if (deque.try_pop(v)) { use(v); }
 *
 * // Any thread
 * int stolen;
 * if (deque.try_steal(stolen)) { use(stolen); }
 * @endcode
 */
template <typename T, size_t Capacity = 32>
#if DISPENSO_HAS_CONCEPTS
  requires std::is_trivially_copyable_v<T>
#endif
class ChaseLevDeque {
  static_assert(Capacity >= 1, "ChaseLevDeque capacity must be at least 1");
  static_assert((Capacity & (Capacity - 1)) == 0, "ChaseLevDeque capacity must be a power of two");
#if !DISPENSO_HAS_CONCEPTS
  static_assert(
      std::is_trivially_copyable<T>::value,
      "ChaseLevDeque requires a trivially copyable element type. "
      "Wrap move-only payloads in a raw pointer or POD container.");
#endif

 public:
  using value_type = T;
  using size_type = size_t;

  /**
   * @brief Constructs an empty deque. Storage is uninitialized; elements are
   *        constructed in-place when pushed.
   */
  ChaseLevDeque() = default;

  ChaseLevDeque(const ChaseLevDeque&) = delete;
  ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;
  ChaseLevDeque(ChaseLevDeque&&) = delete;
  ChaseLevDeque& operator=(ChaseLevDeque&&) = delete;

  /**
   * @brief Destroys the deque. Trivially-copyable elements need no explicit teardown.
   *
   * @note Destruction is not thread-safe. Ensure no thread is pushing, popping, or
   *       stealing when the deque is destroyed.
   */
  ~ChaseLevDeque() = default;

  // ---------------------------------------------------------------------------
  // Owner-only operations: push and pop.
  // ---------------------------------------------------------------------------

  /**
   * @brief Pushes an element onto the bottom of the deque (owner-only).
   *
   * @param item Element to push.
   * @return true on success, false if the deque is full.
   *
   * @note Only the owning thread may call this. The fast path is one relaxed load
   *       and one release store (of `bottom_`) plus one acquire load (of `top_`) for
   *       the full/empty check, and the slot write. No fence, no CAS.
   */
  bool try_push(const T& item) {
    const int64_t b = bottom_.load(std::memory_order_relaxed);
    const int64_t t = top_.load(std::memory_order_acquire);
    if (b - t >= static_cast<int64_t>(Capacity)) {
      return false;
    }
    // The slot write is ordered after a prior stealer's CAS on top_ (see try_steal):
    // a stealer that read this slot tentatively must have CAS'd top_ before any
    // subsequent push could wrap around to this index. TSAN doesn't model the
    // through-CAS HB chain on non-atomic slot accesses, so annotate as benign.
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
    *slotPtr(b) = item;
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
    // Release on bottom_ publishes the slot write to any acquire load of bottom_
    // (i.e., the load in try_steal).
    bottom_.store(b + 1, std::memory_order_release);
    return true;
  }

  /**
   * @brief Pops the most recently pushed element (LIFO, owner-only).
   *
   * @param[out] out Destination; written on success; value is unspecified on failure.
   * @return true on success, false if the deque is empty or the last element was
   *         simultaneously stolen.
   *
   * @note Only the owning thread may call this. On the non-contended path there is
   *       no CAS — only a relaxed store, a seq_cst fence, and a relaxed load.
   */
  bool try_pop(T& out) {
    const int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    // Seq-cst fence orders our bottom store before the top load, matching steal's fence.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top_.load(std::memory_order_relaxed);

    if (t > b) {
      // Empty. Restore bottom.
      bottom_.store(b + 1, std::memory_order_relaxed);
      return false;
    }
    // Read directly into out. Safe because T is trivially copyable; on CAS loss,
    // out is unspecified per the try_* contract.
    out = *slotPtr(b);
    if (t < b) {
      // Multiple elements remained; no race possible at this slot.
      return true;
    }
    // t == b: last element. Race with stealers via CAS on top.
    bottom_.store(b + 1, std::memory_order_relaxed);
    return top_.compare_exchange_strong(
        t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed);
  }

  /**
   * @brief Pops the most recently pushed element into uninitialized storage
   *        (LIFO, owner-only). Useful when T is not default-constructible.
   *
   * @param[out] storage Pointer to suitably-aligned uninitialized storage.
   * @return true on success, false otherwise.
   */
  bool try_pop_into(T* storage) {
    const int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top_.load(std::memory_order_relaxed);

    if (t > b) {
      bottom_.store(b + 1, std::memory_order_relaxed);
      return false;
    }
    if (t < b) {
      std::memcpy(storage, slotPtr(b), sizeof(T));
      return true;
    }
    bottom_.store(b + 1, std::memory_order_relaxed);
    alignas(T) char tmp[sizeof(T)];
    std::memcpy(tmp, slotPtr(b), sizeof(T));
    if (!top_.compare_exchange_strong(
            t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return false;
    }
    std::memcpy(storage, tmp, sizeof(T));
    return true;
  }

  // ---------------------------------------------------------------------------
  // Stealer operations: callable from any thread, including the owner.
  // ---------------------------------------------------------------------------

  /**
   * @brief Steals the oldest element from the top of the deque (FIFO).
   *
   * @param[out] out Destination; written on success; value is unspecified on failure.
   * @return true on success, false if the deque is empty or the CAS was lost to
   *         another concurrent stealer or a last-element pop.
   *
   * @note Callable from any thread. Distinguishing "empty" from "contended" is
   *       not exposed; callers may retry on false if work is expected.
   */
  bool try_steal(T& out) {
    int64_t t = top_.load(std::memory_order_acquire);
    // Seq-cst fence pairs with the fence in try_pop: ensures we observe an updated
    // bottom if the owner has decremented past us.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int64_t b = bottom_.load(std::memory_order_acquire);
    if (t >= b) {
      return false;
    }
    // Read directly into out before CAS. Safe because T is trivially copyable —
    // slot[t] is unchanged whether we win or lose. On CAS loss, out is unspecified
    // per the try_* contract. The non-atomic slot read is ordered against the
    // owner's earlier slot write via the chain: owner writes slot[b] → owner
    // bottom_.store(b+1, release) → our bottom_.load(acquire) above synchronizes
    // with that release → slot read happens-before stealer CAS. The seq_cst CAS
    // on top_ orders us against other concurrent stealers. TSAN doesn't model
    // this chain through non-atomic reads, so annotate as benign.
    DISPENSO_TSAN_ANNOTATE_IGNORE_READS_BEGIN();
    out = *slotPtr(t);
    DISPENSO_TSAN_ANNOTATE_IGNORE_READS_END();
    return top_.compare_exchange_strong(
        t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed);
  }

  /**
   * @brief Steals the oldest element into uninitialized storage (FIFO).
   * @see try_steal(T&)
   */
  bool try_steal_into(T* storage) {
    int64_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int64_t b = bottom_.load(std::memory_order_acquire);
    if (t >= b) {
      return false;
    }
    // Tentative copy via memcpy (T is trivially copyable). Discarded on CAS loss.
    // See try_steal for the reasoning behind the TSAN annotations.
    alignas(T) char tmp[sizeof(T)];
    DISPENSO_TSAN_ANNOTATE_IGNORE_READS_BEGIN();
    std::memcpy(tmp, slotPtr(t), sizeof(T));
    DISPENSO_TSAN_ANNOTATE_IGNORE_READS_END();
    if (!top_.compare_exchange_strong(
            t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return false;
    }
    std::memcpy(storage, tmp, sizeof(T));
    return true;
  }

  // ---------------------------------------------------------------------------
  // Observers. Snapshots only; results may be immediately stale.
  // ---------------------------------------------------------------------------

  /// @brief Returns true if the deque appears empty.
  bool empty() const {
    const int64_t b = bottom_.load(std::memory_order_acquire);
    const int64_t t = top_.load(std::memory_order_acquire);
    return b <= t;
  }

  /// @brief Returns the apparent number of elements in the deque.
  size_type size() const {
    const int64_t b = bottom_.load(std::memory_order_acquire);
    const int64_t t = top_.load(std::memory_order_acquire);
    return b > t ? static_cast<size_type>(b - t) : 0;
  }

  /// @brief Returns the maximum number of elements the deque can hold.
  static constexpr size_type capacity() noexcept {
    return Capacity;
  }

 private:
  static constexpr size_t kMask = Capacity - 1;
  static constexpr size_t kStorageAlign =
      (alignof(T) > kCacheLineSize) ? alignof(T) : kCacheLineSize;

  T* slotPtr(int64_t index) {
    return reinterpret_cast<T*>(&storage_[(static_cast<size_t>(index) & kMask) * sizeof(T)]);
  }

  /// Top of deque: incremented by stealers and by the owner's last-element pop.
  /// Cache-line isolated so stealer CAS contention does not invalidate the owner's
  /// hot bottom_ line.
  alignas(kCacheLineSize) std::atomic<int64_t> top_{0};

  /// Bottom of deque: incremented on push, decremented on pop. Owner-only writer.
  alignas(kCacheLineSize) std::atomic<int64_t> bottom_{0};

  /// Element storage, cache-line isolated from top_/bottom_.
  alignas(kStorageAlign) char storage_[sizeof(T) * Capacity];
};

} // namespace dispenso
