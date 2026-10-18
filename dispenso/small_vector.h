/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file small_vector.h
 * @ingroup group_util
 * A vector-like container with inline storage for small sizes.
 *
 * SmallVector stores up to N elements inline (on the stack), falling back to
 * heap allocation when size exceeds N. Uses the high bit of the size field to
 * track storage mode, keeping the struct compact with no extra flags.
 *
 * Once transitioned to heap storage, the vector stays on heap until clear()
 * is called. This ensures reserve() guarantees are honored and avoids
 * unnecessary copies between storage modes.
 **/

#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <new>
#include <utility>

#include <dispenso/platform.h>

namespace dispenso {

/**
 * A vector-like container that stores up to N elements inline (on the stack),
 * avoiding heap allocation for small sizes. Falls back to heap allocation when
 * the size exceeds N.
 *
 * @tparam T The element type.
 * @tparam N The number of elements to store inline (default 4). Must be in range [1, 65535].
 *
 * Memory layout: Uses a union to share space between inline storage and heap
 * pointer/capacity. The high bit of the size field tracks whether heap storage
 * is active, so no additional bool or flag field is needed.
 *
 * Once on heap, the vector stays on heap until clear() is called. This ensures
 * reserve() capacity is preserved across push_back/pop_back sequences.
 */
template <typename T, size_t N = 4>
class SmallVector {
  static_assert(N > 0, "SmallVector requires at least 1 inline element. Use std::vector for N=0.");
  static_assert(N < 65536, "SmallVector inline capacity is too large. Use std::vector instead.");

 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;

  /**
   * Construct an empty SmallVector.
   **/
  SmallVector() noexcept : size_(0) {}

  /**
   * Construct a SmallVector with @p count default-constructed elements.
   *
   * @param count The number of elements.
   **/
  explicit SmallVector(size_type count) : size_(0) {
    constructOrRelease([&]() { resize(count); });
  }

  /**
   * Construct a SmallVector with @p count copies of @p value.
   *
   * @param count The number of elements.
   * @param value The value to copy into each element.
   **/
  SmallVector(size_type count, const T& value) : size_(0) {
    constructOrRelease([&]() { resize(count, value); });
  }

  /**
   * Construct a SmallVector from an initializer list.
   *
   * @param init The initializer list of elements.
   **/
  SmallVector(std::initializer_list<T> init) : size_(0) {
    constructOrRelease([&]() {
      ensureCapacity(init.size());
      for (const auto& v : init) {
        emplace_back(v);
      }
    });
  }

  /** Copy constructor. **/
  SmallVector(const SmallVector& other) : size_(0) {
    constructOrRelease([&]() {
      ensureCapacity(other.rawSize());
      for (const auto& v : other) {
        emplace_back(v);
      }
    });
  }

  /** Move constructor. Moves elements and leaves @p other empty. **/
  SmallVector(SmallVector&& other) noexcept : size_(0) {
    if (other.isInline()) {
      // Inline means the heap bit is clear, so size_ is already the count, and
      // hoisting it keeps the stores below from forcing a reload each pass.
      const size_type n = other.size_;
      for (size_type i = 0; i < n; ++i) {
        new (inlineData() + i) T(std::move(other.inlineData()[i]));
        other.inlineData()[i].~T();
      }
    } else {
      storage_.heap_.ptr = other.storage_.heap_.ptr;
      storage_.heap_.capacity = other.storage_.heap_.capacity;
    }
    // Count and heap bit both transfer verbatim.
    size_ = other.size_;
    other.size_ = 0;
  }

  ~SmallVector() {
    destroyAll();
  }

  /** Copy assignment operator. **/
  SmallVector& operator=(const SmallVector& other) {
    if (this != &other) {
      destroyAll();
      size_ = 0;
      ensureCapacity(other.rawSize());
      for (const auto& v : other) {
        emplace_back(v);
      }
    }
    return *this;
  }

  /** Move assignment operator. **/
  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this != &other) {
      destroyAll();

      if (other.isInline()) {
        // Inline means the heap bit is clear, so size_ is already the count,
        // and hoisting it keeps the stores below from forcing a reload each
        // pass.
        const size_type n = other.size_;
        for (size_type i = 0; i < n; ++i) {
          new (inlineData() + i) T(std::move(other.inlineData()[i]));
          other.inlineData()[i].~T();
        }
      } else {
        storage_.heap_.ptr = other.storage_.heap_.ptr;
        storage_.heap_.capacity = other.storage_.heap_.capacity;
      }
      // Count and heap bit both transfer verbatim. Nothing between destroyAll()
      // and here reads size_: the inline branch goes through inlineData()
      // rather than data(), and the heap branch overwrites the freed pointer
      // before anything can observe it.
      size_ = other.size_;
      other.size_ = 0;
    }
    return *this;
  }

  // --- Element Access ---

  /** Access element at @p pos (unchecked). **/
  reference operator[](size_type pos) {
    return data()[pos];
  }
  /** @copydoc operator[](size_type) **/
  const_reference operator[](size_type pos) const {
    return data()[pos];
  }
  /** Access the first element. **/
  reference front() {
    return data()[0];
  }
  /** @copydoc front() **/
  const_reference front() const {
    return data()[0];
  }
  /** Access the last element. **/
  reference back() {
    return data()[rawSize() - 1];
  }
  /** @copydoc back() **/
  const_reference back() const {
    return data()[rawSize() - 1];
  }

  /** Return a pointer to the underlying element storage. **/
  pointer data() noexcept {
    return isInline() ? inlineData() : storage_.heap_.ptr;
  }
  /** @copydoc data() **/
  const_pointer data() const noexcept {
    return isInline() ? inlineData() : storage_.heap_.ptr;
  }

  // --- Iterators ---

  /** Return an iterator to the first element. **/
  iterator begin() noexcept {
    return data();
  }
  /** @copydoc begin() **/
  const_iterator begin() const noexcept {
    return data();
  }
  /** @copydoc begin() **/
  const_iterator cbegin() const noexcept {
    return data();
  }
  /** Return an iterator past the last element. **/
  iterator end() noexcept {
    return data() + rawSize();
  }
  /** @copydoc end() **/
  const_iterator end() const noexcept {
    return data() + rawSize();
  }
  /** @copydoc end() **/
  const_iterator cend() const noexcept {
    return data() + rawSize();
  }

  // --- Capacity ---

  /** Check whether the vector is empty. **/
  bool empty() const noexcept {
    return rawSize() == 0;
  }
  /** Return the number of elements. **/
  size_type size() const noexcept {
    return rawSize();
  }

  /**
   * Return the current capacity.
   * Returns N while using inline storage, or the heap capacity otherwise.
   **/
  size_type capacity() const noexcept {
    return isInline() ? N : storage_.heap_.capacity;
  }

  /**
   * Reserve capacity for at least newCap elements.
   * If newCap > N, transitions to heap storage.
   * Once on heap, capacity is preserved until clear().
   */
  void reserve(size_type newCap) {
    ensureCapacity(newCap);
  }

  // --- Modifiers ---

  /**
   * Remove all elements and release heap storage (if any).
   * After clear(), the vector returns to inline storage mode.
   **/
  void clear() noexcept {
    destroyAll();
    size_ = 0;
  }

  /** Append a copy of @p value. **/
  void push_back(const T& value) {
    emplace_back(value);
  }
  /** Append @p value by moving. **/
  void push_back(T&& value) {
    emplace_back(std::move(value));
  }

  /**
   * Construct an element in-place at the end.
   *
   * @param args Arguments forwarded to the element constructor.
   * @return A reference to the newly constructed element.
   **/
  template <typename... Args>
  reference emplace_back(Args&&... args) {
    T* ptr;
    if (isInline()) {
      size_type sz = rawSize();
      if (sz < N) {
        ptr = inlineData();
      } else {
        return growAndEmplaceBack(N * 2, std::forward<Args>(args)...);
      }
    } else {
      size_type sz = rawSize();
      if (sz == storage_.heap_.capacity) {
        return growAndEmplaceBack(storage_.heap_.capacity * 2, std::forward<Args>(args)...);
      }
      ptr = storage_.heap_.ptr;
    }
    size_type idx = rawSize();
    new (ptr + idx) T(std::forward<Args>(args)...);
    // Increment preserves heap bit naturally
    ++size_;
    assert(rawSize() > 0 && "Size overflow into heap bit");
    return ptr[idx];
  }

  /** Remove the last element. **/
  void pop_back() {
    T* ptr = data();
    size_type sz = rawSize();
    ptr[sz - 1].~T();
    // Decrement preserves heap bit naturally
    --size_;
  }

  /**
   * Resize the vector to @p count elements, default-constructing new elements if growing.
   *
   * @param count The desired number of elements.
   **/
  void resize(size_type count) {
    size_type sz = rawSize();
    if (count > sz) {
      ensureCapacity(count);
      constructRange(data(), sz, count, [](T* p) { new (p) T(); });
      setSize(count);
    } else if (count < sz) {
      T* ptr = data();
      for (size_type i = count; i < sz; ++i) {
        ptr[i].~T();
      }
      setSize(count);
    }
  }

  /**
   * Resize the vector to @p count elements, copy-constructing new elements from @p value if
   * growing.
   *
   * @param count The desired number of elements.
   * @param value The value to copy into new elements.
   **/
  void resize(size_type count, const T& value) {
    size_type sz = rawSize();
    if (count > sz) {
      ensureCapacity(count);
      constructRange(data(), sz, count, [&value](T* p) { new (p) T(value); });
      setSize(count);
    } else if (count < sz) {
      T* ptr = data();
      for (size_type i = count; i < sz; ++i) {
        ptr[i].~T();
      }
      setSize(count);
    }
  }

  /**
   * Erase the element at @p pos.
   * Elements after @p pos are shifted left. Returns an iterator to the element that now
   * occupies the erased position (or end() if the last element was erased).
   *
   * @param pos Iterator to the element to erase.
   * @return Iterator to the element following the erased one.
   **/
  iterator erase(const_iterator pos) {
    T* ptr = data();
    size_type sz = rawSize();
    size_type index = pos - ptr;

    for (size_type i = index; i + 1 < sz; ++i) {
      ptr[i] = std::move(ptr[i + 1]);
    }
    ptr[sz - 1].~T();
    --size_; // Preserves heap bit
    return data() + index;
  }

 private:
  // High bit of size_ tracks heap vs inline mode.
  // Since kHeapBit >> N, comparisons like size_ <= N and size_ < N
  // are naturally false when the heap bit is set, so isInline() and
  // many internal checks work without masking.
  static constexpr size_type kHeapBit = size_type(1) << (sizeof(size_type) * 8 - 1);
  static constexpr size_type kSizeMask = ~kHeapBit;

  bool isInline() const noexcept {
    return (size_ & kHeapBit) == 0;
  }

  size_type rawSize() const noexcept {
    return size_ & kSizeMask;
  }

  // Set the size portion, preserving the heap bit.
  void setSize(size_type s) noexcept {
    assert((s & kHeapBit) == 0 && "Size overflow into heap bit");
    size_ = (size_ & kHeapBit) | s;
  }

  T* inlineData() noexcept {
    return reinterpret_cast<T*>(&storage_.inline_);
  }
  const T* inlineData() const noexcept {
    return reinterpret_cast<const T*>(&storage_.inline_);
  }

  // For constructors: run fill(), and if it throws, destroy whatever it built and free any heap
  // storage before propagating, since the destructor does not run for a constructor that throws.
  template <typename Fill>
  void constructOrRelease(Fill fill) {
#if defined(__cpp_exceptions)
    try {
      fill();
    } catch (...) {
      destroyAll();
      throw;
    }
#else
    fill();
#endif // __cpp_exceptions
  }

  // Construct ptr[begin, end) by calling construct(ptr + i) for each slot. If one throws, the
  // elements already built in this range are destroyed before the exception propagates.
  template <typename Construct>
  static void constructRange(T* ptr, size_type begin, size_type end, Construct construct) {
    size_type i = begin;
#if defined(__cpp_exceptions)
    try {
#endif // __cpp_exceptions
      for (; i < end; ++i) {
        construct(ptr + i);
      }
#if defined(__cpp_exceptions)
    } catch (...) {
      for (size_type j = begin; j < i; ++j) {
        ptr[j].~T();
      }
      throw;
    }
#endif // __cpp_exceptions
  }

  void destroyAll() noexcept {
    T* ptr = data();
    size_type sz = rawSize();
    for (size_type i = 0; i < sz; ++i) {
      ptr[i].~T();
    }
    if (!isInline()) {
      deallocate(storage_.heap_.ptr);
    }
  }

  // Heap storage for over-aligned T must honour alignof(T); plain ::operator new only guarantees
  // alignof(std::max_align_t), and the aligned overloads need C++17.
  static constexpr bool kOverAligned = alignof(T) > alignof(std::max_align_t);

  static T* allocate(size_type count) {
    if (kOverAligned) {
      return static_cast<T*>(detail::alignedMallocOrThrow(count * sizeof(T), alignof(T)));
    }
    return static_cast<T*>(::operator new(count * sizeof(T)));
  }

  static void deallocate(T* ptr) noexcept {
    if (kOverAligned) {
      detail::alignedFree(ptr);
      return;
    }
    // GCC on 32-bit ARM reports this, called from destroyAll(), as freeing the
    // inline buffer. It is a false positive: the heap bit lives inside size_
    // and the buffer is a union member aliasing heap_.ptr, and across the
    // inlined move paths GCC stops being able to prove the !isInline() guard
    // excludes inline storage. The same code is clean under GCC on x86_64,
    // x86-32, aarch64, riscv32 and riscv64, which is not what a genuine logic
    // error would look like. Simplifying the move paths to a single size_
    // assignment did not change its mind either.
    DISPENSO_DISABLE_WARNING_PUSH
    DISPENSO_DISABLE_WARNING_FREE_NONHEAP_OBJECT
    ::operator delete(ptr);
    DISPENSO_DISABLE_WARNING_POP
  }

  // Grow to heap storage with the specified capacity.
  // Moves existing elements, frees old heap if applicable, sets heap bit. If transferring an
  // element throws, the new storage is freed and the vector is left as it was.
  void growToHeap(size_type newCap) {
    T* newData = allocate(newCap);
#if defined(__cpp_exceptions)
    try {
      transferTo(newData);
    } catch (...) {
      deallocate(newData);
      throw;
    }
#else
    transferTo(newData);
#endif // __cpp_exceptions
    adoptHeap(newData, newCap);
  }

  // Construct the existing elements in newData, moving them when T's move constructor cannot throw
  // and copying them otherwise (std::move_if_noexcept, as std::vector does). If a construction
  // throws, the elements constructed in newData so far are destroyed and the exception propagates
  // with the existing elements intact (unless T is move-only with a throwing move constructor, in
  // which case some of them may have been moved from). newData is left for the caller to free.
  void transferTo(T* newData) {
    T* oldData = data();
    size_type sz = rawSize();
    size_type i = 0;
#if defined(__cpp_exceptions)
    try {
#endif // __cpp_exceptions
      for (; i < sz; ++i) {
        new (newData + i) T(std::move_if_noexcept(oldData[i]));
      }
#if defined(__cpp_exceptions)
    } catch (...) {
      for (size_type j = 0; j < i; ++j) {
        newData[j].~T();
      }
      throw;
    }
#endif // __cpp_exceptions
  }

  // Commit after a successful transferTo(newData): destroy the old elements, free the old heap
  // storage if any, and switch to heap mode using newData (capacity newCap).
  void adoptHeap(T* newData, size_type newCap) noexcept {
    T* oldData = data();
    size_type sz = rawSize();

    for (size_type i = 0; i < sz; ++i) {
      oldData[i].~T();
    }

    if (!isInline()) {
      deallocate(storage_.heap_.ptr);
    }

    storage_.heap_.ptr = newData;
    storage_.heap_.capacity = newCap;
    size_ = kHeapBit | sz;
  }

  // emplace_back() when full: grow to newCap and append. The new element is constructed in the new
  // storage before the existing elements are moved out of the old storage, because args may refer
  // to one of those elements (e.g. v.push_back(v[0])). If anything throws, the new storage is freed
  // and the vector is left as it was.
  template <typename... Args>
  reference growAndEmplaceBack(size_type newCap, Args&&... args) {
    T* newData = allocate(newCap);
    size_type sz = rawSize();
#if defined(__cpp_exceptions)
    try {
      new (newData + sz) T(std::forward<Args>(args)...);
    } catch (...) {
      deallocate(newData);
      throw;
    }
    try {
      transferTo(newData);
    } catch (...) {
      newData[sz].~T();
      deallocate(newData);
      throw;
    }
#else
    new (newData + sz) T(std::forward<Args>(args)...);
    transferTo(newData);
#endif // __cpp_exceptions
    adoptHeap(newData, newCap);
    ++size_; // Preserves heap bit
    assert(rawSize() > 0 && "Size overflow into heap bit");
    return newData[sz];
  }

  void ensureCapacity(size_type newCap) {
    if (newCap <= N && isInline()) {
      return;
    }
    if (isInline()) {
      growToHeap(newCap);
    } else if (newCap > storage_.heap_.capacity) {
      growToHeap(newCap);
    }
  }

  size_type size_;

  struct HeapStorage {
    T* ptr;
    size_type capacity;
  };

  union Storage {
    alignas(T) unsigned char inline_[sizeof(T) * N];
    HeapStorage heap_;
    Storage() noexcept {}
    ~Storage() {}
  } storage_;
};

} // namespace dispenso
