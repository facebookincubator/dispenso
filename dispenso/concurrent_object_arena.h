
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

/**
 * @file future.h
 * A file providing concurrent object arena container.
 **/

#pragma once

#include <dispenso/platform.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>

namespace detail {

template <class T>
constexpr T log2i(const T v) {
  T log2 = 0, val = v;
  while (val >>= 1)
    ++log2;
  return log2;
}

} // namespace detail

namespace dispenso {
/**
 * <code>ConcurrentObjectArena</code> is an indexed sequence container that
 * allows concurrent insertion to its end. Insertion never invalidates pointers
 * or references to the rest of the elements. As opposet to std::vector, the
 * elements of a <code>ConcurrentObjectArena</code> are not stored
 * contiguously. In memory it is sequence of individually allocated fixed-size
 * arrays, with additional bookkeeping. The size of arrays is always power of
 * two (to optimize indexed access)
 * <pre>
 *  buffers  |<────     bufferSize      ───>|
 *    ┌─┐    ┌──────────────────────────────┐
 *    │*├───>│                              │
 *    ├─┤    ├──────────────────────────────┤
 *    │*├───>│                              │
 *    ├─┤    ├──────────────────────────────┤
 *    │*├───>│                              │
 *    └─┘    └──────────────────────────────┘
 *</pre>
 **/
template <class T, class Index = size_t, size_t alignment = dispenso::kCacheLineSize>
struct ConcurrentObjectArena {
  ConcurrentObjectArena() = delete;
  /**
   * Construct a <code>ConcurrentObjectArena</code> with given or bigger contiguous array size
   *
   * @param minBuffSize The minimum size of the internal buffer. If given
   * size is not power of 2 the closest bigger power of two would be chosen.
   **/
  explicit ConcurrentObjectArena(const Index minBuffSize)
      : kLog2BuffSize(
            ::detail::log2i(minBuffSize) +
            ((Index{1} << ::detail::log2i(minBuffSize)) == minBuffSize ? 0 : 1)),
        kBufferSize(Index{1} << kLog2BuffSize),
        kMask((Index{1} << kLog2BuffSize) - 1),
        pos_(0),
        allocatedSize_(0),
        buffers_(nullptr),
        buffersSize_(0),
        buffersPos_(0) {
    allocateBuffer();
    allocatedSize_.store(kBufferSize, std::memory_order_relaxed);
  }

  ~ConcurrentObjectArena() {
    T** buffers = buffers_.load(std::memory_order_relaxed);

    for (Index i = 0; i < buffersPos_; i++)
      detail::alignedFree(buffers[i]);

    delete[] buffers;

    for (T** p : deleteLater_)
      delete[] p;
  }

  /**
   * Grow a container
   *
   * This function is thread safe and never invalidates pointers or
   * references to the rest of the elements. It is lock-free if new elements
   * can be placed in current buffer. It locks if it allocates a new buffer.
   * @param delta New size of the container will be <code>delta<\code> elements bigger.
   * @return index of the first element of the allocated group.
   **/
  Index grow_by(const Index delta) {
    Index newPos;
    Index oldPos = pos_.load(std::memory_order_relaxed);

    do {
      Index curSize = allocatedSize_.load(std::memory_order_acquire);

      if (oldPos + delta >= curSize) {
        const std::lock_guard<std::mutex> guard(resizeMutex_);
        curSize = allocatedSize_.load(std::memory_order_relaxed);
        while (oldPos + delta >= curSize) {
          allocateBuffer();
          allocatedSize_.store(curSize + kBufferSize, std::memory_order_release);
          curSize = curSize + kBufferSize;
        }
      }

      newPos = oldPos + delta;
    } while (!std::atomic_compare_exchange_weak_explicit(
        &pos_, &oldPos, newPos, std::memory_order_release, std::memory_order_relaxed));

    constructObjects(oldPos, oldPos + delta);

    return oldPos;
  }

  inline const T& operator[](const Index index) const {
    const Index bufIndex = index >> kLog2BuffSize;
    const Index i = index & kMask;
    assert(bufIndex < numBuffers());

    return buffers_.load(std::memory_order_relaxed)[bufIndex][i];
  }

  inline T& operator[](const Index index) {
    return const_cast<T&>(const_cast<const ConcurrentObjectArena<T, Index>&>(*this)[index]);
  }

  Index size() const {
    return pos_.load(std::memory_order_relaxed);
  }

  Index capacity() const {
    return allocatedSize_.load(std::memory_order_relaxed);
  }

  Index numBuffers() const {
    return buffersPos_;
  }

  const T* getBuffer(const Index index) const {
    return buffers_.load(std::memory_order_relaxed)[index];
  }

  T* getBuffer(const Index index) {
    return buffers_.load(std::memory_order_relaxed)[index];
  }

  Index getBufferSize(const Index index) const {
    const Index numBuffs = numBuffers();
    assert(index < numBuffs);

    if (index < numBuffs - 1)
      return kBufferSize;
    else
      return pos_.load(std::memory_order_relaxed) - (kBufferSize * (numBuffs - 1));
  }

 private:
  void allocateBuffer() {
    void* ptr = detail::alignedMalloc(kBufferSize * sizeof(T), alignment);
#if defined(__cpp_exceptions)
    if (ptr == nullptr)
      throw std::bad_alloc();
#endif // __cpp_exceptions

    if (buffersPos_ < buffersSize_) {
      buffers_.load(std::memory_order_relaxed)[buffersPos_++] = static_cast<T*>(ptr);
    } else {
      const Index oldBuffersSize = buffersSize_;
      T** oldBuffers = buffers_.load(std::memory_order_relaxed);

      buffersSize_ = oldBuffersSize == 0 ? 2 : oldBuffersSize * 2;
      T** newBuffers = new T*[buffersSize_];

      if (oldBuffers != nullptr) {
        std::memcpy(newBuffers, oldBuffers, sizeof(T*) * oldBuffersSize);
        deleteLater_.push_back(oldBuffers);
      }

      newBuffers[buffersPos_++] = static_cast<T*>(ptr);
      buffers_.store(newBuffers, std::memory_order_relaxed);
    }
  }

  void constructObjects(const Index beginIndex, const Index endIndex) {
    const Index startBuffer = beginIndex >> kLog2BuffSize;
    const Index endBuffer = endIndex >> kLog2BuffSize;

    Index bufStart = beginIndex & kMask;
    for (Index b = startBuffer; b <= endBuffer; ++b) {
      T* buf = buffers_.load(std::memory_order_relaxed)[b];
      const Index bufEnd = b == endBuffer ? (endIndex & kMask) : kMask + 1;
      for (Index i = bufStart; i < bufEnd; ++i)
        new (buf + i) T();

      bufStart = 0;
    }
  }
  //
  //    kBufferSize = 2^kLog2BuffSize
  //    mask = 0b00011111
  //                ──┬──
  //                  └─number of 1s is log2BuffSize
  //
  const Index kLog2BuffSize;
  const Index kBufferSize;
  const Index kMask;

  std::mutex resizeMutex_;

  std::atomic<Index> pos_;
  std::atomic<Index> allocatedSize_;

  std::atomic<T**> buffers_;
  Index buffersSize_;
  Index buffersPos_;
  std::vector<T**> deleteLater_;
};

} // namespace dispenso
