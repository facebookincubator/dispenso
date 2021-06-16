// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

// This file intended for textual inclusion into concurrent_vector.h only

namespace cv {

struct BucketInfo {
  size_t bucket;
  size_t bucketIndex;
  size_t bucketCapacity;
};

template <typename VecT, typename T>
class ConVecIterBase {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = ssize_t;
  using value_type = T;

  ConVecIterBase(const VecT* vec, BucketInfo info);

 protected:
  ConVecIterBase(uintptr_t vb, T* bucketStart, T* bucketPtr, T* bucketEnd)
      : vb_(vb), bucketStart_(bucketStart), bucketPtr_(bucketPtr), bucketEnd_(bucketEnd) {}

  struct VecAndBucket {
    const VecT* vec;
    const size_t bucket;
  };
  VecAndBucket getVecAndBucket() const {
    static_assert(
        alignof(VecT) >= 64,
        "This code relies on the ConcurrentVector pointer to be 64-byte aligned");
    if (sizeof(uintptr_t) == 8) {
      return {reinterpret_cast<const VecT*>(vb_ & 0xffffffffffffffc0UL), vb_ & 0x3f};
    } else {
      return {reinterpret_cast<const VecT*>(vb_ & 0xffffffc0), vb_ & 0x3f};
    }
  }

  // Note: Could have a size-optimized iterator via traits.  It is easy to reduce this to 24 bytes
  // on 64-bit platform, but this will result in a large enough performance hit that storing an
  // index is preferable (see CompactCVecIterator).  16 bytes on 32-bit platform.
  alignas(4 * sizeof(uintptr_t)) uintptr_t vb_;
  T* bucketStart_;
  T* bucketPtr_;
  T* bucketEnd_;

 private:
  friend bool operator!=(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.bucketPtr_ != b.bucketPtr_;
  }
  friend bool operator==(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.bucketPtr_ == b.bucketPtr_;
  }
  friend bool operator<(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.vb_ < b.vb_ || (a.vb_ == b.vb_ && a.bucketPtr_ < b.bucketPtr_);
  }
  friend bool operator<=(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.vb_ < b.vb_ || (a.vb_ == b.vb_ && a.bucketPtr_ <= b.bucketPtr_);
  }
  friend bool operator>(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.vb_ > b.vb_ || (a.vb_ == b.vb_ && a.bucketPtr_ > b.bucketPtr_);
  }
  friend bool operator>=(const ConVecIterBase& a, const ConVecIterBase& b) {
    return a.vb_ > b.vb_ || (a.vb_ == b.vb_ && a.bucketPtr_ >= b.bucketPtr_);
  }

  friend difference_type operator-(const ConVecIterBase& a, const ConVecIterBase& b) {
    size_t abucket = a.vb_ & 0x3f;
    size_t bbucket = b.vb_ & 0x3f;

    if (abucket == bbucket) {
      return a.bucketPtr_ - b.bucketPtr_;
    }

    ssize_t aLeft = a.bucketPtr_ - a.bucketStart_;
    ssize_t bLeft = b.bucketPtr_ - b.bucketStart_;

    aLeft += (bool)abucket * (a.bucketEnd_ - a.bucketStart_);
    bLeft += (bool)bbucket * (b.bucketEnd_ - b.bucketStart_);

    return aLeft - bLeft;
  }
};

template <typename VecT, typename T, bool kIsConst>
class ConcurrentVectorIterator : public ConVecIterBase<VecT, T> {
 public:
  using iterator_category = typename ConVecIterBase<VecT, T>::iterator_category;
  using difference_type = typename ConVecIterBase<VecT, T>::difference_type;
  using value_type = typename ConVecIterBase<VecT, T>::value_type;
  using pointer = std::conditional_t<kIsConst, std::add_const_t<T*>, T*>;
  using reference = std::conditional_t<kIsConst, std::add_const_t<T&>, T&>;

  ConcurrentVectorIterator(const VecT* vec, BucketInfo info) : ConVecIterBase<VecT, T>(vec, info) {}
  ConcurrentVectorIterator(const VecT* vec, size_t /*index*/, BucketInfo info)
      : ConcurrentVectorIterator(vec, info) {}

  template <bool kWasConst, typename = std::enable_if_t<kIsConst && !kWasConst>>
  ConcurrentVectorIterator(const ConcurrentVectorIterator<VecT, T, kWasConst>& other)
      : ConVecIterBase<VecT, T>(other) {}

  reference operator*() const;
  pointer operator->() const;

  reference operator[](difference_type n) const;

  ConcurrentVectorIterator& operator++();

  ConcurrentVectorIterator operator++(int) {
    auto result = *this;
    operator++();
    return result;
  }

  ConcurrentVectorIterator& operator--();

  ConcurrentVectorIterator operator--(int) {
    auto result = *this;
    operator--();
    return result;
  }

  ConcurrentVectorIterator& operator+=(difference_type n);
  ConcurrentVectorIterator& operator-=(difference_type n) {
    return operator+=(-n);
  }

  ConcurrentVectorIterator operator+(difference_type n) const;
  ConcurrentVectorIterator operator-(difference_type n) const {
    return operator+(-n);
  }

 private:
  ConcurrentVectorIterator(uintptr_t vb, T* bucketStart, T* bucketPtr, T* bucketEnd)
      : ConVecIterBase<VecT, T>(vb, bucketStart, bucketPtr, bucketEnd) {}

  using ConVecIterBase<VecT, T>::getVecAndBucket;

  using ConVecIterBase<VecT, T>::vb_;
  using ConVecIterBase<VecT, T>::bucketStart_;
  using ConVecIterBase<VecT, T>::bucketPtr_;
  using ConVecIterBase<VecT, T>::bucketEnd_;
};

template <typename VecT, typename T>
class CompactCVecIterBase {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = ssize_t;
  using value_type = T;

  CompactCVecIterBase(const VecT* vec, size_t index) : vec_(vec), index_(index) {}

 protected:
  // 16 bytes on 64-bit platform, 8 bytes on 32-bit platform.
  const VecT* vec_;
  size_t index_;

 private:
  friend bool operator!=(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ != b.index_;
  }
  friend bool operator==(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ == b.index_;
  }
  friend bool operator<(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ < b.index_;
  }
  friend bool operator<=(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ <= b.index_;
  }
  friend bool operator>(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ > b.index_;
  }
  friend bool operator>=(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ >= b.index_;
  }

  friend difference_type operator-(const CompactCVecIterBase& a, const CompactCVecIterBase& b) {
    return a.index_ - b.index_;
  }
};

template <typename VecT, typename T, bool kIsConst>
class CompactCVecIterator : public CompactCVecIterBase<VecT, T> {
 public:
  using difference_type = ssize_t;
  using value_type = T;
  using pointer = std::conditional_t<kIsConst, std::add_const_t<T*>, T*>;
  using reference = std::conditional_t<kIsConst, std::add_const_t<T&>, T&>;
  using iterator_category = std::random_access_iterator_tag;

  CompactCVecIterator(const VecT* vec, size_t index, BucketInfo)
      : CompactCVecIterator(vec, index) {}
  CompactCVecIterator(const VecT* vec, size_t index) : CompactCVecIterBase<VecT, T>(vec, index) {}
  template <bool kWasConst, typename = std::enable_if_t<kIsConst && !kWasConst>>
  CompactCVecIterator(const CompactCVecIterator<VecT, T, kWasConst>& other)
      : CompactCVecIterBase<VecT, T>(other) {}
  reference operator*() const;
  pointer operator->() const;

  reference operator[](difference_type n) const;

  CompactCVecIterator& operator++() {
    ++index_;
    return *this;
  }

  CompactCVecIterator operator++(int) {
    auto result = *this;
    operator++();
    return result;
  }

  CompactCVecIterator& operator--() {
    --index_;
    return *this;
  }

  CompactCVecIterator operator--(int) {
    auto result = *this;
    operator--();
    return result;
  }

  CompactCVecIterator& operator+=(difference_type n) {
    index_ += n;
    return *this;
  }
  CompactCVecIterator& operator-=(difference_type n) {
    return operator+=(-n);
  }

  CompactCVecIterator operator+(difference_type n) const {
    return {vec_, index_ + n};
  }
  CompactCVecIterator operator-(difference_type n) const {
    return operator+(-n);
  }

 private:
  using CompactCVecIterBase<VecT, T>::vec_;
  using CompactCVecIterBase<VecT, T>::index_;
};

template <typename T>
inline T* alloc(size_t elts) {
  return reinterpret_cast<T*>(detail::alignedMalloc(elts * sizeof(T), alignof(T)));
}

template <typename T>
inline void dealloc(T* p) {
  detail::alignedFree(p);
}

template <typename T, size_t kMinBufferSize, size_t kMaxVectorSize, bool kMakeInline>
class ConVecBufferBase {};

template <typename T, size_t kMinBufferSize, size_t kMaxVectorSize>
class ConVecBufferBase<T, kMinBufferSize, kMaxVectorSize, false> {
 public:
  static constexpr size_t kMaxBuffers = detail::log2const(kMaxVectorSize / kMinBufferSize) + 1;

  ConVecBufferBase() : buffers_(alloc<detail::AlignedAtomic<T>>(kMaxBuffers)) {
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      buffers_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  ~ConVecBufferBase() {
    dealloc<detail::AlignedAtomic<T>>(buffers_);
  }

  ConVecBufferBase(ConVecBufferBase&& other)
      : buffers_(std::exchange(other.buffers_, alloc<detail::AlignedAtomic<T>>(kMaxBuffers))) {
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      other.buffers_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  ConVecBufferBase& operator=(ConVecBufferBase&& other) {
    using std::swap;
    swap(other.buffers_, buffers_);
    return *this;
  }

 protected:
  detail::AlignedAtomic<T>* buffers_;
};

template <typename T, size_t kMinBufferSize, size_t kMaxVectorSize>
class ConVecBufferBase<T, kMinBufferSize, kMaxVectorSize, true> {
 public:
  static constexpr size_t kMaxBuffers = detail::log2const(kMaxVectorSize / kMinBufferSize) + 1;

  ConVecBufferBase() {
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      buffers_[i].store(nullptr, std::memory_order_relaxed);
    }
  }
  ConVecBufferBase(ConVecBufferBase&& other) {
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      buffers_[i].store(
          other.buffers_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
      other.buffers_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  ConVecBufferBase& operator=(ConVecBufferBase&& other) {
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      T* cur = buffers_[i].load(std::memory_order_relaxed);
      T* oth = other.buffers_[i].load(std::memory_order_relaxed);

      buffers_[i].store(oth, std::memory_order_relaxed);
      other.buffers_[i].store(cur, std::memory_order_relaxed);
    }
    return *this;
  }

 protected:
  detail::AlignedAtomic<T> buffers_[kMaxBuffers];
};

template <
    typename T,
    size_t kMinBufferSize,
    size_t kMaxVectorSize,
    bool kMakeInline,
    ConcurrentVectorReallocStrategy kStrategy>
class ConVecBuffer : public ConVecBufferBase<T, kMinBufferSize, kMaxVectorSize, kMakeInline> {
 public:
  ConVecBuffer() {
    std::memset(shouldDealloc_, 0, BaseType::kMaxBuffers * sizeof(bool));
  }

  ConVecBuffer(ConVecBuffer&& other) : BaseType(std::move(other)) {
    std::memcpy(shouldDealloc_, other.shouldDealloc_, BaseType::kMaxBuffers * sizeof(bool));
    std::memset(other.shouldDealloc_, 0, BaseType::kMaxBuffers * sizeof(bool));
  }

  ConVecBuffer& operator=(ConVecBuffer&& other) {
    if (&other != this) {
      BaseType::operator=(std::move(other));
      std::swap_ranges(
          other.shouldDealloc_, other.shouldDealloc_ + BaseType::kMaxBuffers, shouldDealloc_);
    }
    return *this;
  }

  detail::AlignedAtomic<T>& operator[](size_t bucket) {
    return this->buffers_[bucket];
  }

  const detail::AlignedAtomic<T>& operator[](size_t bucket) const {
    return this->buffers_[bucket];
  }

  void allocAsNecessary(const BucketInfo& binfo) {
    const size_t indexToCheck = (kStrategy == ConcurrentVectorReallocStrategy::kFullBufferAhead)
        ? 0
        : (kStrategy == ConcurrentVectorReallocStrategy::kHalfBufferAhead
               ? binfo.bucketCapacity / 2
               : binfo.bucketCapacity - 1);

    if (DISPENSO_EXPECT(binfo.bucketIndex == indexToCheck, 0)) {
      if (!this->buffers_[binfo.bucket + 1].load(std::memory_order_acquire)) {
        this->buffers_[binfo.bucket + 1].store(
            cv::alloc<T>(binfo.bucketCapacity << 1), std::memory_order_release);
        shouldDealloc_[binfo.bucket + 1] = true;
      }
    }
    while (DISPENSO_EXPECT(!this->buffers_[binfo.bucket].load(std::memory_order_acquire), 0)) {
    }
  }

  void allocAsNecessary(const BucketInfo& binfo, ssize_t rangeLen, const BucketInfo& bend) {
    const size_t indexToCheck = (kStrategy == ConcurrentVectorReallocStrategy::kFullBufferAhead)
        ? 0
        : (kStrategy == ConcurrentVectorReallocStrategy::kHalfBufferAhead
               ? binfo.bucketCapacity / 2
               : binfo.bucketCapacity - 1);

    bool allocNextBucket =
        binfo.bucketIndex <= indexToCheck && binfo.bucketIndex + rangeLen > indexToCheck;
    if (DISPENSO_EXPECT(allocNextBucket || binfo.bucket < bend.bucket, 0)) {
      size_t sizeToAlloc = 0;

      size_t cap = binfo.bucketCapacity << ((bool)binfo.bucket + !allocNextBucket);
      size_t bucket = binfo.bucket + 1 + !allocNextBucket;
      for (; bucket <= bend.bucket; ++bucket, cap <<= 1) {
        if (!this->buffers_[bucket].load(std::memory_order_acquire)) {
          sizeToAlloc += cap;
        }
      }

      assert(bucket == bend.bucket + 1);
      assert((bucket == 1 && cap == bend.bucketCapacity) || cap == bend.bucketCapacity * 2);

      const size_t endToCheck = (kStrategy == ConcurrentVectorReallocStrategy::kFullBufferAhead)
          ? 0
          : (kStrategy == ConcurrentVectorReallocStrategy::kHalfBufferAhead
                 ? bend.bucketCapacity / 2
                 : bend.bucketCapacity - 1);

      if (DISPENSO_EXPECT(bend.bucketIndex >= endToCheck, 0)) {
        if (!this->buffers_[bucket].load(std::memory_order_acquire)) {
          sizeToAlloc += cap;
        }
      }

      T* allocBufs = nullptr;
      if (sizeToAlloc) {
        allocBufs = cv::alloc<T>(sizeToAlloc);
      }
      bool firstAccounted = false;

      cap = binfo.bucketCapacity << ((bool)binfo.bucket + !allocNextBucket);
      bucket = binfo.bucket + 1 + !allocNextBucket;
      for (; bucket <= bend.bucket; ++bucket, cap <<= 1) {
        if (!this->buffers_[bucket].load(std::memory_order_acquire)) {
          this->buffers_[bucket].store(allocBufs, std::memory_order_release);
          allocBufs += cap;
          shouldDealloc_[bucket] = !firstAccounted;
          firstAccounted = true;
        }
      }

      assert(bucket == bend.bucket + 1);
      assert((bucket == 1 && cap == bend.bucketCapacity) || cap == bend.bucketCapacity * 2);

      if (DISPENSO_EXPECT(bend.bucketIndex >= endToCheck, 0)) {
        if (!this->buffers_[bucket].load(std::memory_order_acquire)) {
          this->buffers_[bucket].store(allocBufs, std::memory_order_release);
          shouldDealloc_[bucket] = !firstAccounted;
        }
      }
    }
    for (size_t bucket = binfo.bucket; bucket <= bend.bucket; ++bucket) {
      while (DISPENSO_EXPECT(!this->buffers_[bucket].load(std::memory_order_acquire), 0)) {
      }
    }
  }

  bool shouldDealloc(size_t bucket) const {
    return shouldDealloc_[bucket];
  }

 private:
  using BaseType = ConVecBufferBase<T, kMinBufferSize, kMaxVectorSize, kMakeInline>;
  bool shouldDealloc_[BaseType::kMaxBuffers];
};

} // namespace cv
