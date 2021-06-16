// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

// This file intended for textual inclusion into concurrent_vector.h only

namespace cv {

template <typename VecT, typename T>
DISPENSO_INLINE ConVecIterBase<VecT, T>::ConVecIterBase(const VecT* vec, cv::BucketInfo info)
    : vb_(reinterpret_cast<uintptr_t>(vec) | info.bucket),
      bucketStart_(vec->buffers_[info.bucket].load(std::memory_order_relaxed)),
      bucketPtr_(bucketStart_ + info.bucketIndex),
      bucketEnd_(bucketStart_ + info.bucketCapacity) {}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE ConcurrentVectorIterator<VecT, T, kIsConst>&
ConcurrentVectorIterator<VecT, T, kIsConst>::operator++() {
  ++bucketPtr_;
  if (bucketPtr_ == bucketEnd_) {
    auto len = bucketEnd_ - bucketStart_;
    ++vb_;
    auto vb = getVecAndBucket();
    len <<= int{vb.bucket > 1};
    bucketPtr_ = bucketStart_ = vb.vec->buffers_[vb.bucket].load(std::memory_order_relaxed);
    bucketEnd_ = bucketPtr_ + len;
  }
  return *this;
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE ConcurrentVectorIterator<VecT, T, kIsConst>&
ConcurrentVectorIterator<VecT, T, kIsConst>::operator--() {
  --bucketPtr_;
  if (bucketPtr_ < bucketStart_) {
    auto vb = getVecAndBucket();
    if (vb.bucket) {
      auto len = bucketEnd_ - bucketStart_;
      --vb_;
      len >>= int{vb.bucket > 1};
      bucketStart_ = vb.vec->buffers_[vb.bucket - 1].load(std::memory_order_relaxed);
      bucketPtr_ = bucketStart_ + len;
      bucketEnd_ = bucketPtr_;
      --bucketPtr_;
    }
  }
  return *this;
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename ConcurrentVectorIterator<VecT, T, kIsConst>::reference
ConcurrentVectorIterator<VecT, T, kIsConst>::operator*() const {
  return *bucketPtr_;
}
template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename ConcurrentVectorIterator<VecT, T, kIsConst>::pointer
ConcurrentVectorIterator<VecT, T, kIsConst>::operator->() const {
  return &operator*();
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename ConcurrentVectorIterator<VecT, T, kIsConst>::reference
ConcurrentVectorIterator<VecT, T, kIsConst>::operator[](difference_type n) const {
  T* nPtr = bucketPtr_ + n;
  if (nPtr >= bucketStart_ && nPtr < bucketEnd_) {
    return *nPtr;
  }

  auto vb = getVecAndBucket();

  // Reconstruct index
  ssize_t oldIndex = bucketPtr_ - bucketStart_;
  oldIndex += (bool)vb.bucket * (bucketEnd_ - bucketStart_);
  auto binfo = vb.vec->bucketAndSubIndexForIndex(oldIndex + n);
  return *(vb.vec->buffers_[binfo.bucket].load(std::memory_order_relaxed) + binfo.bucketIndex);
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE ConcurrentVectorIterator<VecT, T, kIsConst>&
ConcurrentVectorIterator<VecT, T, kIsConst>::operator+=(difference_type n) {
  T* nPtr = bucketPtr_ + n;
  if (nPtr >= bucketStart_ && nPtr < bucketEnd_) {
    bucketPtr_ = nPtr;
    return *this;
  }

  auto vb = getVecAndBucket();

  // Reconstruct index
  ssize_t oldIndex = bucketPtr_ - bucketStart_;
  oldIndex += (bool)vb.bucket * (bucketEnd_ - bucketStart_);
  auto binfo = vb.vec->bucketAndSubIndexForIndex(oldIndex + n);
  bucketStart_ = vb.vec->buffers_[binfo.bucket].load(std::memory_order_relaxed);
  bucketEnd_ = bucketStart_ + binfo.bucketCapacity;
  bucketPtr_ = bucketStart_ + binfo.bucketIndex;
  vb_ = reinterpret_cast<uintptr_t>(vb.vec) | binfo.bucket;
  return *this;
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE ConcurrentVectorIterator<VecT, T, kIsConst>
ConcurrentVectorIterator<VecT, T, kIsConst>::operator+(difference_type n) const {
  T* nPtr = bucketPtr_ + n;
  if (nPtr >= bucketStart_ && nPtr < bucketEnd_) {
    return {vb_, bucketStart_, nPtr, bucketEnd_};
  }

  auto vb = getVecAndBucket();
  // Reconstruct index
  ssize_t oldIndex = bucketPtr_ - bucketStart_;
  oldIndex += (bool)vb.bucket * (bucketEnd_ - bucketStart_);
  auto binfo = vb.vec->bucketAndSubIndexForIndex(oldIndex + n);
  return {vb.vec, binfo};
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename CompactCVecIterator<VecT, T, kIsConst>::reference
CompactCVecIterator<VecT, T, kIsConst>::operator*() const {
  return const_cast<VecT&>(*vec_)[index_];
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename CompactCVecIterator<VecT, T, kIsConst>::pointer
CompactCVecIterator<VecT, T, kIsConst>::operator->() const {
  return &operator*();
}

template <typename VecT, typename T, bool kIsConst>
DISPENSO_INLINE typename CompactCVecIterator<VecT, T, kIsConst>::reference
CompactCVecIterator<VecT, T, kIsConst>::operator[](ssize_t n) const {
  return const_cast<VecT&>(*vec_)[index_ + n];
}
} // namespace cv
