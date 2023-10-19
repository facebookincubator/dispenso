/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file concurrent_vector.h
 * A file providing a concurrent vector implementation.  The basic implementation is similar to
 * common implementation of deques, with a two-level array structure.  For ConcurrentVector, this is
 * an array of buffers that grow by powers of two.  The interface is intended to be a reasonably
 * complete standin for both std::vector and tbb::concurrent_vector.  When compared to std::vector,
 * it is missing only the .data() accessor, because it is not possible to access the contents as a
 * contiguous buffer.  The interface is also compatible with TBB's concurrent_vector, but provides
 * slightly more functionality to be compatible with std::vector (e.g. .insert() and .erase()), and
 * also to enable higher performance without requiring double-initialization in some grow_by cases
 * (via .grow_by_generator()).  ConcurrentVector also has a reserving Constructor that can allow for
 * better performance when the size (or maximum size, or even a guess at the size) is known ahead of
 * time.
 *
 * Like std::deque, and unlike std::vector, it is possible to use non-movable objects in
 * ConcurrentVector, and references are not invalidated when growing the ConcurrentVector. Iterators
 * are also stable under these conditions.  One other important difference to the std containers,
 * and also to tbb::concurrentvector is that ConcurrentVector does not take an Allocator, but just
 * uses appropriately aligned malloc/free for allocation.
 *
 * Basically speaking, it is possible to grow the ConcurrentVector concurrently (e.g. via
 * .grow_by(), .emplace_back(), etc...) very quickly, and it is safe to iterate ranges of the vector
 * that have already been inserted (and e.g. .begin(), and .end() are thread-safe).  As with TBB's
 * implementation, it is not safe to concurrently call .pop_back(), .reserve(), .resize(), etc...
 * The operation of these functions is lock-free if memory allocation is also lock-free.
 *
 * As of this writing, with the benchmarks developed for testing ConcurrentVector, ConcurrentVector
 * appears to be faster than tbb::concurrent_vector by a factor of between about 15% and 3x
 * (depending on the operation).  ConcurrentVector's iteration and random access is on-par with
 * std::deque (libstdc++), sometimes a bit faster, sometimes a bit slower, but .push_back() is about
 * an order of magnitude slower for serial use (note that by using one of the .grow_by() variants
 * could make this on-par for serial code, if applicable, see the "alternative" benchmarks).
 *
 * Most notably, in the parallel growth benchmarks, (on Clang 8, Linux, 32-core Threadripper
 * 2990WX), ConcurrentVector is between about 5x and 20x faster than std::vector + std::mutex, and
 * is about 1.6x faster than tbb::concurrent_vector.
 **/

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <dispenso/detail/math.h>
#include <dispenso/platform.h>

namespace dispenso {

/**
 * Available strategies to use for ConcurrentVector reallocation.  kFullBufferAhead means that once
 * we begin to use a buffer, we will also ensure that the next buffer is allocated. kHalfBufferAhead
 * means that once we begin to use an address halfway through the current buffer, we will allocate
 * the next buffer.  kAsNeeded means that once we are ready to use an address in the next buffer (we
 * are using the last valid address in the current buffer), we allocate the next buffer.  kAsNeeded
 * should be the default to avoid using too much memory, but a small speedup is possible due to less
 * thread waiting when the other options are used.  kFullBufferAhead is fastest, then
 * kHalfBufferAhead, then kAsNeeded, and the correspondingly kFullBufferAhead uses the most memory,
 * then kHalfBufferAhead, and kAsNeeded uses the least.
 **/
enum class ConcurrentVectorReallocStrategy { kFullBufferAhead, kHalfBufferAhead, kAsNeeded };

struct ReserveTagS {};
/**
 * This ReserveTag can be passed into the reserving constructor for better performance, if even a
 * rough guess can be made at the number of elements that will be required.
 **/
constexpr ReserveTagS ReserveTag;

// Textual inclusion.  Includes undocumented implementation details, e.g. iterators.
#include <dispenso/detail/concurrent_vector_impl.h>

/**
 * The default ConcurrentVector traits type for defining capacities and max capacities.  Both
 * members are required should one wish to supply a custom set of traits.
 **/
template <typename T>
struct DefaultConcurrentVectorSizeTraits {
  /**
   * @brief This is the starting user-expected capacity in number of elements (algorithm may provide
   * more based on kReallocStrategy)
   **/
  static constexpr size_t kDefaultCapacity = (sizeof(T) >= 256) ? 2 : 512 / sizeof(T);

  /**
   * @brief The maximum possible size for the vector.
   *
   * The reason this exists is because if someone doesn't require vectors of length in e.g.
   * petabytes, the class can use less space.  Additionally, some minor optimizations may be
   * possible if the max size is less than 32-bits.
   **/
  static constexpr size_t kMaxVectorSize =
      (size_t{1} << (sizeof(size_t) * CHAR_BIT >= 47 ? 47 : sizeof(size_t) * CHAR_BIT - 1)) /
      sizeof(T);
};

/**
 * The default ConcurrentVector traits type.  All members are required should one wish to supply a
 * custom set of traits.
 **/
struct DefaultConcurrentVectorTraits {
  /**
   * @brief Prefer to place the pointers to the buffers inline in the class.
   *
   * This can consume a lot of space, e.g. on the stack.  This can be a couple of kilobytes.  But
   * performance is improved by 5% to 15%.  When set to false, this results in those pointers
   * residing in a separate heap allocation.
   **/
  static constexpr bool kPreferBuffersInline = true;

  /**
   * @brief How far/whether to allocate before memory is required.
   *
   * We can allocate ahead, which may reduce the chances of another thread blocking while waiting
   * for memory to be allocated.  This implies a pretty stiff memory overhead, which people may not
   * want to pay (up to 3x overhead after the first bucket for kFullBufferAhead).  This can be set
   * to kAsNeeded instead of kFullBufferAhead, which makes overheads similar to typical std::vector
   * implementation (up to 2x overhead).  Performance differences are modest, so unless you need to
   * squeeze the most out of your use, kAsNeeded is okay.
   **/
  static constexpr ConcurrentVectorReallocStrategy kReallocStrategy =
      ConcurrentVectorReallocStrategy::kAsNeeded;

  /**
   * @brief Should we prefer faster, but larger iterators, or slower, but smaller iterators.
   *
   * If an algorithm needs to store iterators for later use, size overhead could become a
   * concern.  By using kIteratorPreferSpeed == true, the size of an iterator will be roughly double
   * the iterator when kIteratorPreferSpeed == false.  Conversely, iterate + dereference is also
   * about twice as fast.
   *
   **/
  static constexpr bool kIteratorPreferSpeed = true;
};

/**
 * A concurrent vector type.  It is safe to call .push_back(), .emplace_back(), the various .grow_()
 * functions, .begin(), .end(), .size(), .empty() concurrently, and existing iterators and
 *references remain valid with these functions' use.
 **/
template <
    typename T,
    typename Traits = DefaultConcurrentVectorTraits,
    typename SizeTraits = DefaultConcurrentVectorSizeTraits<T>>
class ConcurrentVector {
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using size_type = size_t;
  using difference_type = ssize_t;
  using reference_type = T&;
  using const_reference_type = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = std::conditional_t<
      Traits::kIteratorPreferSpeed,
      cv::ConcurrentVectorIterator<ConcurrentVector<T, Traits>, T, false>,
      cv::CompactCVecIterator<ConcurrentVector<T, Traits>, T, false>>;
  using const_iterator = std::conditional_t<
      Traits::kIteratorPreferSpeed,
      cv::ConcurrentVectorIterator<ConcurrentVector<T, Traits>, T, true>,
      cv::CompactCVecIterator<ConcurrentVector<T, Traits>, T, true>>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /**
   * Default construct the ConcurrentVector.
   **/
  ConcurrentVector() : ConcurrentVector(SizeTraits::kDefaultCapacity / 2, ReserveTag) {}

  /**
   * The reserving constructor.  By supplying a reasonable starting capacity, e.g. close to max
   * expected vector size, this can often improve performance substantially by reducing allocations
   * and increasing data coherency.
   **/
  ConcurrentVector(size_t startCapacity, ReserveTagS)
      : firstBucketShift_(detail::log2(
            detail::nextPow2(std::max(startCapacity, SizeTraits::kDefaultCapacity / 2)))),
        firstBucketLen_(size_type{1} << firstBucketShift_) {
    T* firstTwo = cv::alloc<T>(2 * firstBucketLen_);
    buffers_[0].store(firstTwo, std::memory_order_release);
    buffers_[1].store(firstTwo + firstBucketLen_, std::memory_order_release);
  }

  /**
   * Sizing constructor with default initialization.
   **/
  explicit ConcurrentVector(size_t startSize) : ConcurrentVector(startSize, ReserveTag) {
    size_.store(startSize, std::memory_order_relaxed);
    T* buf = buffers_[0].load(std::memory_order_relaxed);
    for (size_t i = 0; i < startSize; ++i) {
      new (buf + i) T();
    }
  }

  /**
   * Sizing constructor with specified default value.
   **/
  ConcurrentVector(size_t startSize, const T& defaultValue)
      : ConcurrentVector(startSize, ReserveTag) {
    size_.store(startSize, std::memory_order_relaxed);
    T* buf = buffers_[0].load(std::memory_order_relaxed);
    for (size_t i = 0; i < startSize; ++i) {
      new (buf + i) T(defaultValue);
    }
  }

  /**
   * Constructor taking an iterator range.
   **/
  template <typename InIterator>
  ConcurrentVector(InIterator start, InIterator end)
      : ConcurrentVector(std::distance(start, end), start, end) {}

  /**
   * Sizing constructor taking an iterator range.  If size is known in advance, this may be faster
   * than just providing the iterator range, especially for input iterators that are not random
   * access.
   **/
  template <typename InIterator>
  ConcurrentVector(size_type startSize, InIterator start, InIterator end)
      : ConcurrentVector(startSize, ReserveTag) {
    size_.store(startSize, std::memory_order_relaxed);
    assert(std::distance(start, end) == static_cast<difference_type>(startSize));
    internalInit(start, end, begin());
  }

  /**
   * Construct via initializer list
   **/
  ConcurrentVector(std::initializer_list<T> l)
      : ConcurrentVector(l.size(), std::begin(l), std::end(l)) {}

  /**
   * Copy constructor
   **/
  ConcurrentVector(const ConcurrentVector& other)
      : ConcurrentVector(other.size(), other.cbegin(), other.cend()) {}

  /**
   * Move constructor
   **/
  ConcurrentVector(ConcurrentVector&& other)
      : buffers_(std::move(other.buffers_)),
        firstBucketShift_(other.firstBucketShift_),
        firstBucketLen_(other.firstBucketLen_),
        size_(other.size_.load(std::memory_order_relaxed)) {
    other.size_.store(0, std::memory_order_relaxed);
    // This is possibly unnecessary overhead, but enables the "other" vector to be in a valid,
    // usable state right away, no empty check or clear required, as it is for std::vector.
    T* firstTwo = cv::alloc<T>(2 * firstBucketLen_);
    other.buffers_[0].store(firstTwo, std::memory_order_relaxed);
    other.buffers_[1].store(firstTwo + firstBucketLen_, std::memory_order_relaxed);
  }

  /**
   * Copy assignment operator.  This is not concurrency safe.
   **/
  ConcurrentVector& operator=(const ConcurrentVector& other) {
    if (&other == this) {
      return *this;
    }

    clear();
    reserve(other.size());
    size_.store(other.size(), std::memory_order_relaxed);
    internalInit(other.cbegin(), other.cend(), begin());

    return *this;
  }

  /**
   * Move assignment operator.  This is not concurrency safe.
   **/
  ConcurrentVector& operator=(ConcurrentVector&& other) {
    using std::swap;
    if (&other == this) {
      return *this;
    }

    clear();
    swap(firstBucketShift_, other.firstBucketShift_);
    swap(firstBucketLen_, other.firstBucketLen_);
    buffers_ = std::move(other.buffers_);
    size_t curLen = size_.load(std::memory_order_relaxed);
    size_.store(other.size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    other.size_.store(curLen, std::memory_order_relaxed);

    return *this;
  }

  /**
   * Assign the vector.  Not concurrency safe.
   *
   * @param count The number of elements to have in the vector
   * @param value The value to copy into each element
   **/
  void assign(size_type count, const T& value) {
    clear();
    reserve(count);
    size_.store(count, std::memory_order_relaxed);
    internalFillN(begin(), count, value);
  }

  /**
   * Assign the vector.  Not concurrency safe.
   *
   * @param start The beginning of the iterator range to assign into the vector
   * @param end The end of the iterator range to assign into the vector
   **/
  template <
      typename It,
      typename = typename std::iterator_traits<It>::difference_type,
      typename = typename std::iterator_traits<It>::pointer,
      typename = typename std::iterator_traits<It>::reference,
      typename = typename std::iterator_traits<It>::value_type,
      typename = typename std::iterator_traits<It>::iterator_category>
  void assign(It start, It end) {
    clear();
    auto count = std::distance(start, end);
    reserve(count);
    size_.store(count, std::memory_order_relaxed);
    internalInit(start, end, begin());
  }

  /**
   * Reserve some capacity for the vector.  Not concurrency safe.
   *
   * @param capacity The amount of space to ensure is available to avoid further allocations.
   **/
  void reserve(difference_type capacity) {
    buffers_.allocAsNecessary({0, 0, firstBucketLen_}, capacity, bucketAndSubIndex(capacity));
  }

  /**
   * Resize the vector, using default initialization for any new values.  Not concurrency safe.
   *
   * @param len The length of the vector after the resize.
   **/
  void resize(difference_type len) {
    difference_type curLen = static_cast<difference_type>(size_.load(std::memory_order_relaxed));
    if (curLen < len) {
      grow_to_at_least(len);
    } else if (curLen > len) {
      auto it = end();
      auto newEnd = begin() + len;
      do {
        --it;
        it->~T();
      } while (it != newEnd);
      size_.store(len, std::memory_order_relaxed);
    }
  }

  /**
   * Resize the vector, copying the provided value into any new elements.  Not concurrency safe.
   *
   * @param len The length of the vector after the resize.
   * @param value The value to copy into any new elements.
   **/
  void resize(difference_type len, const T& value) {
    difference_type curLen = static_cast<difference_type>(size_.load(std::memory_order_relaxed));
    if (curLen < len) {
      grow_to_at_least(len, value);
    } else if (curLen > len) {
      auto it = end();
      auto newEnd = begin() + len;
      do {
        --it;
        it->~T();
      } while (it != newEnd);
      size_.store(len, std::memory_order_relaxed);
    }
  }

  /**
   * The default capacity of this vector.
   * @return The capacity if the vector were cleared and then called shrink_to_fit.
   **/
  size_type default_capacity() const {
    return 2 * firstBucketLen_;
  }

  /**
   * The current capacity of the vector.  Not concurrency safe.
   * @return The current capacity.
   **/
  size_type capacity() const {
    size_t cap = 2 * firstBucketLen_;
    for (size_t b = 2; b < kMaxBuffers; ++b) {
      if (!buffers_[b].load(std::memory_order_relaxed)) {
        break;
      }
      cap *= 2;
    }
    return cap;
  }

  /**
   * Clear the vector.  This does not deallocate buffers, but just ensures that all elements are
   * destructed.  Size will be zero after the call.  Not concurrency safe.
   **/
  void clear() {
    auto binfo = bucketAndSubIndex(size_.load(std::memory_order_relaxed));
    size_t len = binfo.bucketIndex;
    size_t cap = binfo.bucketCapacity;
    size_t b = binfo.bucket;
    do {
      T* buf = buffers_[b].load(std::memory_order_relaxed);
      T* t = buf + len;

      while (t != buf) {
        --t;
        t->~T();
      }
      cap >>= int{b > 1};
      len = cap;
    } while (b--);
    size_.store(0, std::memory_order_release);
  }

  /**
   * Gets rid of extra capacity that is not needed to maintain preconditions.  At least the default
   * capacity will remain, even if the size of the vector is zero.  Not concurrency safe.
   **/
  void shrink_to_fit() {
    constexpr size_t kMaxExtra = 2;
    auto binfo = bucketAndSubIndex(size_.load(std::memory_order_relaxed));

    // We need to at least skip the first two buckets, since we have those as a single allocation.
    size_t startBucket = std::max<size_t>(2, binfo.bucket + kMaxExtra);

    for (size_t b = startBucket; b < kMaxBuffers; ++b) {
      T* ptr = buffers_[b].load(std::memory_order_relaxed);
      if (!ptr) {
        break;
      }
      if (buffers_.shouldDealloc(b)) {
        cv::dealloc<T>(ptr);
      }
      buffers_[b].store(nullptr, std::memory_order_release);
    }
  }

  /**
   * Destruct the vector.
   **/
  ~ConcurrentVector() {
    clear();
    shrink_to_fit();
    cv::dealloc<T>(buffers_[0].load(std::memory_order_acquire));
  }

  /**
   * Insert a value. Not concurrency safe.
   * @param pos The point of insertion.
   * @param value The value to copy into the element at pos.
   * @return The iterator at the inserted position.
   **/
  iterator insert(const_iterator pos, const T& value) {
    auto it = insertPartial(pos);
    new (&*it) T(value);
    return it;
  }

  /**
   * Insert a value.  Not concurrency safe.
   * @param pos The point of insertion.
   * @param value The value to move into the element at pos.
   * @return The iterator at the inserted position.
   **/
  iterator insert(const_iterator pos, T&& value) {
    auto it = insertPartial(pos);
    new (&*it) T(std::move(value));
    return it;
  }

  /**
   * Insert a value. Not concurrency safe.
   * @param pos The point of insertion.
   * @param count The number of elements to insert.
   * @param value The value to copy into each inserted element starting at pos.
   * @return The iterator at the inserted position.
   **/
  iterator insert(const_iterator pos, size_type count, const T& value) {
    auto it = insertPartial(pos, count);
    std::fill_n(it, count, value);
    return it;
  }

  /**
   * Insert a range of values. Not concurrency safe.
   * @param pos The point of insertion.
   * @param first The start of the input iterator range.
   * @param last The end of the input iterator range.
   * @return The iterator at the inserted position.
   **/
  template <
      typename InputIt,
      typename = typename std::iterator_traits<InputIt>::difference_type,
      typename = typename std::iterator_traits<InputIt>::pointer,
      typename = typename std::iterator_traits<InputIt>::reference,
      typename = typename std::iterator_traits<InputIt>::value_type,
      typename = typename std::iterator_traits<InputIt>::iterator_category>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    size_t len = std::distance(first, last);
    auto it = insertPartial(pos, len);
    std::copy_n(first, len, it);
    return it;
  }

  /**
   * Insert an initializer_list. Not concurrency safe.
   * @param pos The point of insertion.
   * @param ilist The initializer_list.
   * @return The iterator at the inserted position.
   **/
  iterator insert(const_iterator pos, std::initializer_list<T> ilist) {
    return insert(pos, ilist.begin(), ilist.end());
  }

  /**
   * Erase one element.  Not concurrency safe.
   * @param pos The point of erasure.
   * @return Iterator following the removed element. If pos refers to the last element, then
   * the end() iterator is returned.
   **/
  iterator erase(const_iterator pos) {
    auto e = end();
    if (e == pos) {
      return e;
    }
    size_.fetch_sub(1, std::memory_order_relaxed);
    --e;
    if (e == pos) {
      e->~T();
      return e;
    }
    ++e;
    auto it = begin();
    it += (pos - it);
    return std::move(pos + 1, const_iterator(e), it);
  }

  /**
   * Erase a range of elements.  Not concurrency safe.
   * @param first The starting point of erasure.
   * @param last The ending point of erasure.
   * @return Iterator following the last removed element. If pos refers to the last element, then
   * the end() iterator is returned. If last==end() prior to removal, then the updated end()
   * iterator is returned. If [first, last) is an empty range, then last is returned.
   **/
  iterator erase(const_iterator first, const_iterator last) {
    size_t len = std::distance(first, last);
    auto it = begin();
    size_t startIdx = first - it;
    if (len == 0) {
      return it + (startIdx + len);
    }
    it += startIdx;

    auto e_it = std::move(last, cend(), it);

    if (e_it < last) {
      // remove any values that were not already moved into
      do {
        --last;
        last->~T();
      } while (e_it != last);
    }
    size_.fetch_sub(len, std::memory_order_relaxed);
    return e_it;
  }

  /**
   * Push a value onto the end of the vector.  Concurrency safe.
   * @param val The value to copy into the new element.
   * @return The iterator at the point of insertion.
   **/
  iterator push_back(const T& val) {
    return emplace_back(val);
  }

  /**
   * Push a value onto the end of the vector.  Concurrency safe.
   * @param val The value to move into the new element.
   * @return The iterator at the point of insertion.
   **/
  iterator push_back(T&& val) {
    return emplace_back(std::move(val));
  }

  /**
   * Push a value onto the end of the vector.  Concurrency safe.
   * @param args The arg pack used to construct the new element.
   * @return The iterator at the point of insertion.
   **/
  template <typename... Args>
  iterator emplace_back(Args&&... args) {
    auto index = size_.fetch_add(1, std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(index);

    buffers_.allocAsNecessary(binfo);

    iterator ret{this, index, binfo};

    if (Traits::kIteratorPreferSpeed) {
      new (&*ret) T(std::forward<Args>(args)...);
    } else {
      new (buffers_[binfo.bucket] + binfo.bucketIndex) T(std::forward<Args>(args)...);
    }

    return ret;
  }

  /**
   * Grow the vector, constructing new elements via a generator.  Concurrency safe.
   * @param delta The number of elements to grow by.
   * @param gen The generator to use to construct new elements.  Must have operator()() and return a
   * valid T.
   * @return The iterator to the start of the grown range.
   **/
  template <typename Gen>
  iterator grow_by_generator(size_type delta, Gen gen) {
    iterator ret = growByUninitialized(delta);
    for (auto it = ret; delta--; ++it) {
      new (&*it) T(gen());
    }
    return ret;
  }

  /**
   * Grow the vector, copying the provided value into new elements.  Concurrency safe.
   * @param delta The number of elements to grow by.
   * @param t The value to copy into all new elements.
   * @return The iterator to the start of the grown range.
   **/
  iterator grow_by(size_type delta, const T& t) {
    iterator ret = growByUninitialized(delta);
    internalFillN(ret, delta, t);
    return ret;
  }

  /**
   * Grow the vector, default initializing new elements.  Concurrency safe.
   * @param delta The number of elements to grow by.
   * @return The iterator to the start of the grown range.
   **/
  iterator grow_by(size_type delta) {
    iterator ret = growByUninitialized(delta);
    internalFillDefaultN(ret, delta);
    return ret;
  }

  /**
   * Grow the vector with an input iterator range.  Concurrency safe.
   * @param start The start of the input iterator range.
   * @param end The end of the input iterator range.
   * @return The iterator to the start of the grown range.
   **/
  template <
      typename It,
      typename = typename std::iterator_traits<It>::difference_type,
      typename = typename std::iterator_traits<It>::pointer,
      typename = typename std::iterator_traits<It>::reference,
      typename = typename std::iterator_traits<It>::value_type,
      typename = typename std::iterator_traits<It>::iterator_category>
  iterator grow_by(It start, It end) {
    iterator ret = growByUninitialized(std::distance(start, end));
    internalInit(start, end, ret);
    return ret;
  }

  /**
   * Grow the vector with an initializer_list.  Concurrency safe.
   * @param initList The initializer_list to use to initialize the newly added range.
   * @return The iterator to the start of the grown range.
   **/
  iterator grow_by(std::initializer_list<T> initList) {
    return grow_by(std::begin(initList), std::end(initList));
  }

  /**
   * Grow the vector to at least the desired size, default initializing new elements.  Concurrency
   * safe.
   * @param n The required length
   * @return The iterator to the nth element.
   **/
  iterator grow_to_at_least(size_type n) {
    size_t curSize = size_.load(std::memory_order_relaxed);
    if (curSize < n) {
      return grow_by(n - curSize);
    }
    return {this, n - 1, bucketAndSubIndex(n - 1)};
  }

  /**
   * Grow the vector to at least the desired size, copying the provided value into new elements.
   * Concurrency safe.
   * @param n The required length
   * @param t The value to copy into each new element.
   * @return The iterator to the nth element.
   **/
  iterator grow_to_at_least(size_type n, const T& t) {
    size_t curSize = size_.load(std::memory_order_relaxed);
    if (curSize < n) {
      return grow_by(n - curSize, t);
    }
    return {this, n - 1, bucketAndSubIndex(n - 1)};
  }

  /**
   * Pop the last element off the vector.  Not concurrency safe.
   **/
  void pop_back() {
    auto binfo = bucketAndSubIndex(size_.fetch_sub(1, std::memory_order_relaxed) - 1);
    T* elt = buffers_[binfo.bucket].load(std::memory_order_relaxed) + binfo.bucketIndex;
    elt->~T();
  }

  /**
   * Access an element of the vector.  Concurrency safe.
   * @param index The index of the element to access.
   * @return A reference to the element at index.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const T& operator[](size_type index) const {
    auto binfo = bucketAndSubIndexForIndex(index);
    T* buf = buffers_[binfo.bucket].load(std::memory_order_relaxed);
    return buf[binfo.bucketIndex];
  }

  /**
   * Access an element of the vector.  Concurrency safe.
   * @param index The index of the element to access.
   * @return A reference to the element at index.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  T& operator[](size_type index) {
    auto binfo = bucketAndSubIndexForIndex(index);
    T* buf = buffers_[binfo.bucket].load(std::memory_order_relaxed);
    return buf[binfo.bucketIndex];
  }

  /**
   * Access an element of the vector.  Out-of-boundes accesses generate an exception.  Concurrency
   * safe.
   * @param index The index of the element to access.
   * @return A reference to the element at index.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const T& at(size_type index) const {
#if defined(__cpp_exceptions)
    if (index >= size_.load(std::memory_order_relaxed)) {
      throw std::out_of_range("Index too large");
    }
#endif
    return operator[](index);
  }

  /**
   * Access an element of the vector.  Out-of-boundes accesses generate an exception.  Concurrency
   * safe.
   * @param index The index of the element to access.
   * @return A reference to the element at index.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  T& at(size_type index) {
#if defined(__cpp_exceptions)
    if (index >= size_.load(std::memory_order_relaxed)) {
      throw std::out_of_range("Index too large");
    }
#endif
    return operator[](index);
  }

  /**
   * Get an iterator to the start of the vector.  Concurrency safe.
   * @return An iterator to the start of the vector.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  iterator begin() {
    return {this, 0, {0, 0, firstBucketLen_}};
  }

  /**
   * Get an iterator to the end of the vector.  Concurrency safe.
   * @return An iterator to the end of the vector.  Note that it is possible for the end to change
   * concurrently with this call.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  iterator end() {
    size_t curSize = size_.load(std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(curSize);
    return {this, curSize, binfo};
  }

  /**
   * Get an iterator to the start of the vector.  Concurrency safe.
   * @return An iterator to the start of the vector.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_iterator begin() const {
    return {this, 0, {0, 0, firstBucketLen_}};
  }

  /**
   * Get an iterator to the end of the vector.  Concurrency safe.
   * @return An iterator to the end of the vector.  Note that it is possible for the end to change
   * concurrently with this call.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_iterator end() const {
    size_t curSize = size_.load(std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(curSize);
    return {this, curSize, binfo};
  }

  /**
   * Get an iterator to the start of the vector.  Concurrency safe.
   * @return An iterator to the start of the vector.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_iterator cbegin() const {
    return {this, 0, {0, 0, firstBucketLen_}};
  }

  /**
   * Get an iterator to the end of the vector.  Concurrency safe.
   * @return An iterator to the end of the vector.  Note that it is possible for the end to change
   * concurrently with this call.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_iterator cend() const {
    size_t curSize = size_.load(std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(curSize);
    return {this, curSize, binfo};
  }

  /**
   * Get a starting reverse iterator to the vector.  Concurrency safe.
   * @return A starting reverse iterator to the vector. Note that it is possible for the the rbegin
   * to change concurrently with this call.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  /**
   * Get an ending reverse iterator to the vector.  Concurrency safe.
   * @return An ending reverse iterator to the vector.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  /**
   * Get a starting reverse iterator to the vector.  Concurrency safe.
   * @return A starting reverse iterator to the vector. Note that it is possible for the the rbegin
   * to change concurrently with this call.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(cend());
  }

  /**
   * Get an ending reverse iterator to the vector.  Concurrency safe.
   * @return An ending reverse iterator to the vector.
   *
   * @note Iterators and references are stable even if other threads are inserting, but it is
   * still the users responsibility to avoid racing on the element itself.
   **/
  const_reverse_iterator rend() const {
    return const_reverse_iterator(cbegin());
  }

  /**
   * Checks if the vector contains any elements. Concurrency safe.
   * @return true if the vector contains no elements, false otherwise.  Note that an element could
   * be appended concurrently with this call.
   **/
  bool empty() const {
    return size_.load(std::memory_order_relaxed) == 0;
  }

  /**
   * Get the maximum size a vector of this type can theoretically have.  Concurrency safe
   * (constexpr).
   * @return The max size a vector of this type could theoretically have.
   **/
  constexpr size_type max_size() const noexcept {
    return Traits::kMaxVectorSize;
  }

  /**
   * Get the size of the vector.  Concurrency safe.
   * @return The number of elements in the vector.  Note that elements can be appended concurrently
   * with this call.
   **/
  size_type size() const {
    return size_.load(std::memory_order_relaxed);
  }

  /**
   * Get the front element of the vector.  Concurrency safe.
   * @return A reference to the first element in the vector.
   **/
  T& front() {
    return *buffers_[0].load(std::memory_order_relaxed);
  }

  /**
   * Get the front element of the vector.  Concurrency safe.
   * @return A reference to the first element in the vector.
   **/
  const T& front() const {
    return *buffers_[0].load(std::memory_order_relaxed);
  }

  /**
   * Get the last element of the vector.  Concurrency safe.
   * @return A reference to the last element in the vector.  Note that elements could be appended
   * concurrently with this call (in which case it won't be the back anymore).
   **/
  T& back() {
    return *(end() - 1);
  }

  /**
   * Get the last element of the vector.  Concurrency safe.
   * @return A reference to the last element in the vector.  Note that elements could be appended
   * concurrently with this call (in which case it won't be the back anymore).
   **/
  const T& back() const {
    return *(end() - 1);
  }

  /**
   * Swap the contents (and iterators, and element references) of the current vector with oth.
   * @param oth The vector to swap with.
   **/
  void swap(ConcurrentVector& oth) {
    using std::swap;
    swap(firstBucketShift_, oth.firstBucketShift_);
    swap(firstBucketLen_, oth.firstBucketLen_);
    size_t othlen = oth.size_.load(std::memory_order_relaxed);
    oth.size_.store(size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    size_.store(othlen, std::memory_order_relaxed);

    // okay, this relies on the fact that we're essentially swapping in the move operator.
    buffers_ = std::move(oth.buffers_);
  }

 private:
  DISPENSO_INLINE cv::BucketInfo bucketAndSubIndexForIndex(size_t index) const {
#if defined(__clang__)
    if (index < firstBucketLen_) {
      return {0, index, firstBucketLen_};
    }

    size_t l2idx = detail::log2(index);
    size_t bucket = (l2idx + 1) - firstBucketShift_;
    size_t bucketCapacity = size_t{1} << l2idx;
    size_t bucketIndex = index - bucketCapacity;

    return {bucket, bucketIndex, bucketCapacity};
#else
    return bucketAndSubIndex(index);
#endif // __clang__
  }

  DISPENSO_INLINE cv::BucketInfo bucketAndSubIndex(size_t index) const {
    size_t l2idx = detail::log2(index | 1);
    size_t bucket = (l2idx + 1) - firstBucketShift_;
    size_t bucketCapacity = size_t{1} << l2idx;
    size_t bucketIndex = index - bucketCapacity;

    bucket = index < firstBucketLen_ ? 0 : bucket;
    bucketIndex = index < firstBucketLen_ ? index : bucketIndex;
    bucketCapacity = index < firstBucketLen_ ? firstBucketLen_ : bucketCapacity;

    return {bucket, bucketIndex, bucketCapacity};
  }

  template <typename InIterator>
  void internalInit(InIterator start, InIterator end, iterator it) {
    while (start != end) {
      new (&*it) T(*start);
      ++it;
      ++start;
    }
  }

  void internalFillN(iterator it, size_t len, const T& value) {
    for (; len--; ++it) {
      new (&*it) T(value);
    }
  }

  void internalFillDefaultN(iterator it, size_t len) {
    for (; len--; ++it) {
      new (&*it) T();
    }
  }

  iterator growByUninitialized(size_type delta) {
    auto index = size_.fetch_add(delta, std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(index);
    buffers_.allocAsNecessary(binfo, delta, bucketAndSubIndex(index + delta));
    return {this, index, binfo};
  }

  iterator insertPartial(const_iterator pos) {
    auto e = end();
    auto index = size_.fetch_add(1, std::memory_order_relaxed);
    auto binfo = bucketAndSubIndex(index);
    buffers_.allocAsNecessary(binfo);
    new (&*e) T();
    return std::move_backward(pos, const_iterator(e), e + 1) - 1;
  }

  iterator insertPartial(const_iterator pos, size_t len) {
    auto e = end();
    auto index = size_.fetch_add(len, std::memory_order_relaxed);
    buffers_.allocAsNecessary(bucketAndSubIndex(index), len, bucketAndSubIndex(index + len));
    for (auto it = e + len; it != e;) {
      --it;
      new (&*it) T();
    }
    return std::move_backward(pos, const_iterator(e), e + len) - len;
  }

  alignas(kCacheLineSize) cv::ConVecBuffer<
      T,
      SizeTraits::kDefaultCapacity / 2,
      SizeTraits::kMaxVectorSize,
      Traits::kPreferBuffersInline,
      Traits::kReallocStrategy> buffers_;
  static constexpr size_t kMaxBuffers = decltype(buffers_)::kMaxBuffers;

  size_t firstBucketShift_;
  size_t firstBucketLen_;

  alignas(kCacheLineSize) std::atomic<size_t> size_{0};

  friend class cv::ConVecIterBase<ConcurrentVector<T, Traits>, T>;
  friend class cv::ConcurrentVectorIterator<ConcurrentVector<T, Traits>, T, false>;
  friend class cv::ConcurrentVectorIterator<ConcurrentVector<T, Traits>, T, true>;
};

template <typename T, class Traits1, class Traits2>
inline bool operator==(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <typename T, class Traits1, class Traits2>
inline bool operator!=(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return !(a == b);
}

template <typename T, class Traits1, class Traits2>
inline bool operator<(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <typename T, class Traits1, class Traits2>
inline bool operator>(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return b < a;
}

template <typename T, class Traits1, class Traits2>
inline bool operator<=(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return !(b < a);
}

template <typename T, class Traits1, class Traits2>
inline bool operator>=(
    const ConcurrentVector<T, Traits1>& a,
    const ConcurrentVector<T, Traits2>& b) {
  return !(a < b);
}

template <typename T, class Traits>
inline void swap(ConcurrentVector<T, Traits>& a, ConcurrentVector<T, Traits>& b) {
  a.swap(b);
}

// Textual inclusion.  Includes undocumented implementation details, e.g. iterators.
#include <dispenso/detail/concurrent_vector_impl2.h>

} // namespace dispenso
