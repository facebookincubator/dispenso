/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file resource_pool.h
 * A file providing ResourcePool.  This is syntactic sugar over what is essentially a set of
 * semaphore guarded resources.
 **/

#pragma once

#include <blockingconcurrentqueue.h>
#include <dispenso/platform.h>
#include <dispenso/tsan_annotations.h>

namespace dispenso {

template <typename T>
class ResourcePool;

/**
 * A RIAA wrapper for a user's type that can manage accessibility and ensures the resource will go
 * back into the ResourcePool upon destruction.
 **/
template <typename T>
class Resource {
 public:
  Resource(Resource&& other) : resource_(other.resource_), pool_(other.pool_) {
    other.resource_ = nullptr;
  }

  Resource& operator=(Resource&& other) {
    if (&other != this) {
      recycle();
      resource_ = other.resource_;
      pool_ = other.pool_;
      other.resource_ = nullptr;
    }
    return *this;
  }

  /**
   * Access the underlying resource object.
   *
   * @return a reference to the resource.
   **/
  T& get() {
    return *resource_;
  }

  ~Resource() {
    recycle();
  }

 private:
  Resource(T* res, ResourcePool<T>* pool) : resource_(res), pool_(pool) {}

  void recycle();

  T* resource_;
  ResourcePool<T>* pool_;

  friend class ResourcePool<T>;
};

/**
 * A pool of resources that can be accessed from multiple threads.  This is akin to a set of
 * resources and a semaphore ensuring enough resources exist.
 **/
template <typename T>
class ResourcePool {
 public:
  /**
   * Construct a ResourcePool.
   *
   * @param size The number of <code>T</code> objects in the pool.
   * @param init A functor with signature T() which can be called to initialize the pool's
   * resources.
   **/
  template <typename F>
  ResourcePool(size_t size, const F& init)
      : pool_(size),
        backingResources_(reinterpret_cast<char*>(
            detail::alignedMalloc(size * detail::alignToCacheLine(sizeof(T))))),
        size_(size) {
    char* buf = backingResources_;

    // There are three reasons we create our own buffer and use placement new:
    // 1. We want to be able to handle non-movable non-copyable objects
    //   * Note that we could do this with std::deque
    // 2. We want to minimize memory allocations, since that can be a common point of contention in
    //    multithreaded programs.
    // 3. We can easily ensure that the objects are cache aligned to help avoid false sharing.

    for (size_t i = 0; i < size; ++i) {
      pool_.enqueue(new (buf) T(init()));
      buf += detail::alignToCacheLine(sizeof(T));
    }
  }

  /**
   * Acquire a resource from the pool.  This function may block until a resource becomes available.
   *
   * @return a <code>Resource</code>-wrapped resource.
   **/
  Resource<T> acquire() {
    T* t;
    pool_.wait_dequeue(t);
    return Resource<T>(t, this);
  }

  /**
   * Destruct the ResourcePool.  The user must ensure that all resources are returned to the pool
   * prior to destroying the pool.
   **/
  ~ResourcePool() {
    assert(pool_.size_approx() == size_);
    for (size_t i = 0; i < size_; ++i) {
      T* t;
      pool_.wait_dequeue(t);
      t->~T();
    }
    detail::alignedFree(backingResources_);
  }

 private:
  void recycle(T* t) {
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_BEGIN();
    pool_.enqueue(t);
    DISPENSO_TSAN_ANNOTATE_IGNORE_WRITES_END();
  }

  moodycamel::BlockingConcurrentQueue<T*> pool_;
  char* backingResources_;
  size_t size_;

  friend class Resource<T>;
};

template <typename T>
void Resource<T>::recycle() {
  if (resource_) {
    pool_->recycle(resource_);
  }
}

} // namespace dispenso
