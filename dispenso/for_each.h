/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file for_each.h
 * @ingroup group_parallel
 * Functions for performing parallel for_each over iterables.  This intends to more-or-less
 * mimic std::for_each, with a possible (Concurrent)TaskSet passed in for external wait capability,
 * and ForEachOptions for controlling the wait behavior and limiting of parallelism.
 **/

#pragma once

#include <algorithm>
#include <iterator>

#include <dispenso/detail/per_thread_info.h>
#include <dispenso/small_vector.h>
#include <dispenso/task_set.h>

namespace dispenso {

#if DISPENSO_HAS_CONCEPTS
/**
 * @concept ForEachFunc
 * @brief A callable suitable for for_each operations.
 *
 * The callable must be invocable with a reference to the iterator's value type.
 **/
template <typename F, typename Iter>
concept ForEachFunc = std::invocable<F, decltype(*std::declval<Iter>())>;
#endif // DISPENSO_HAS_CONCEPTS

/**
 * A set of options to control for_each
 **/
struct ForEachOptions {
  /**
   * The maximum number of threads to use.  This can be used to limit the number of threads below
   * the number associated with the TaskSet's thread pool to control the degree of concurrency.
   * Setting maxThreads to zero or one will result in serial operation.
   **/
  uint32_t maxThreads = std::numeric_limits<int32_t>::max();
  /**
   * Specify whether the return of the for_each signifies the work is complete.  If the
   * for_each is initiated without providing a TaskSet, the for_each will always wait.
   *
   * @note If wait is true, the calling thread will always participate in computation.  If this is
   * not desired, pass wait as false, and wait manually outside of the for_each on the passed
   * TaskSet.
   **/
  bool wait = true;
};

namespace detail {

// Random-access iterators: compute chunk boundaries arithmetically, avoiding SmallVector.
template <typename TaskSetT, typename Iter, typename F>
void for_each_n_schedule(
    TaskSetT& tasks,
    Iter start,
    F&& f,
    ssize_t numThreads,
    size_t chunkSize,
    ssize_t transitionIdx,
    size_t smallChunkSize,
    const ForEachOptions& options,
    std::random_access_iterator_tag) {
  ssize_t numToSchedule = options.wait ? numThreads - 1 : numThreads;

  if (numToSchedule > 0) {
    tasks.scheduleBulk(
        static_cast<size_t>(numToSchedule),
        [start, &f, chunkSize, smallChunkSize, transitionIdx](size_t idx) {
          ssize_t sidx = static_cast<ssize_t>(idx);
          ssize_t offset;
          ssize_t thisChunkSize;
          if (sidx < transitionIdx) {
            offset = sidx * static_cast<ssize_t>(chunkSize);
            thisChunkSize = static_cast<ssize_t>(chunkSize);
          } else {
            offset = transitionIdx * static_cast<ssize_t>(chunkSize) +
                (sidx - transitionIdx) * static_cast<ssize_t>(smallChunkSize);
            thisChunkSize = static_cast<ssize_t>(smallChunkSize);
          }
          Iter s = start + offset;
          Iter e = s + thisChunkSize;
          return [s, e, f]() {
            auto recurseInfo = PerPoolPerThreadInfo::parForRecurse();
            for (Iter it = s; it != e; ++it) {
              f(*it);
            }
          };
        });
  }

  if (options.wait) {
    ssize_t lastIdx = numThreads - 1;
    ssize_t offset;
    ssize_t thisChunkSize;
    if (lastIdx < transitionIdx) {
      offset = lastIdx * static_cast<ssize_t>(chunkSize);
      thisChunkSize = static_cast<ssize_t>(chunkSize);
    } else {
      offset = transitionIdx * static_cast<ssize_t>(chunkSize) +
          (lastIdx - transitionIdx) * static_cast<ssize_t>(smallChunkSize);
      thisChunkSize = static_cast<ssize_t>(smallChunkSize);
    }
    Iter lastStart = start + offset;
    Iter lastEnd = lastStart + thisChunkSize;
    for (Iter it = lastStart; it != lastEnd; ++it) {
      f(*it);
    }
    tasks.wait();
  }
}

// Non-random-access iterators: pre-compute boundary iterators into SmallVector.
template <typename TaskSetT, typename Iter, typename F, typename IterCategory>
void for_each_n_schedule(
    TaskSetT& tasks,
    Iter start,
    F&& f,
    ssize_t numThreads,
    size_t chunkSize,
    ssize_t transitionIdx,
    size_t smallChunkSize,
    const ForEachOptions& options,
    IterCategory) {
  SmallVector<Iter, 64> boundaries;
  boundaries.reserve(static_cast<size_t>(numThreads) + 1);
  boundaries.push_back(start);
  for (ssize_t t = 0; t < numThreads; ++t) {
    size_t cs = (t < transitionIdx) ? chunkSize : smallChunkSize;
    Iter next = boundaries[t];
    std::advance(next, cs);
    boundaries.push_back(next);
  }

  ssize_t numToSchedule = options.wait ? numThreads - 1 : numThreads;

  if (numToSchedule > 0) {
    tasks.scheduleBulk(static_cast<size_t>(numToSchedule), [&boundaries, &f](size_t idx) {
      return [s = boundaries[idx], e = boundaries[idx + 1], f]() {
        auto recurseInfo = PerPoolPerThreadInfo::parForRecurse();
        for (Iter it = s; it != e; ++it) {
          f(*it);
        }
      };
    });
  }

  if (options.wait) {
    for (Iter it = boundaries[numThreads - 1]; it != boundaries[numThreads]; ++it) {
      f(*it);
    }
    tasks.wait();
  }
}

} // namespace detail

/**
 * A function like std::for_each_n, but where the function is invoked in parallel across the passed
 * range.
 *
 * @param tasks The task set to schedule the for_each on.
 * @param start The iterator for the start of the range.
 * @param n The length of the range.
 * @param f The function to execute in parallel.  This is a unary function that must be capable of
 * taking dereference of Iter.
 * @param options See ForEachOptions for details.
 **/
template <typename TaskSetT, typename Iter, typename F>
DISPENSO_REQUIRES(ForEachFunc<F, Iter>)
void for_each_n(TaskSetT& tasks, Iter start, size_t n, F&& f, ForEachOptions options = {}) {
  // TODO(bbudge): With options.maxThreads, we might want to allow a small fanout factor in
  // recursive case?
  if (!n || !options.maxThreads || detail::PerPoolPerThreadInfo::isParForRecursive(&tasks.pool())) {
    for (size_t i = 0; i < n; ++i) {
      f(*start);
      ++start;
    }
    if (options.wait) {
      tasks.wait();
    }
    return;
  }

  // 0 indicates serial execution per API spec
  int32_t maxThreads = std::max<int32_t>(options.maxThreads, 1);

  ssize_t numThreads = std::min<ssize_t>(tasks.numPoolThreads() + options.wait, maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<ssize_t>(numThreads, n);

  auto chunking = detail::staticChunkSize(n, numThreads);
  size_t chunkSize = chunking.ceilChunkSize;

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;
  ssize_t transitionIdx = chunking.transitionTaskIndex;
  size_t smallChunkSize = chunkSize - !perfectlyChunked;

  detail::for_each_n_schedule(
      tasks,
      start,
      std::forward<F>(f),
      numThreads,
      chunkSize,
      transitionIdx,
      smallChunkSize,
      options,
      typename std::iterator_traits<Iter>::iterator_category{});
}

/**
 * A function like std::for_each_n, but where the function is invoked in parallel across the passed
 * range.
 *
 * @param start The iterator for the start of the range.
 * @param n The length of the range.
 * @param f The function to execute in parallel.  This is a unary function that must be capable of
 * taking dereference of Iter.
 * @param options See ForEachOptions for details; however it should be noted that this function must
 * always wait, and therefore options.wait is ignored.
 **/
template <typename Iter, typename F>
DISPENSO_REQUIRES(ForEachFunc<F, Iter>)
void for_each_n(Iter start, size_t n, F&& f, ForEachOptions options = {}) {
  TaskSet taskSet(globalThreadPool());
  options.wait = true;
  for_each_n(taskSet, start, n, std::forward<F>(f), options);
}

/**
 * A function like std::for_each, but where the function is invoked in parallel across the passed
 * range.
 *
 * @param tasks The task set to schedule the for_each on.
 * @param start The iterator for the start of the range.
 * @param end The iterator for the end of the range.
 * @param f The function to execute in parallel.  This is a unary function that must be capable of
 * taking dereference of Iter.
 * @param options See ForEachOptions for details.
 **/
template <typename TaskSetT, typename Iter, typename F>
DISPENSO_REQUIRES(ForEachFunc<F, Iter>)
void for_each(TaskSetT& tasks, Iter start, Iter end, F&& f, ForEachOptions options = {}) {
  for_each_n(tasks, start, std::distance(start, end), std::forward<F>(f), options);
}

/**
 * A function like std::for_each, but where the function is invoked in parallel across the passed
 * range.
 *
 * @param start The iterator for the start of the range.
 * @param end The iterator for the end of the range.
 * @param f The function to execute in parallel.  This is a unary function that must be capable of
 * taking dereference of Iter.
 * @param options See ForEachOptions for details; however it should be noted that this function must
 * always wait, and therefore options.wait is ignored.
 **/
template <typename Iter, typename F>
DISPENSO_REQUIRES(ForEachFunc<F, Iter>)
void for_each(Iter start, Iter end, F&& f, ForEachOptions options = {}) {
  for_each_n(start, std::distance(start, end), std::forward<F>(f), options);
}

// TODO(bbudge): Implement ranges versions for these in C++20 (currently not in an env where this
// can be tested)

} // namespace dispenso
