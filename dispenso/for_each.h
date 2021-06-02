// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

/**
 * @file for_each.h
 * Functions for performing parallel for_each over iterables.  This intends to more-or-less
 * mimic std::for_each, with a possible (Concurrent)TaskSet passed in for external wait capability,
 * and ForEachOptions for controlling the wait behavior and limiting of parallelism.
 **/

#pragma once

#include <algorithm>

#include <dispenso/detail/per_thread_info.h>
#include <dispenso/task_set.h>

namespace dispenso {

/**
 * A set of options to control for_each
 **/
struct ForEachOptions {
  /**
   * The maximum number of threads to use.  This can be used to limit the number of threads below
   * the number associated with the TaskSet's thread pool.  Setting maxThreads to zero will result
   * in serial operation.
   **/
  uint32_t maxThreads = std::numeric_limits<uint32_t>::max();
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
void for_each_n(TaskSetT& tasks, Iter start, size_t n, F&& f, ForEachOptions options = {}) {
  // TODO(bbudge): With options.maxThreads, we might want to allow a small fanout factor in
  // recursive case?
  if (!options.maxThreads || detail::PerPoolPerThreadInfo::isParForRecursive(&tasks.pool())) {
    for (size_t i = 0; i < n; ++i) {
      f(*start);
      ++start;
    }
    return;
  }

  ssize_t numThreads = std::min<ssize_t>(tasks.numPoolThreads(), options.maxThreads);
  // Reduce threads used if they exceed work to be done.
  numThreads = std::min<ssize_t>(numThreads, n);

  auto chunking = detail::staticChunkSize(n, numThreads);
  size_t chunkSize = chunking.ceilChunkSize;

  bool perfectlyChunked = chunking.transitionTaskIndex == numThreads;

  // (!perfectlyChunked) ? chunking.transitionTaskIndex : numThreads - 1;
  ssize_t firstLoopLen = chunking.transitionTaskIndex - perfectlyChunked;

  ssize_t t;
  for (t = 0; t < firstLoopLen; ++t) {
    Iter next = start;
    std::advance(next, chunkSize);
    tasks.schedule([start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      for (Iter it = start; it != next; ++it) {
        f(*it);
      }
    });
    start = next;
  }

  // Reduce the remaining chunk sizes by 1.
  chunkSize -= !perfectlyChunked;
  // Finish submitting all but the last item.
  for (; t < numThreads - 1; ++t) {
    Iter next = start;
    std::advance(next, chunkSize);
    tasks.schedule([start, next, f]() {
      auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
      for (Iter it = start; it != next; ++it) {
        f(*it);
      }
    });
    start = next;
  }

  Iter end = start;
  std::advance(end, chunkSize);

  if (options.wait) {
    for (Iter it = start; it != end; ++it) {
      f(*it);
    }
    tasks.wait();
  } else {
    tasks.schedule(
        [start, end, f]() {
          auto recurseInfo = detail::PerPoolPerThreadInfo::parForRecurse();
          for (Iter it = start; it != end; ++it) {
            f(*it);
          }
        },
        ForceQueuingTag());
  }
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
void for_each(Iter start, Iter end, F&& f, ForEachOptions options = {}) {
  for_each_n(start, std::distance(start, end), std::forward<F>(f), options);
}

// TODO(bbudge): Implement ranges versions for these in C++20 (currently not in an env where this
// can be tested)

} // namespace dispenso
