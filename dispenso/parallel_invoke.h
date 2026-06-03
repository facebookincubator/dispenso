/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>

#include <dispenso/platform.h>
#include <dispenso/task_set.h>

namespace dispenso {

/**
 * Fork-join invocation of two or more functors using a ConcurrentTaskSet for
 * scheduling.  The last functor runs inline on the calling thread while the
 * preceding ones are submitted to the pool for parallel execution.
 *
 * This is the canonical fork-join idiom: submit siblings, do useful work on
 * the calling thread instead of just waiting.  Composes naturally with
 * recursive divide-and-conquer code (parallel tree builds, kd-tree
 * construction, parallel quicksort, etc.).
 *
 * Implementation: recursively peels the first functor as a schedule() and
 * tail-recurses on the rest.  The compiler unrolls this at the call site
 * for a fixed arity, producing N-1 schedule() calls followed by an inline
 * invocation of the last functor.  No type erasure or staging buffer.
 *
 * @param tasks A ConcurrentTaskSet whose pool will execute the queued tasks.
 *              The caller is responsible for calling tasks.wait() (typically
 *              once at the top of an algorithm) to synchronize completion.
 * @param fs    Two or more nullary functors to execute concurrently.
 *
 * @note parallel_invoke does not call wait() internally.  This keeps it
 * composable with recursive divide-and-conquer where the caller drives a
 * single wait at the top of the algorithm.
 **/
template <typename F>
DISPENSO_INLINE void parallel_invoke(ConcurrentTaskSet& /*tasks*/, F&& f) {
  // Single-functor base case: run inline.
  std::forward<F>(f)();
}

template <typename F1, typename F2, typename... Fs>
DISPENSO_INLINE void parallel_invoke(ConcurrentTaskSet& tasks, F1&& f1, F2&& f2, Fs&&... fs) {
  // skipRecheck=true: the per-call inline gate's TaskSet check is enough; the
  // pool-level recheck is redundant when the caller is in a fork-join
  // pattern.
  tasks.schedule(std::forward<F1>(f1), /*skipRecheck=*/true);
  parallel_invoke(tasks, std::forward<F2>(f2), std::forward<Fs>(fs)...);
}

} // namespace dispenso
