/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A user's own global `detail` namespace must not collide with anything graph.h declares. graph.h
// used to define callFunctor/destroyFunctor (and forward-declare ExecutorBase) in ::detail, so
// these definitions failed to compile with "redefinition" errors. They are deliberately included
// before graph.h.

#include <atomic>

namespace detail {
template <typename F>
void callFunctor(void*) {}
template <typename F>
void destroyFunctor(void*) {}
using ExecutorBase = int;
} // namespace detail

#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>

#include <gtest/gtest.h>

TEST(GraphNamespace, UserGlobalDetailNamespaceDoesNotCollide) {
  std::atomic<int> runs(0);
  dispenso::Graph graph;
  graph.addNode([&runs]() { runs.fetch_add(1, std::memory_order_relaxed); });
  dispenso::SingleThreadExecutor executor;
  executor(graph);
  EXPECT_EQ(runs.load(), 1);
}
