/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <dispenso/concurrent_vector.h>
#include <dispenso/graph.h>
#include <dispenso/parallel_for.h>
#include <dispenso/platform.h>
#include <unordered_set>

namespace detail {} // namespace detail

namespace dispenso {
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> on current thread.
 **/
class SingleThreadExecutor {
 public:
  /**
   * Invoke the graph. This is not concurrency safe.
   *
   * @param graph graph to invoke
   **/
  template <typename G>
  void operator()(const G& graph);

 private:
  std::vector<const Node*> nodesToExecute;
  std::vector<const Node*> nodesToExecuteNext;
};
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> using
 * <code>dispenso::parallel_for</code> for every layer of the graph.
 **/
class ParallelForExecutor {
 public:
  /**
   * Invoke the graph. This is not concurrency safe.
   *
   * @param taskSet taksSet to use with <code>parallel_for</code>.
   * @param graph graph to invoke
   **/
  template <typename TaskSetT, typename G>
  void operator()(TaskSetT& taskSet, const G& graph);

 private:
  dispenso::ConcurrentVector<const Node*> nodesToExecute;
  dispenso::ConcurrentVector<const Node*> nodesToExecuteNext;
};
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> using
 * <code>dispenso::ConcurrentTaskSet</code>
 **/
class ConcurrentTaskSetExecutor {
 public:
  /**
   * Invoke the graph. This is not concurrency safe.
   *
   * @param tasks <code>ConcurrentTaskSet</code> to schedule tasks.
   * @param graph graph to invoke
   * @param wait if true run <code>tasks.wait()</code> at the end of the function
   **/
  template <typename G>
  void operator()(dispenso::ConcurrentTaskSet& tasks, const G& graph, bool wait = true);

 private:
  std::vector<const Node*> startNodes;
};

template <typename G>
void setAllNodesIncomplete(const G& graph);

template <typename G>
void propagateIncompleteState(const G& graph);
} // namespace dispenso
