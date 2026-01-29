/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file graph_executor.h
 * @ingroup group_graph
 * A file providing executors for running task graphs.
 **/

#pragma once
#include <dispenso/concurrent_vector.h>
#include <dispenso/detail/graph_executor_impl.h>
#include <dispenso/graph.h>
#include <dispenso/parallel_for.h>
#include <dispenso/platform.h>

namespace dispenso {
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> on current thread.
 **/
class SingleThreadExecutor : public ::detail::ExecutorBase {
 public:
  /**
   * Invoke the graph. This is not concurrency safe.
   *
   * @param graph graph to invoke
   **/
  template <typename G>
  void operator()(const G& graph);

 private:
  std::vector<const Node*> nodesToExecute_;
  std::vector<const Node*> nodesToExecuteNext_;
};
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> using
 * <code>dispenso::parallel_for</code> for every layer of the graph.
 **/
class ParallelForExecutor : public ::detail::ExecutorBase {
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
  dispenso::ConcurrentVector<const Node*> nodesToExecute_;
  dispenso::ConcurrentVector<const Node*> nodesToExecuteNext_;
};
/**
 * Class to invoke <code>Graph</code> or <code>BiPropGraph</code> using
 * <code>dispenso::ConcurrentTaskSet</code>
 **/
class ConcurrentTaskSetExecutor : public ::detail::ExecutorBase {
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
  std::vector<const Node*> startNodes_;
};

/**
 * Class to propagate incomplete state recursively from nodes to dependents
 **/
class ForwardPropagator : public ::detail::ExecutorBase {
 public:
  /**
   * Propagate incomplete state recursively from nodes to dependents
   * This is not concurrency safe.
   **/
  template <class G>
  void operator()(const G& graph);

 private:
  template <class N>
  void propagateIncompleteStateBidirectionally();

  std::vector<const Node*> nodesToVisit_;
  std::vector<const Node*> nodesToVisitNext_;
  std::unordered_set<const Node*> visited_;
  std::unordered_set<const std::vector<const BiPropNode*>*> groups_;
};
} // namespace dispenso
