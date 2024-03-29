/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph_executor.h>

namespace dispenso {

template <typename G>
void SingleThreadExecutor::operator()(const G& graph) {
  using NodeType = typename G::NodeType;
  nodesToExecute_.clear();
  nodesToExecuteNext_.clear();

  graph.forEachNode([&](const NodeType& node) {
    if (hasNoIncompletePredecessors(node)) {
      nodesToExecute_.emplace_back(&node);
    }
  });

  while (!nodesToExecute_.empty()) {
    for (const Node* n : nodesToExecute_) {
      const NodeType* node = static_cast<const NodeType*>(n);
      node->run();
      node->forEachDependent([&](const Node& d) {
        if (decNumIncompletePredecessors(
                static_cast<const NodeType&>(d), std::memory_order_relaxed)) {
          nodesToExecuteNext_.emplace_back(static_cast<const NodeType*>(&d));
        }
      });
    }
    nodesToExecute_.swap(nodesToExecuteNext_);
    nodesToExecuteNext_.clear();
  }
}

template <typename TaskSetT, typename G>
void ParallelForExecutor::operator()(TaskSetT& taskSet, const G& graph) {
  using NodeType = typename G::NodeType;
  nodesToExecute_.clear();
  nodesToExecuteNext_.clear();

  graph.forEachNode([&](const NodeType& node) {
    if (hasNoIncompletePredecessors(node)) {
      nodesToExecute_.emplace_back(&node);
    }
  });
  while (!nodesToExecute_.empty()) {
    dispenso::parallel_for(taskSet, size_t(0), nodesToExecute_.size(), [this](size_t i) {
      const NodeType* node = static_cast<const NodeType*>(nodesToExecute_[i]);
      node->run();

      node->forEachDependent([&](const Node& d) {
        if (decNumIncompletePredecessors(
                static_cast<const NodeType&>(d), std::memory_order_acq_rel)) {
          nodesToExecuteNext_.emplace_back(static_cast<const NodeType*>(&d));
        }
      });
    });

    nodesToExecute_.swap(nodesToExecuteNext_);
    nodesToExecuteNext_.clear();
  }
}

template <typename G>
void ConcurrentTaskSetExecutor::operator()(
    dispenso::ConcurrentTaskSet& tasks,
    const G& graph,
    bool wait) {
  using NodeType = typename G::NodeType;
  startNodes_.clear();

  graph.forEachNode([&](const NodeType& node) {
    if (hasNoIncompletePredecessors(node)) {
      startNodes_.emplace_back(&node);
    }
  });

  for (const Node* n : startNodes_) {
    const NodeType* node = static_cast<const NodeType*>(n);
    tasks.schedule([&tasks, node]() { evaluateNodeConcurrently(tasks, node); });
  }
  if (wait) {
    tasks.wait();
  }
}

template <typename G>
void setAllNodesIncomplete(const G& graph) {
  using NodeType = typename G::NodeType;

  graph.forEachNode([&](const NodeType& node) {
    node.numIncompletePredecessors_.store(node.numPredecessors(), std::memory_order_relaxed);
  });
}

template <typename G>
void ForwardPropagator::operator()(const G& graph) {
  using NodeType = typename G::NodeType;

  nodesToVisit_.clear();
  nodesToVisitNext_.clear();
  visited_.clear();
  groups_.clear();

  graph.forEachNode([&](const NodeType& node) {
    if (!node.isCompleted()) {
      nodesToVisit_.emplace_back(&node);
      visited_.insert(&node);
      appendGroup(static_cast<const NodeType*>(&node), groups_);
    }
  });

  while (!nodesToVisit_.empty()) {
    for (const Node* node : nodesToVisit_) {
      node->forEachDependent([&](const Node& d) {
        addIncompletePredecessor(d);
        if (visited_.insert(static_cast<const NodeType*>(&d)).second) {
          nodesToVisitNext_.emplace_back(static_cast<const NodeType*>(&d));
          appendGroup(static_cast<const NodeType*>(&d), groups_);
        }
      });
    }
    nodesToVisit_.swap(nodesToVisitNext_);
    nodesToVisitNext_.clear();
  }

  propagateIncompleteStateBidirectionally<NodeType>();
}

template <>
void ForwardPropagator::propagateIncompleteStateBidirectionally<Node>() {}
template <>
void ForwardPropagator::propagateIncompleteStateBidirectionally<BiPropNode>() {
  nodesToVisit_.clear();

  for (const std::vector<const BiPropNode*>* group : groups_) {
    for (const dispenso::BiPropNode* gnode : *group) {
      if (gnode->setIncomplete()) {
        nodesToVisit_.emplace_back(gnode);
      }
    }
  }

  for (const Node* node : nodesToVisit_) {
    const BiPropNode* biPropNode = static_cast<const BiPropNode*>(node);
    biPropNode->forEachDependent([](const Node& d) { ifIncompleteAddIncompletePredecessor(d); });
  }
}

template DISPENSO_DLL_ACCESS void SingleThreadExecutor::operator()<Graph>(const Graph&);
template DISPENSO_DLL_ACCESS void SingleThreadExecutor::operator()<BiPropGraph>(const BiPropGraph&);

template DISPENSO_DLL_ACCESS void ParallelForExecutor::operator()<TaskSet, Graph>(
    TaskSet&,
    const Graph&);
template DISPENSO_DLL_ACCESS void ParallelForExecutor::operator()<TaskSet, BiPropGraph>(
    TaskSet&,
    const BiPropGraph&);
template DISPENSO_DLL_ACCESS void ParallelForExecutor::operator()<ConcurrentTaskSet, Graph>(
    ConcurrentTaskSet& tasks,
    const Graph& graph);
template DISPENSO_DLL_ACCESS void ParallelForExecutor::operator()<ConcurrentTaskSet, BiPropGraph>(
    ConcurrentTaskSet& tasks,
    const BiPropGraph& graph);

template DISPENSO_DLL_ACCESS void ConcurrentTaskSetExecutor::operator()<Graph>(
    dispenso::ConcurrentTaskSet& tasks,
    const Graph& graph,
    bool wait);
template DISPENSO_DLL_ACCESS void ConcurrentTaskSetExecutor::operator()<BiPropGraph>(
    dispenso::ConcurrentTaskSet& tasks,
    const BiPropGraph& graph,
    bool wait);

template DISPENSO_DLL_ACCESS void setAllNodesIncomplete<Graph>(const Graph&);
template DISPENSO_DLL_ACCESS void setAllNodesIncomplete<BiPropGraph>(const BiPropGraph&);
template DISPENSO_DLL_ACCESS void ForwardPropagator::operator()<Graph>(const Graph&);
template DISPENSO_DLL_ACCESS void ForwardPropagator::operator()<BiPropGraph>(const BiPropGraph&);

} // namespace dispenso
