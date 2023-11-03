/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph_executor.h>
namespace {

class RunningNode : public dispenso::Node {
 public:
  inline std::atomic<size_t>& numIncompletePredecessors() const {
    return numIncompletePredecessors_;
  }
  static constexpr size_t kCompleted = Node::kCompleted;
};

inline bool hasNoIncompletePredecessors(const dispenso::Node& node) {
  const RunningNode& rnode = static_cast<const RunningNode&>(node);
  return rnode.numIncompletePredecessors().load(std::memory_order_relaxed) == 0;
}

inline void resetNumIncompletePredecessors(const dispenso::Node& node) {
  const RunningNode& rnode = static_cast<const RunningNode&>(node);
  rnode.numIncompletePredecessors().store(rnode.numPredecessors(), std::memory_order_relaxed);
}

inline void addIncompletePredecessor(const dispenso::Node& node) {
  const RunningNode& rnode = static_cast<const RunningNode&>(node);

  if (rnode.isCompleted()) {
    rnode.numIncompletePredecessors().store(1, std::memory_order_relaxed);
  } else {
    rnode.numIncompletePredecessors().fetch_add(1, std::memory_order_relaxed);
  }
}

inline void ifIncompleteAddIncompletePredecessor(const dispenso::Node& node) {
  const RunningNode& rnode = static_cast<const RunningNode&>(node);

  if (!rnode.isCompleted()) {
    rnode.numIncompletePredecessors().fetch_add(1, std::memory_order_relaxed);
  }
}

inline bool decNumIncompletePredecessors(const dispenso::Node& node, std::memory_order order) {
  const RunningNode& rnode = static_cast<const RunningNode&>(node);

  return rnode.numIncompletePredecessors().fetch_sub(1, order) == 1;
}

inline bool decNumIncompletePredecessors(
    const dispenso::BiPropNode& node,
    std::memory_order order) {
  const RunningNode& rnode =
      static_cast<const RunningNode&>(static_cast<const dispenso::Node&>(node));

  const std::memory_order loadOrder =
      order == std::memory_order_relaxed ? std::memory_order_relaxed : std::memory_order_acquire;
  if (rnode.numIncompletePredecessors().load(loadOrder) == RunningNode::kCompleted) {
    return false;
  }

  return rnode.numIncompletePredecessors().fetch_sub(1, order) == 1;
}

template <class N>
inline void evaluateNodeConcurrently(dispenso::ConcurrentTaskSet& tasks, const N* node) {
  node->run();
  for (const dispenso::Node* const d : node->dependents()) {
    if (decNumIncompletePredecessors(static_cast<const N&>(*d), std::memory_order_acq_rel)) {
      tasks.schedule([&tasks, d]() { evaluateNodeConcurrently(tasks, static_cast<const N*>(d)); });
    }
  }
}

inline void propagateIncompleteStateBidirectionally(
    std::unordered_set<const dispenso::Node*>& /*visited*/,
    std::vector<const dispenso::Node*>& /*nodesToVisit*/
) {}

inline void propagateIncompleteStateBidirectionally(
    std::unordered_set<const dispenso::BiPropNode*>& visited,
    std::vector<const dispenso::BiPropNode*>& nodesToVisit) {
  nodesToVisit.clear();
  std::unordered_set<const std::unordered_set<const dispenso::BiPropNode*>*> groups;
  for (const dispenso::BiPropNode* node : visited) {
    const auto* group = node->bidirectionalPropagationSet();
    if (group == nullptr) {
      continue;
    }
    if (groups.insert(group).second) {
      for (const dispenso::BiPropNode* gnode : *group) {
        if (gnode->setIncomplete()) {
          nodesToVisit.emplace_back(gnode);
        }
      }
    }
  }

  for (const dispenso::BiPropNode* node : nodesToVisit) {
    for (const dispenso::Node* d : node->dependents()) {
      ifIncompleteAddIncompletePredecessor(*d);
    }
  }
}

} // anonymous namespace

namespace dispenso {

template <typename G>
void SingleThreadExecutor::operator()(const G& graph) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;
  nodesToExecute.clear();
  nodesToExecuteNext.clear();

  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      if (hasNoIncompletePredecessors(node)) {
        nodesToExecute.emplace_back(&node);
      }
    }
  }

  while (!nodesToExecute.empty()) {
    for (const Node* n : nodesToExecute) {
      const NodeType* node = static_cast<const NodeType*>(n);
      node->run();
      for (const Node* const d : node->dependents()) {
        if (decNumIncompletePredecessors(
                static_cast<const NodeType&>(*d), std::memory_order_relaxed)) {
          nodesToExecuteNext.emplace_back(static_cast<const NodeType*>(d));
        }
      }
    }
    nodesToExecute.swap(nodesToExecuteNext);
    nodesToExecuteNext.clear();
  }
}

template <typename TaskSetT, typename G>
void ParallelForExecutor::operator()(TaskSetT& taskSet, const G& graph) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;
  nodesToExecute.clear();
  nodesToExecuteNext.clear();

  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      if (hasNoIncompletePredecessors(node)) {
        nodesToExecute.emplace_back(&node);
      }
    }
  }
  while (!nodesToExecute.empty()) {
    dispenso::parallel_for(taskSet, size_t(0), nodesToExecute.size(), [this](size_t i) {
      const NodeType* node = static_cast<const NodeType*>(nodesToExecute[i]);
      node->run();
      for (const Node* const d : node->dependents()) {
        if (decNumIncompletePredecessors(
                static_cast<const NodeType&>(*d), std::memory_order_acq_rel)) {
          nodesToExecuteNext.emplace_back(static_cast<const NodeType*>(d));
        }
      }
    });

    nodesToExecute.swap(nodesToExecuteNext);
    nodesToExecuteNext.clear();
  }
}

template <typename G>
void ConcurrentTaskSetExecutor::operator()(
    dispenso::ConcurrentTaskSet& tasks,
    const G& graph,
    bool wait) {
  using NodeType = typename G::NodeType;
  using SubgraphType = typename G::SubgraphType;
  startNodes.clear();

  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      if (hasNoIncompletePredecessors(node)) {
        startNodes.emplace_back(&node);
      }
    }
  }

  for (const Node* n : startNodes) {
    const NodeType* node = static_cast<const NodeType*>(n);
    tasks.schedule([&tasks, node]() { evaluateNodeConcurrently(tasks, node); });
  }
  if (wait) {
    tasks.wait();
  }
}

template <typename G>
void setAllNodesIncomplete(const G& graph) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;

  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      resetNumIncompletePredecessors(node);
    }
  }
}

template <typename G>
void propagateIncompleteState(const G& graph) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;

  std::vector<const NodeType*> nodesToVisit, nodesToVisitNext;
  std::unordered_set<const NodeType*> visited;

  std::vector<const NodeType*> startNodes;

  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      if (!node.isCompleted()) {
        nodesToVisit.emplace_back(&node);
        visited.insert(&node);
      }
    }
  }

  while (!nodesToVisit.empty()) {
    for (const NodeType* node : nodesToVisit) {
      for (const Node* const d : node->dependents()) {
        addIncompletePredecessor(*d);
        if (visited.insert(static_cast<const NodeType*>(d)).second) {
          nodesToVisitNext.emplace_back(static_cast<const NodeType*>(d));
        }
      }
    }
    nodesToVisit.swap(nodesToVisitNext);
    nodesToVisitNext.clear();
  }

  propagateIncompleteStateBidirectionally(visited, nodesToVisit);
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
template DISPENSO_DLL_ACCESS void propagateIncompleteState<Graph>(const Graph&);
template DISPENSO_DLL_ACCESS void propagateIncompleteState<BiPropGraph>(const BiPropGraph&);

} // namespace dispenso
