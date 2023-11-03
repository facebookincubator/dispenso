/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/platform.h>
#include <atomic>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

/*
Terminology
--------------------------------------------------------------------------------
The Node depends on the predecessor. The dependent depends on the node.
~~~
┌─────────────┐     ┌──────┐     ┌───────────┐
│ predecessor │ ──▶ │ node │ ──▶ │ dependent │
└─────────────┘     └──────┘     └───────────┘
~~~

Graph construction
--------------------------------------------------------------------------------
The Graph class can be used to created tasks with dependencies and execute it once.
Graphs must not contain cycles.
Example:
~~~
//
//   ┌────────────┐     ┌───────────────┐     ┌───────────────────┐
//   │ 0: r[0]+=1 │ ──▶ │ 1: r[1]=r[0]*2│ ──▶ │ 4: r[4]=r[1]+r[3] │
//   └────────────┘     └───────────────┘     └───────────────────┘
//                                              ▲
//   ┌────────────┐     ┌───────────────┐       │
//   │ 2: r[2]+=8 │ ──▶ │ 3: r[3]=r[2]/2│ ──────┘
//   └────────────┘     └───────────────┘

std::array<float, 5> r;
std::array<dispenso::Node*, 5> N;

dispenso::Graph graph;

N[0] = graph.addNode([&]() { r[0] += 1; });
N[2] = graph.addNode([&]() { r[2] += 8; });
N[1] = graph.addNode([&]() { r[1] += r[0] * 2; });
N[3] = graph.addNode([&]() { r[3] += r[2] / 2; });
N[4] = graph.addNode([&]() { r[4] += r[1] + r[3]; });

N[4]->dependsOn(N[1], N[3]);
N[1]->dependsOn(N[0]);
N[3]->dependsOn(N[2]);

dispenso::TaskSet taskSet(dispenso::globalThreadPool());
dispenso::ParallelForExecutor parallelForExecutor;
parallelForExecutor(taskSet, graph);
~~~

Partial revaluation
--------------------------------------------------------------------------------
If graph is big or we need to recompute graph partially we can execute it again.
After execution of the graph all nodes change their state from "incomplete" to
"completed". If order to evaluate whole graph again we can use function `setAllNodesIncomplete`
Example:
~~~
r = {0, 0, 0, 0, 0};
setAllNodesIncomplete(graph);
parallelForExecutor(taskSet, graph);
~~~

The graph can be recomputed partially if we have new input data for one or several nodes in the
graph. It order to do it we need to call `setIncomplete()` method for every node which we need to
recompute and after use functor `ForwardPropagator` to mark as "incomplete" all dependents.

Example:
~~~
N[1]->setIncomplete();
r[1] = r[4] = 0;
ForwardPropagator forwardPropagator;
forwardPropagator(graph);
evaluateGraph(graph);
~~~
In this exaple only node 1 and 4 will be invoked.

 Subgraphs
--------------------------------------------------------------------------------
It is possible to organize nodes into subgraphs and destroy and recreate if we have static and
dynamic parts of the computation graph
Example:
~~~
//
// ∙----subgraph1---∙ ∙---subgraph2-------∙
// ¦ ┌────────────┐ ¦ ¦ ┌───────────────┐ ¦   ┌───────────────────┐
// ¦ │ 0: r[0]+=1 │ ──▶ │ 1: r[1]=r[0]*2│ ──▶ │ 4: r[4]=r[1]+r[3] │
// ¦ └────────────┘ ¦ ¦ └───────────────┘ ¦   └───────────────────┘
// ¦                ¦ ¦                   ¦     ▲
// ¦ ┌────────────┐ ¦ ¦ ┌───────────────┐ ¦     │
// ¦ │ 2: r[2]+=8 │ ──▶ │ 3: r[3]=r[2]/2│ ──────┘
// ¦ └────────────┘ ¦ ¦ └───────────────┘ ¦
// ∙----------------∙ ∙-------------------∙
std::array<float, 5> r;
std::array<dispenso::Node*, 5> N;
dispenso::Graph graph;
dispenso::Subgraph* subgraph1;
dispenso::Subgraph* subgraph2;

subgraph1 = graph.addSubgraph();
subgraph2 = graph.addSubgraph();

N[0] = subgraph1->addNode([&]() { r[0] += 1; });
N[2] = subgraph1->addNode([&]() { r[2] += 8; });
N[1] = subgraph2->addNode([&]() { r[1] += r[0] * 2; });
N[3] = subgraph2->addNode([&]() { r[3] += r[2] / 2; });
N[4] = graph.addNode([&]() { r[4] += r[1] + r[3]; });

N[4]->dependsOn(N[1], N[3]);
N[1]->dependsOn(N[0]);
N[3]->dependsOn(N[2]);

// evaluate graph first time
r = {0, 0, 0, 0, 0};
dispenso::ConcurrentTaskSet concurrentTaskSet(dispenso::globalThreadPool());
dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
concurrentTaskSetExecutor(concurrentTaskSet, graph);

// disconnect and destroy nodes of subgraph2
subgraph2->clear();

// create another nodes
N[1] = subgraph2->addNode([&]() { r[1] += r[0] * 20; });
N[3] = subgraph2->addNode([&]() { r[3] += r[2] / 20; });
N[1]->dependsOn(N[0]);
N[3]->dependsOn(N[2]);
N[4]->dependsOn(N[1], N[3]);

// and revaluae the graph
setAllNodesIncomplete(movedGraph);
concurrentTaskSetExecutor(concurrentTaskSet, graph);
~~~

Bidirectional propagation dependency
--------------------------------------------------------------------------------
In certain scenarios, nodes may alter the same memory. In such instances, it becomes necessary to
compute the predecessors of the node, even if they possess a "completed" state following state
propagation. To facilitate this process automatically, we introduce the notion of a bidirectional
propagation dependency (`BiProp`).

Example:
~~~
//                     ┌─────────────────┐
//                     │ 3: m3+=b*b      │
//                     └─────────────────┘
//                       ▲
// ┌−-----------−−−−−−−−−│−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
// ╎ ┌───────────┐     ┌─────────────────┐     ┌────────────┐ ╎
// ╎ │  0: b+=5  │ ──▷ │     1: b*=5     │ ──▷ │ 2: b/=m4   │ ╎
// ╎ └───────────┘     └─────────────────┘     └────────────┘ ╎
// └−−−-−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−▲−−−−−−−−−−−−┘
//  Legend:                                      │
//  ──▶ Normal dependency                      ┌────────────┐
//  ──▷ Bidirectional propagation dependency   │ 4: m4+=2   │
//  m4  variable modified only in node 4       └────────────┘

float  b,  m3, m4
dispenso::BiPropGraph g;
std::array<dispenso::BiPropNode*, 8> N;

N[0] = g.addNode([&]() { b += 5; });
N[1] = g.addNode([&]() { b *= 5; });
N[2] = g.addNode([&]() { b /= m4; });
N[3] = g.addNode([&]() { m3 += b*b; });
N[4] = g.addNode([&]() { m4 += 2; });

N[3]->dependsOn(N[1]);
N[2]->dependsOn(N[4]);
N[2]->biPropDependsOn(N[1]);
N[1]->biPropDependsOn(N[0]);

// first execution
b = m3 = m4 = 0.f;
dispenso::ConcurrentTaskSet concurrentTaskSet(dispenso::globalThreadPool());
dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
concurrentTaskSetExecutor(concurrentTaskSet, g);

N[4]->setIncomplete();
// if node 4 is incomplete after propagation node 2 become incomplete. Taking in account that node 2
// bidirectionally depends on nodes 0 and 1 they will be marked as incomplete as well
b =  m4 = 0.f;
ForwardPropagator forwardPropagator;
forwardPropagator(g);
concurrentTaskSetExecutor(concurrentTaskSet, g);
~~~

Please read tests from `graph_test.cpp` for more examples.
*/

namespace detail {
class ExecutorBase;
} // namespace detail

namespace dispenso {
/**
 * Class to store task with dependencies
 **/
class Node {
 public:
  Node() = delete;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&& other) noexcept
      : numIncompletePredecessors_(other.numIncompletePredecessors_.load()),
        numPredecessors_(other.numPredecessors_),
        f_(std::move(other.f_)),
        dependents_(std::move(other.dependents_)) {}
  /**
   * Make this node depends on nodes. This is not concurrency safe.
   *
   * @param nodes predecessors of the node
   **/
  template <typename... Ns>
  inline void dependsOn(Ns... nodes) {
    ((void)std::initializer_list<int>{(dependsOnOneNode(std::forward<Ns>(nodes)), 0)...});
  }
  /**
   * Invoke the type-erased functor. Change competed state of the node to "Incomplete".
   * Concurrency safe.
   **/
  inline void run() const {
    f_();
    numIncompletePredecessors_.store(kCompleted, std::memory_order_release);
  }
  /**
   * Return vectoor of nodes depends on this node. Concurrency safe.
   **/
  inline const std::vector<Node*>& dependents() const {
    return dependents_;
  }
  /**
   * Return the number of nodes this node depends on. Concurrency safe.
   **/
  inline size_t numPredecessors() const {
    return numPredecessors_;
  }
  /**
   * Return true if node is completed.
   * New node always incomplete. If node was invoked it become completed. this
   * state can be changed by calling <code>setIncomplete()</code>
   * Concurrency safe.
   **/
  inline bool isCompleted() const {
    return numIncompletePredecessors_.load(std::memory_order_relaxed) == kCompleted;
  }
  /**
   * Mark node incomplete. (that allows reevaluate this node again).
   * Concurrency safe. This methods should never be called concurrent to when
   * the graph execution is happening
   *
   * @return true if state was changed.
   **/
  inline bool setIncomplete() const {
    if (numIncompletePredecessors_.load(std::memory_order_relaxed) == kCompleted) {
      numIncompletePredecessors_.store(0, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

  /**
   * Mark node completed.
   * Concurrency safe. This methods should never be called concurrent to when
   * the graph execution is happening
   **/
  inline void setCompleted() const {
    numIncompletePredecessors_.store(kCompleted, std::memory_order_relaxed);
  }

 protected:
  template <class T, class X = std::enable_if_t<!std::is_base_of<Node, T>::value, void>>
  Node(T&& f) : numIncompletePredecessors_(0), f_(std::forward<T>(f)) {}

  void dependsOnOneNode(Node* node) {
    node->dependents_.emplace_back(this);
    numPredecessors_++;
  }

  static constexpr size_t kCompleted = std::numeric_limits<size_t>::max();
  mutable std::atomic<size_t> numIncompletePredecessors_;
  size_t numPredecessors_ = 0;

 private:
  // TODO(roman fedotov):create more efficient implementation than std::function
  // (like  dispenso::OnceFunction)
  std::function<void()> f_;
  std::vector<Node*> dependents_; // nodes depend on this

  template <class N>
  friend class SubgraphT;
  friend class ::detail::ExecutorBase;
  template <typename G>
  friend void setAllNodesIncomplete(const G& graph);
};
/**
 * Class to store task with dependencies. Support bidirectional propagation dependency between
 *nodes.
 **/
class BiPropNode : public Node {
 public:
  BiPropNode() = delete;
  BiPropNode(const BiPropNode&) = delete;
  BiPropNode& operator=(const BiPropNode&) = delete;
  BiPropNode(BiPropNode&& other) noexcept
      : Node(std::move(other)), biPropSet_(std::move(other.biPropSet_)) {}
  /**
   * Make this node depends on nodes. Create bidirectional propagation dependency. This is not
   *concurrency safe.
   *
   * @param nodes predecessors of the node
   **/
  template <class... Ns>
  inline void biPropDependsOn(Ns... nodes) {
    ((void)std::initializer_list<int>{(biPropDependsOnOneNode(std::forward<Ns>(nodes)), 0)...});
  }
  /**
   * Return Set of nodes connected by bidirectional propagation dependencies with this node.
   **/
  const std::vector<const BiPropNode*>* bidirectionalPropagationSet() const {
    return biPropSet_.get();
  }

 private:
  template <class T, class X = std::enable_if_t<!std::is_base_of<BiPropNode, T>::value, void>>
  BiPropNode(T&& f) : Node(std::forward<T>(f)) {}
  inline void removeFromBiPropSet() {
    if (biPropSet_ != nullptr) {
      auto it = std::find(biPropSet_->begin(), biPropSet_->end(), this);
      if (it != biPropSet_->end()) {
        biPropSet_->erase(it);
      }
    }
  }

  DISPENSO_DLL_ACCESS void biPropDependsOnOneNode(BiPropNode* node);

  std::shared_ptr<std::vector<const BiPropNode*>> biPropSet_;

  template <class N>
  friend class SubgraphT;
};

template <class N>
class GraphT;

template <class N>
class SubgraphT {
 public:
  using NodeType = N;
  SubgraphT() = delete;
  SubgraphT(const SubgraphT<N>&) = delete;
  SubgraphT<N>& operator=(const SubgraphT<N>&) = delete;
  SubgraphT(SubgraphT<N>&& other) : graph_(other.graph_), nodes_(std::move(other.nodes_)) {}
  /**
   * Construct a <code>NodeType</code> with a valid functor. This is not concurrency safe.
   *
   * @param f A functor with signature void().
   **/
  template <class T>
  N* addNode(T&& f) {
    nodes_.push_back(NodeType(std::forward<T>(f)));
    return &nodes_.back();
  }
  /**
   * Return nodes of the subgraph. Concurrency safe.
   **/
  const std::deque<N>& nodes() const {
    return nodes_;
  }
  /**
   * Return nodes of the subgraph. Concurrency safe.
   **/
  std::deque<N>& nodes() {
    return nodes_;
  }
  /**
   * Removes all dependency between nodes of this subgraph and other nodes, destroy this subgraph
   * nodes. This is not concurrency safe.
   **/
  void clear();

 private:
  explicit SubgraphT<N>(GraphT<N>* graph) : graph_(graph), nodes_() {}
  inline void removeNodeFromBiPropSet(Node& /* node */) {}
  void removeNodeFromBiPropSet(BiPropNode& node) {
    node.removeFromBiPropSet();
  }
  void decrementDependentCounters();
  size_t markNodesWithPredicessors();
  void removePredecessorDependencies(size_t numGraphPredecessors);

  GraphT<N>* graph_;
#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
  std::deque<N> nodes_;
#if defined(_WIN32)
#pragma warning(pop)
#endif

  template <class T>
  friend class GraphT;
};

template <class N>
class GraphT {
 public:
  using NodeType = N;
  using SubgraphType = SubgraphT<N>;
  GraphT(const GraphT<N>&) = delete;
  GraphT& operator=(const GraphT<N>&) = delete;
  /**
   * Create empty graph.
   **/
  GraphT<N>() {
    subgraphs_.push_back(SubgraphType(this));
  }
  /**
   * Move constructor
   **/
  GraphT(GraphT<N>&& other);
  /**
   * Move assignment operator
   **/
  GraphT<N>& operator=(GraphT&& other);
  /**
   * Construct a <code>NodeType</code> with a valid functor. This is not concurrency safe.
   *
   * @param f A functor with signature void().
   **/
  template <class T>
  N* addNode(T&& f) {
    return subgraphs_[0].addNode(std::forward<T>(f));
  }
  /**
   * Return nodes that do not belong to subgraphs. Concurrency safe.
   **/
  const std::deque<N>& nodes() const {
    return subgraphs_[0].nodes_;
  }
  /**
   * Return nodes that do not belong to subgraphs. Concurrency safe.
   **/
  std::deque<N>& nodes() {
    return subgraphs_[0].nodes_;
  }
  /**
   * Create an empty subgraph. This is not concurrency safe.
   **/
  SubgraphT<N>* addSubgraph();
  /**
   * Return subgraphs of this graph. Concurrency safe.
   **/
  const std::deque<SubgraphT<N>>& subgraphs() const {
    return subgraphs_;
  }
  /**
   * Destroy all nodes and subgraphs. This is not concurrency safe.
   **/
  void clear() {
    subgraphs_.clear();
    subgraphs_.push_back(SubgraphType(this));
  }

 private:
#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
  std::deque<SubgraphT<N>> subgraphs_;
#if defined(_WIN32)
#pragma warning(pop)
#endif

  template <class T>
  friend class SubgraphT;
};

using Graph = GraphT<Node>;
using BiPropGraph = GraphT<BiPropNode>;

using Subgraph = SubgraphT<Node>;
using BiPropSubgraph = SubgraphT<BiPropNode>;
} // namespace dispenso
