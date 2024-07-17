/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <dispenso/platform.h>
#include <dispenso/pool_allocator.h>
#include <dispenso/small_buffer_allocator.h>
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

dispenso::Graph graph;

dispenso::Node& N0 = graph.addNode([&]() { r[0] += 1; });
dispenso::Node& N2 = graph.addNode([&]() { r[2] += 8; });
dispenso::Node& N1 = graph.addNode([&]() { r[1] += r[0] * 2; });
dispenso::Node& N3 = graph.addNode([&]() { r[3] += r[2] / 2; });
dispenso::Node& N4 = graph.addNode([&]() { r[4] += r[1] + r[3]; });

N4.dependsOn(N1, N3);
N1.dependsOn(N0);
N3.dependsOn(N2);

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
N1.setIncomplete();
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
dispenso::Graph graph;

dispenso::Subgraph& subgraph1 = graph.addSubgraph();
dispenso::Subgraph& subgraph2 = graph.addSubgraph();

dispenso::Node& N0 = subgraph1.addNode([&]() { r[0] += 1; });
dispenso::Node& N2 = subgraph1.addNode([&]() { r[2] += 8; });
dispenso::Node& N1 = subgraph2.addNode([&]() { r[1] += r[0] * 2; });
dispenso::Node& N3 = subgraph2.addNode([&]() { r[3] += r[2] / 2; });
dispenso::Node& N4 = graph.addNode([&]() { r[4] += r[1] + r[3]; });

N4.dependsOn(N1, N3);
N1.dependsOn(N0);
N3.dependsOn(N2);

// evaluate graph first time
r = {0, 0, 0, 0, 0};
dispenso::ConcurrentTaskSet concurrentTaskSet(dispenso::globalThreadPool());
dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
concurrentTaskSetExecutor(concurrentTaskSet, graph);

// disconnect and destroy nodes of subgraph2
// it invalidates node references/pointers of this subgraph
subgraph2.clear();

// create another nodes
dispenso::Node& newN1 = subgraph2.addNode([&]() { r[1] += r[0] * 20; });
dispenso::Node& newN3 = subgraph2.addNode([&]() { r[3] += r[2] / 20; });
newN1.dependsOn(N0);
newN3.dependsOn(N2);
N4.dependsOn(newN1, newN3);

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

dispenso::BiPropNode& N0 = g.addNode([&]() { b += 5; });
dispenso::BiPropNode& N1 = g.addNode([&]() { b *= 5; });
dispenso::BiPropNode& N2 = g.addNode([&]() { b /= m4; });
dispenso::BiPropNode& N3 = g.addNode([&]() { m3 += b*b; });
dispenso::BiPropNode& N4 = g.addNode([&]() { m4 += 2; });

N3.dependsOn(N1);
N2.dependsOn(N4);
N2.biPropDependsOn(N1);
N1.biPropDependsOn(N0);

// first execution
b = m3 = m4 = 0.f;
dispenso::ConcurrentTaskSet concurrentTaskSet(dispenso::globalThreadPool());
dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
concurrentTaskSetExecutor(concurrentTaskSet, g);

N[4].setIncomplete();
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

template <typename F>
void callFunctor(void* ptr) {
  (*static_cast<F*>(ptr))();
}

template <typename F>
void destroyFunctor(void* ptr) {
  static_cast<F*>(ptr)->~F();
  constexpr size_t kFuncSize = static_cast<size_t>(dispenso::detail::nextPow2(sizeof(F)));
  dispenso::deallocSmallBuffer<kFuncSize>(ptr);
}

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
        invoke_(other.invoke_),
        destroy_(other.destroy_),
        funcBuffer_(other.funcBuffer_),
        dependents_(std::move(other.dependents_)) {
    other.funcBuffer_ = nullptr;
  }
  ~Node() {
    if (funcBuffer_) {
      destroy_(funcBuffer_);
    }
  }
  /**
   * Make this node depends on nodes. This is not concurrency safe.
   *
   * @param nodes predecessors of the node
   **/
  template <typename... Ns>
  inline void dependsOn(Ns&... nodes) {
    ((void)std::initializer_list<int>{(dependsOnOneNode(nodes), 0)...});
  }
  /**
   * Invoke the type-erased functor. Change competed state of the node to "Incomplete".
   * Concurrency safe.
   **/
  inline void run() const {
    invoke_(funcBuffer_);
    numIncompletePredecessors_.store(kCompleted, std::memory_order_release);
  }
  /**
   * apply an func to each dependent of the node
   *
   * @param func a functor with signature <code>void(const Node&)</code>
   **/
  template <class F>
  inline void forEachDependent(F&& func) const {
    for (const Node* dependent : dependents_) {
      func(*dependent);
    }
  }
  /**
   * apply an func to each dependent of the node This is not concurrency safe.
   *
   * @param func a functor with signature <code>void(Node&)</code>
   **/
  template <class F>
  inline void forEachDependent(F&& func) {
    for (Node* dependent : dependents_) {
      func(*dependent);
    }
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
  template <class F, class X = std::enable_if_t<!std::is_base_of<Node, F>::value, void>>
  Node(F&& f) : numIncompletePredecessors_(0) {
    using FNoRef = typename std::remove_reference<F>::type;

    constexpr size_t kFuncSize = static_cast<size_t>(detail::nextPow2(sizeof(FNoRef)));
    funcBuffer_ = allocSmallBuffer<kFuncSize>();
    new (funcBuffer_) FNoRef(std::forward<F>(f));
    invoke_ = ::detail::callFunctor<FNoRef>;
    destroy_ = ::detail::destroyFunctor<FNoRef>;
  }

  void dependsOnOneNode(Node& node) {
    node.dependents_.emplace_back(this);
    numPredecessors_++;
  }

  static constexpr size_t kCompleted = std::numeric_limits<size_t>::max();
  mutable std::atomic<size_t> numIncompletePredecessors_;
  size_t numPredecessors_ = 0;

 private:
  using InvokerType = void (*)(void* ptr);

  InvokerType invoke_;
  InvokerType destroy_;
  char* funcBuffer_;

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
  inline void biPropDependsOn(Ns&... nodes) {
    ((void)std::initializer_list<int>{(biPropDependsOnOneNode(nodes), 0)...});
  }
  /**
   * Return true if node belongs to the same propogation set. (That means both nodes after
   * propogation become completed/incomplete together.)
   *
   * @param node to test
   **/
  inline bool isSameSet(const BiPropNode& node) const {
    return biPropSet_ && biPropSet_ == node.biPropSet_;
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

  DISPENSO_DLL_ACCESS void biPropDependsOnOneNode(BiPropNode& node);

  std::shared_ptr<std::vector<const BiPropNode*>> biPropSet_;

  template <class N>
  friend class SubgraphT;
  friend class ::detail::ExecutorBase;
};

template <class N>
class GraphT;

template <class N>
class DISPENSO_DLL_ACCESS SubgraphT {
 public:
  using NodeType = N;
  SubgraphT() = delete;
  SubgraphT(const SubgraphT<N>&) = delete;
  SubgraphT<N>& operator=(const SubgraphT<N>&) = delete;
  SubgraphT(SubgraphT<N>&& other) noexcept
      : graph_(other.graph_),
        nodes_(std::move(other.nodes_)),
        allocator_(std::move(other.allocator_)) {}
  ~SubgraphT();
  /**
   * Construct a <code>NodeType</code> with a valid functor. This is not concurrency safe.
   *
   * @param f A functor with signature void().
   * @return reference to the created node.
   **/
  template <class T>
  N& addNode(T&& f) {
    nodes_.push_back(new (allocator_->alloc()) NodeType(std::forward<T>(f)));
    return *nodes_.back();
  }
  /**
   * Return number of nodes in subgraph. Concurrency safe.
   **/
  size_t numNodes() const {
    return nodes_.size();
  }
  /**
   * Return const reference to node with index. Concurrency safe.
   *
   * @param index - index of the node
   **/
  const N& node(size_t index) const {
    return *nodes_[index];
  }
  /**
   * Return reference to node with index. Concurrency safe.
   *
   * @param index - index of the node
   **/
  N& node(size_t index) {
    return *nodes_[index];
  }
  /**
   * apply an func to each node of the subgraph. Concurrency safe.
   *
   * @param func a functor with signature <code>void(const Node&)</code>
   **/
  template <class F>
  inline void forEachNode(F&& func) const {
    for (const N* node : nodes_) {
      func(*node);
    }
  }
  /**
   * apply an func to each node of the subgraph. This is not concurrency safe.
   * This methods should never be called concurrent to when the graph execution is happening
   *
   * @param func a functor with signature <code>void(Node&)</code>
   **/
  template <class F>
  inline void forEachNode(F&& func) {
    for (N* node : nodes_) {
      func(*node);
    }
  }
  /**
   * Removes all dependency between nodes of this subgraph and other nodes, destroy this subgraph
   * nodes. This is not concurrency safe.
   **/
  void clear();

 private:
  using DeallocFunc = void (*)(NoLockPoolAllocator*);
  using PoolPtr = std::unique_ptr<NoLockPoolAllocator, DeallocFunc>;

  static constexpr size_t kNodeSizeP2 = detail::nextPow2(sizeof(NodeType));

  explicit SubgraphT<N>(GraphT<N>* graph) : graph_(graph), nodes_(), allocator_(getAllocator()) {}

  inline void removeNodeFromBiPropSet(Node* /* node */) {}
  void removeNodeFromBiPropSet(BiPropNode* node) {
    node->removeFromBiPropSet();
  }
  void decrementDependentCounters();
  size_t markNodesWithPredicessors();
  void removePredecessorDependencies(size_t numGraphPredecessors);

  void destroyNodes();

  static PoolPtr getAllocator();
  static void releaseAllocator(NoLockPoolAllocator* ptr);

  GraphT<N>* graph_;
#if defined(_WIN32) && !defined(__MINGW32__)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
  std::vector<N*> nodes_;

  PoolPtr allocator_;
#if defined(_WIN32) && !defined(__MINGW32__)
#pragma warning(pop)
#endif

  template <class T>
  friend class GraphT;
};

template <class N>
class DISPENSO_DLL_ACCESS GraphT {
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
  GraphT<N>& operator=(GraphT&& other) noexcept;
  /**
   * Construct a <code>NodeType</code> with a valid functor. This node is created into subgraph 0.
   * This is not concurrency safe.
   *
   * @param f A functor with signature void().
   **/
  template <class T>
  N& addNode(T&& f) {
    return subgraphs_[0].addNode(std::forward<T>(f));
  }
  /**
   * Return number of nodes in subgraph 0. Concurrency safe.
   **/
  size_t numNodes() const {
    return subgraphs_[0].numNodes();
  }
  /**
   * Return const reference to node with index in subgraph 0. Concurrency safe.
   *
   * @param index - index of the node
   **/
  const N& node(size_t index) const {
    return subgraphs_[0].node(index);
  }
  /**
   * Return reference to node with index in subgraph 0. Concurrency safe.
   *
   * @param index - index of the node
   **/
  N& node(size_t index) {
    return subgraphs_[0].node(index);
  }
  /**
   * Create an empty subgraph. This is not concurrency safe.
   **/
  SubgraphT<N>& addSubgraph();
  /**
   * Return number of subgraphs in the graph including subgraph 0. Concurrency safe.
   **/
  size_t numSubgraphs() const {
    return subgraphs_.size();
  }
  /**
   * Return const reference to subgraph with index. Concurrency safe.
   *
   * @param index - index of the subgraph.
   **/
  const SubgraphT<N>& subgraph(size_t index) const {
    return subgraphs_[index];
  }
  /**
   * Return reference to subgraph with index. Concurrency safe.
   *
   * @param index - index of the subgraph.
   **/
  SubgraphT<N>& subgraph(size_t index) {
    return subgraphs_[index];
  }
  /**
   * apply an func to each subgraph in the graph. Concurrency safe.
   *
   * @param func a functor with signature <code>void(const SubgraphT<N>&)</code>
   **/
  template <class F>
  inline void forEachSubgraph(F&& func) const {
    for (const SubgraphT<N>& subgraph : subgraphs_) {
      func(subgraph);
    }
  }
  /**
   * apply an func to each subgraph in the graph. Concurrency safe.
   *
   * @param func a functor with signature <code>void(SubgraphT<N>&)</code>
   **/
  template <class F>
  inline void forEachSubgraph(F&& func) {
    for (SubgraphT<N>& subgraph : subgraphs_) {
      func(subgraph);
    }
  }
  /**
   * apply an func to each node in the graph including all nodes from all subgraphs. Concurrency
   * safe.
   *
   * @param func a functor with signature <code>void(const Node&)</code>
   **/
  template <class F>
  inline void forEachNode(F&& func) const {
    for (const SubgraphT<N>& subgraph : subgraphs_) {
      for (const N* node : subgraph.nodes_) {
        func(*node);
      }
    }
  }
  /**
   * apply an func to each node in the graph. Concurrency safe.
   *
   * @param func a functor with signature <code>void(const Node&)</code>
   **/
  template <class F>
  inline void forEachNode(F&& func) {
    for (SubgraphT<N>& subgraph : subgraphs_) {
      for (N* node : subgraph.nodes_) {
        func(*node);
      }
    }
  }
  /**
   * Destroy all nodes and subgraphs. This is not concurrency safe.
   **/
  inline void clear() {
    subgraphs_.clear();
    subgraphs_.push_back(SubgraphType(this));
  }
  /**
   * Destroy all nodes. Keeps subgraphs. This is not concurrency safe.
   **/
  inline void clearSubgraphs() {
    for (SubgraphT<N>& subgraph : subgraphs_) {
      subgraph.destroyNodes();
    }
  }

 private:
  static constexpr size_t kSubgraphSizeP2 = detail::nextPow2(sizeof(SubgraphType));

#if defined(_WIN32) && !defined(__MINGW32__)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
  std::deque<SubgraphT<N>> subgraphs_;
#if defined(_WIN32) && !defined(__MINGW32__)
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
