/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>
#include <array>
#include <numeric>
#include <random>

template <class G>
class BigTree {
 public:
  static size_t sizeOfLevel(size_t level) {
    return 1ul << (numBits_ - shiftStep_ * level);
  }
  void buildTree() {
    for (size_t level = 1; level < numLevels_; ++level) {
      SubGraphType* subgraph = &g_.addSubgraph();
      subgraphs_[level] = subgraph;
      buildLevel(level);
    }
  }

  void allocateMemory() {
    // the goal of this task is to calculate the sum of the numbers in this array
    for (size_t level = 1; level < numLevels_; ++level) {
      const size_t s = sizeOfLevel(level);
      data_[level].resize(s, 0lu);
    }

    // zero level data is input data for calculation
    data_[0].resize(1ul << numBits_);
    std::iota(data_[0].begin(), data_[0].end(), 0);
  }

  void buildLevel(size_t level) {
    SubGraphType* subgraph = subgraphs_[level];

    const size_t numNodes = 1ul << (numBits_ - shiftStep_ * level);

    for (size_t n = 0; n < numNodes; ++n) {
      subgraph->addNode([this, level, n]() {
        for (size_t j = 0; j < numPredecessors_; ++j) {
          const size_t index = (n << shiftStep_) | j;
          data_[level][n] += data_[level - 1][index];
        }
      });
      if (level > 1) {
        for (size_t j = 0; j < numPredecessors_; ++j) {
          const size_t index = (n << shiftStep_) | j;
          subgraphs_[level]->node(n).dependsOn(subgraphs_[level - 1]->node(index));
        }
      }
    }
  }
  static constexpr size_t shiftStep_ = 4; // 4
  static constexpr size_t numBits_ = 5 * shiftStep_; // 5 // should be divisible by shiftStep
  static constexpr size_t numPredecessors_ = 1 << shiftStep_;
  static constexpr size_t startShnft_ = shiftStep_;
  static constexpr size_t numLevels_ = numBits_ / shiftStep_ + 1;
  static constexpr size_t level0Size_ = 1ul << numBits_;

  using SubGraphType = typename G::SubgraphType;

  std::array<std::vector<size_t>, numLevels_> data_;
  std::array<SubGraphType*, numLevels_> subgraphs_;
  G g_;
};

template <class G>
static void BM_build_big_tree(benchmark::State& state) {
  BigTree<G> bigTree;
  bigTree.allocateMemory();
  for (auto _ : state) {
    bigTree.buildTree();
    bigTree.g_.clear();
  }
}

static void BM_build_bi_prop_dependency_chain(benchmark::State& state) {
  size_t counter;
  for (auto _ : state) {
    dispenso::BiPropGraph graph;
    dispenso::BiPropNode* prevNode = &graph.addNode([&counter]() { counter++; });
    for (size_t i = 1; i < 1024; ++i) {
      dispenso::BiPropNode* node = &graph.addNode([&counter]() { counter++; });
      prevNode->biPropDependsOn(*node);
      prevNode = node;
    }
  }
}

template <class G>
static void BM_build_dependnecy_chain(benchmark::State& state) {
  size_t counter;
  for (auto _ : state) {
    G graph;
    typename G::NodeType* prevNode = &graph.addNode([&counter]() { counter++; });
    for (size_t i = 1; i < 1024; ++i) {
      typename G::NodeType* node = &graph.addNode([&counter]() { counter++; });
      prevNode->dependsOn(*node);
      prevNode = node;
    }
  }
}

template <class G>
static void BM_execute_dependnecy_chain(benchmark::State& state) {
  size_t counter;
  G graph;
  typename G::NodeType* prevNode = &graph.addNode([&counter]() { counter++; });
  for (size_t i = 1; i < 1024; ++i) {
    typename G::NodeType* node = &graph.addNode([&counter]() { counter++; });
    prevNode->dependsOn(*node);
    prevNode = node;
  }

  dispenso::SingleThreadExecutor singleThreadExecutor;
  for (auto _ : state) {
    setAllNodesIncomplete(graph);
    counter = 0;
    singleThreadExecutor(graph);
  }
}

static void BM_build_bi_prop_dependency_group(benchmark::State& state) {
  size_t counter;
  constexpr size_t numNodes = 4;
  std::array<dispenso::BiPropNode*, numNodes> nodes1, nodes2;
  for (auto _ : state) {
    dispenso::BiPropGraph graph;
    for (size_t i = 0; i < numNodes; ++i) {
      nodes1[i] = &graph.addNode([&counter]() { counter++; });
      nodes2[i] = &graph.addNode([&counter]() { counter++; });
      nodes1[i]->biPropDependsOn(*nodes2[i]);
    }
    for (size_t i = 1; i < numNodes; ++i) {
      nodes1[i - 1]->biPropDependsOn(*nodes1[i]);
    }
  }
}

template <class G>
static void BM_forward_propagator_node(benchmark::State& state) {
  G graph;
  constexpr size_t numNodes = 32768;
  std::array<dispenso::Node*, numNodes> nodes;

  size_t counter;

  std::mt19937 rng(12345);

  nodes[0] = &graph.addNode([&counter]() { counter++; }); // root
  for (size_t i = 1; i < numNodes; ++i) {
    nodes[i] = &graph.addNode([&counter]() { counter++; });
    std::uniform_int_distribution<> parentDistr(0, i - 1);
    nodes[i]->dependsOn(*nodes[parentDistr(rng)]);
  }

  dispenso::ForwardPropagator forwardPropagator;

  for (auto _ : state) {
    state.PauseTiming();
    graph.forEachNode([](const dispenso::Node& node) { node.setCompleted(); });

    nodes[0]->setIncomplete();
    state.ResumeTiming();
    forwardPropagator(graph);
  }
}

BENCHMARK(BM_build_big_tree<dispenso::Graph>);
BENCHMARK(BM_build_big_tree<dispenso::BiPropGraph>);
BENCHMARK(BM_build_bi_prop_dependency_chain);
BENCHMARK(BM_build_bi_prop_dependency_group);
BENCHMARK(BM_build_dependnecy_chain<dispenso::Graph>);
BENCHMARK(BM_build_dependnecy_chain<dispenso::BiPropGraph>);
BENCHMARK(BM_execute_dependnecy_chain<dispenso::Graph>);
BENCHMARK(BM_execute_dependnecy_chain<dispenso::BiPropGraph>);
BENCHMARK(BM_forward_propagator_node<dispenso::Graph>);
BENCHMARK(BM_forward_propagator_node<dispenso::BiPropGraph>);

BENCHMARK_MAIN();
