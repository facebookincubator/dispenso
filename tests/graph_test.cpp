/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <numeric>
#include <random>

#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>
#include <gtest/gtest.h>

TEST(Graph, Simple) {
  // ┌─────────┐     ┌─────────┐     ┌─────────┐
  // │ 0: v+=1 │ ──▶ │ 1: v*=2 │ ──▶ │ 4: v+=p │
  // └─────────┘     └─────────┘     └─────────┘
  //                                   ▲
  // ┌─────────┐     ┌─────────┐       │
  // │ 2: p+=4 │ ──▶ │ 3: p/=2 │ ──────┘
  // └─────────┘     └─────────┘
  float v = 0.f;
  float p = 0.f;
  dispenso::Graph g;

  dispenso::Node* N0 = g.addNode([&v]() { v += 1; });
  dispenso::Node* N1 = g.addNode([&v]() { v *= 2; });
  dispenso::Node* N2 = g.addNode([&p]() { p += 8; });
  dispenso::Node* N3 = g.addNode([&p]() { p /= 2; });
  dispenso::Node* N4 = g.addNode([&p, &v]() { v += p; });

  N4->dependsOn(N1, N3);
  N1->dependsOn(N0);
  N3->dependsOn(N2);

  dispenso::ConcurrentTaskSet concurrentTaskSet(dispenso::globalThreadPool());
  dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
  setAllNodesIncomplete(g);
  concurrentTaskSetExecutor(concurrentTaskSet, g);
  EXPECT_EQ(v, 6.f);

  v = 0.f;
  p = 0.f;
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::ParallelForExecutor parallelForExecutor;
  setAllNodesIncomplete(g);
  parallelForExecutor(taskSet, g);
  EXPECT_EQ(v, 6.f);

  dispenso::SingleThreadExecutor singleThreadExecutor;
  v = 0.f;
  p = 0.f;
  setAllNodesIncomplete(g);
  singleThreadExecutor(g);
  EXPECT_EQ(v, 6.f);
}
enum class EvalMode : uint8_t { singleThread, parallelFor, concurrentTaskSet };

template <class T>
std::string modeName(const testing::TestParamInfo<typename T::ParamType>& info) {
  static std::string names[3] = {"singleThread", "parallelFor", "concurrentTaskSet"};
  return names[static_cast<uint8_t>(info.param)];
};

class Executor {
 public:
  Executor()
      : taskSet(dispenso::globalThreadPool()), concurrentTaskSet(dispenso::globalThreadPool()) {}

  template <typename G>
  void operator()(EvalMode mode, const G& graph) {
    if (mode == EvalMode::singleThread) {
      singleThreadExecutor(graph);
    } else if (mode == EvalMode::parallelFor) {
      parallelForExecutor(taskSet, graph);
    } else if (mode == EvalMode::concurrentTaskSet) {
      concurrentTaskSetExecutor(concurrentTaskSet, graph);
    }
  }

 private:
  dispenso::TaskSet taskSet;
  dispenso::ConcurrentTaskSet concurrentTaskSet;
  dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
  dispenso::SingleThreadExecutor singleThreadExecutor;
  dispenso::ParallelForExecutor parallelForExecutor;
};

struct SingleThreadMode {
  static constexpr EvalMode mode = EvalMode::singleThread;
};
struct ParallelForMode {
  static constexpr EvalMode mode = EvalMode::parallelFor;
};
struct ConcurrentTaskSetMode {
  static constexpr EvalMode mode = EvalMode::concurrentTaskSet;
};

class TwoSubgraphs : public testing::TestWithParam<EvalMode> {
 protected:
  void SetUp() override {
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

    subgraph1_ = graph_.addSubgraph();
    subgraph2_ = graph_.addSubgraph();

    N_[0] = subgraph1_->addNode([&]() { r_[0] += 1; });
    N_[2] = subgraph1_->addNode([&]() { r_[2] += 8; });
    N_[1] = subgraph2_->addNode([&]() { r_[1] += r_[0] * 2; });
    N_[3] = subgraph2_->addNode([&]() { r_[3] += r_[2] / 2; });
    N_[4] = graph_.addNode([&]() { r_[4] += r_[1] + r_[3]; });

    N_[4]->dependsOn(N_[1], N_[3]);
    N_[1]->dependsOn(N_[0]);
    N_[3]->dependsOn(N_[2]);
  }

  void evaluateGraph(const dispenso::Graph& graph) {
    executor_(GetParam(), graph);
  }

  std::array<float, 5> r_;
  std::array<dispenso::Node*, 5> N_;
  dispenso::Graph graph_;
  dispenso::Subgraph* subgraph1_;
  dispenso::Subgraph* subgraph2_;
  Executor executor_;
};

TEST_P(TwoSubgraphs, ReplaceSourceGraph) {
  setAllNodesIncomplete(graph_);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  subgraph1_->clear();

  N_[0] = subgraph1_->addNode([&]() { r_[0] += 1; });
  N_[2] = subgraph1_->addNode([&]() { r_[2] += 8; });
  N_[1]->dependsOn(N_[0]);
  N_[3]->dependsOn(N_[2]);

  setAllNodesIncomplete(graph_);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);
}

TEST_P(TwoSubgraphs, ReplaceMiddleGraph) {
  setAllNodesIncomplete(graph_);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  dispenso::Graph movedGraph(std::move(graph_));
  subgraph2_->clear();

  N_[1] = subgraph2_->addNode([&]() { r_[1] += r_[0] * 2; });
  N_[3] = subgraph2_->addNode([&]() { r_[3] += r_[2] / 2; });
  N_[1]->dependsOn(N_[0]);
  N_[3]->dependsOn(N_[2]);
  N_[4]->dependsOn(N_[1], N_[3]);

  setAllNodesIncomplete(movedGraph);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(movedGraph);
  EXPECT_EQ(r_[4], 6.f);
}

TEST_P(TwoSubgraphs, ReplaceBothGraphs) {
  setAllNodesIncomplete(graph_);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  subgraph2_->clear();
  dispenso::Graph movedGraph = std::move(graph_);
  subgraph1_->clear();

  N_[0] = subgraph1_->addNode([&]() { r_[0] += 1; });
  N_[2] = subgraph1_->addNode([&]() { r_[2] += 8; });
  N_[1] = subgraph2_->addNode([&]() { r_[1] += r_[0] * 2; });
  N_[3] = subgraph2_->addNode([&]() { r_[3] += r_[2] / 2; });

  N_[4]->dependsOn(N_[1], N_[3]);
  N_[1]->dependsOn(N_[0]);
  N_[3]->dependsOn(N_[2]);

  setAllNodesIncomplete(movedGraph);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(movedGraph);
  EXPECT_EQ(r_[4], 6.f);
}

TEST_P(TwoSubgraphs, PartialEvaluation) {
  setAllNodesIncomplete(graph_);
  r_ = {0, 0, 0, 0, 0};
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  for (const dispenso::Subgraph& graph : graph_.subgraphs()) {
    const auto& nodes = graph.nodes();
    for (const dispenso::Node& node : nodes) {
      EXPECT_FALSE(!node.isCompleted());
    }
  }

  N_[1]->setIncomplete();
  r_[1] = r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[2]->setIncomplete();
  r_[2] = r_[3] = r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[1]->setIncomplete();
  N_[3]->setIncomplete();
  r_[1] = r_[3] = r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[4]->setIncomplete();
  r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[2]->setIncomplete();
  N_[0]->setIncomplete();
  r_ = {0, 0, 0, 0, 0};
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[0]->setIncomplete();
  N_[1]->setIncomplete();
  N_[4]->setIncomplete();
  r_[0] = r_[1] = r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);

  N_[2]->setIncomplete();
  N_[3]->setIncomplete();
  N_[4]->setIncomplete();
  r_[2] = r_[3] = r_[4] = 0;
  propagateIncompleteState(graph_);
  evaluateGraph(graph_);
  EXPECT_EQ(r_[4], 6.f);
}

INSTANTIATE_TEST_SUITE_P(
    GraphEvaluation,
    TwoSubgraphs,
    testing::Values(EvalMode::singleThread, EvalMode::parallelFor, EvalMode::concurrentTaskSet),
    modeName<TwoSubgraphs>);

class BiPropGraphTest : public testing::TestWithParam<EvalMode> {
 protected:
  void SetUp() override {
    //                                     ┌─────────────────┐
    //                                     │ 5: m5+=m3*2     │
    //                                     └─────────────────┘
    //                                       ▲
    // ┌−−−−−−−−−−−−−−−−Group1 (a,b)−−−−−−−−−│−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
    // ╎ ┌─────────┐     ┌───────────┐     ┌─────────────────┐     ┌────────────┐ ╎
    // ╎ │         │     │  0: a+=1  │     │     3: b+=5     │     │            │ ╎
    // ╎ │ 2: a+=3 │     │   b+=5    │     │     m3+=b       │     │ 6: b/=m4   │ ╎
    // ╎ │         │ ◁── │ m0+=a+b   │ ──▷ │                 │ ──▷ │            │ ╎
    // ╎ └─────────┘     └───────────┘     └─────────────────┘     └────────────┘ ╎
    // └−−−−−−−−−−−−−−−−−−−│−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−▲−−−−−−−−−−−−┘
    //                     └─────────────────┐                       │
    //                 ┌−−−−−−−Group2 (c)−−−−▼−−−−−−−−−−−−−−−−−┐     │
    //                 ╎ ┌───────────┐     ┌─────────────────┐ ╎   ┌────────────┐
    //                 ╎ │  7: c+=5  │ ──▷ │   1: c+=m0      │ ╎   │ 4: m4+=2   │
    //                 ╎ └───────────┘     └─────────────────┘ ╎   └────────────┘
    //                 └−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
    //  Legend:
    //  ──▶ Normal dependency
    //  ──▷ Bidirectional propagation dependency
    //  m4  variable modified only in node 4
    N[0] = g.addNode([&]() {
      a += 1;
      b += 5;
      m0 += a + b;
    });
    N[1] = g.addNode([&]() { c += m0; });
    N[2] = g.addNode([&]() { a += 3; });
    N[3] = g.addNode([&]() {
      b += 5;
      m3 += b;
    });
    N[4] = g.addNode([&]() { m4 += 2; });
    N[5] = g.addNode([&]() { m5 += m3 * 2; });
    N[6] = g.addNode([&]() { b /= m4; });
    N[7] = g.addNode([&]() { c += 5; });

    N[5]->dependsOn(N[3]);
    N[1]->dependsOn(N[0]);
    N[6]->dependsOn(N[4]);

    N[2]->biPropDependsOn(N[0]);
    N[3]->biPropDependsOn(N[0]);
    N[6]->biPropDependsOn(N[3]);
    N[1]->biPropDependsOn(N[7]);
  }

  void checkResults() {
    EXPECT_EQ(a, 4.f);
    EXPECT_EQ(b, 5.f);
    EXPECT_EQ(c, 11.f);
    EXPECT_EQ(m5, 20.f);
  }

  float a, b, c, m0, m3, m4, m5;
  dispenso::BiPropGraph g;
  std::array<dispenso::BiPropNode*, 8> N;
  Executor executor;
};

TEST_P(BiPropGraphTest, SimpleEvaluation) {
  a = b = c = m0 = m3 = m4 = m5 = 0.f;
  setAllNodesIncomplete(g);
  executor(GetParam(), g);

  checkResults();

  a = b = c = m0 = m3 = m4 = m5 = 0.f;
  setAllNodesIncomplete(g);
  N[4]->setIncomplete();
  executor(GetParam(), g);
  checkResults();

  N[4]->setIncomplete();
  m4 = m3 = m0 = a = b = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();

  N[1]->setIncomplete();
  c = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();

  N[7]->setIncomplete();
  c = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();

  N[0]->setIncomplete();
  a = b = c = m0 = m3 = m5 = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();

  N[5]->setIncomplete();
  m5 = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();

  N[4]->setIncomplete();
  N[6]->setIncomplete();
  m4 = m3 = m0 = a = b = 0.f;
  propagateIncompleteState(g);
  executor(GetParam(), g);
  checkResults();
}

INSTANTIATE_TEST_SUITE_P(
    BiPropEvaluation,
    BiPropGraphTest,
    testing::Values(EvalMode::singleThread, EvalMode::parallelFor, EvalMode::concurrentTaskSet),
    modeName<BiPropGraphTest>);

template <class T>
class BigTree : public testing::Test {
 protected:
  static size_t sizeOfLevel(size_t level) {
    return size_t(1) << (numBits_ - shiftStep_ * level);
  }
  void SetUp() override {
    // the goal of this task is to calculate the sum of the numbers in this array
    for (size_t level = 1; level < numLevels_; ++level) {
      const size_t s = sizeOfLevel(level);
      data_[level].resize(s, size_t(0));
      nodes_[level].resize(s);
    }

    // zero level data is input data for calcualtion
    data_[0].resize(size_t(1) << numBits_);
    std::iota(data_[0].begin(), data_[0].end(), 0);

    for (size_t level = 1; level < numLevels_; ++level) {
      SubGraphType* subgraph = g_.addSubgraph();
      levelSubgraphs_[level] = subgraph;
      buildLevel(level);
    }
  }

  void rebuildLevel(size_t level) {
    buildLevel(level);
    if (level == numLevels_ - 1) {
      return;
    }

    const size_t level1 = level + 1;
    const size_t numNodes = size_t(1) << (numBits_ - shiftStep_ * level1);

    for (size_t n = 0; n < numNodes; ++n) {
      if (level1 > 1) {
        for (size_t j = 0; j < numPredecessors_; ++j) {
          const size_t index = (n << shiftStep_) | j;
          nodes_[level1][n]->dependsOn(nodes_[level1 - 1][index]);
        }
      }
    }
  }

  void buildLevel(size_t level) {
    SubGraphType* subgraph = levelSubgraphs_[level];

    const size_t numNodes = size_t(1) << (numBits_ - shiftStep_ * level);

    for (size_t n = 0; n < numNodes; ++n) {
      nodes_[level][n] = subgraph->addNode([this, level, n]() {
        for (size_t j = 0; j < numPredecessors_; ++j) {
          const size_t index = (n << shiftStep_) | j;
          data_[level][n] += data_[level - 1][index];
        }
      });
      if (level > 1) {
        for (size_t j = 0; j < numPredecessors_; ++j) {
          const size_t index = (n << shiftStep_) | j;
          nodes_[level][n]->dependsOn(nodes_[level - 1][index]);
        }
      }
    }
  }
  static constexpr size_t shiftStep_ = 4; // 4
  static constexpr size_t numBits_ = 5 * shiftStep_; // 5 // should be divisible by shiftStep
  static constexpr size_t numPredecessors_ = 1 << shiftStep_;
  static constexpr size_t startShnft_ = shiftStep_;
  static constexpr size_t numLevels_ = numBits_ / shiftStep_ + 1;
  static constexpr size_t level0Size_ = size_t(1) << numBits_;

  using GraphType = std::tuple_element_t<0, T>;
  using SubGraphType = typename GraphType::SubgraphType;
  static constexpr EvalMode mode_ = std::tuple_element_t<1, T>::mode;

  std::array<std::vector<size_t>, numLevels_> data_;
  std::array<SubGraphType*, numLevels_> levelSubgraphs_;
  std::array<std::vector<typename GraphType::NodeType*>, numLevels_> nodes_;
  GraphType g_;
  Executor executor_;
};

using GraphCases = ::testing::Types<
    std::tuple<dispenso::Graph, SingleThreadMode>,
    std::tuple<dispenso::Graph, ParallelForMode>,
    std::tuple<dispenso::Graph, ConcurrentTaskSetMode>,
    std::tuple<dispenso::BiPropGraph, SingleThreadMode>,
    std::tuple<dispenso::BiPropGraph, ParallelForMode>,
    std::tuple<dispenso::BiPropGraph, ConcurrentTaskSetMode>>;

TYPED_TEST_SUITE(BigTree, GraphCases, );

TYPED_TEST(BigTree, FullAndPartialEvaluation) {
  size_t result = std::accumulate(this->data_[0].begin(), this->data_[0].end(), size_t(0));

  setAllNodesIncomplete(this->g_);
  this->executor_(this->mode_, this->g_);

  EXPECT_EQ(this->data_[this->numLevels_ - 1][0], result);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, this->level0Size_ - 1);
  std::uniform_int_distribution<std::mt19937::result_type> numValuesDist(1, this->level0Size_ / 16);

  const size_t numExperiments = 20;
  for (uint32_t i = 0; i < numExperiments; ++i) {
    const size_t numValues = numValuesDist(rng);

    for (size_t j = 0; j < numValues; ++j) {
      const size_t dataIndex = dist(rng);
      const size_t node1Index = dataIndex >> this->shiftStep_;
      this->data_[0][dataIndex] = dist(rng);
      this->nodes_[1][node1Index]->setIncomplete();
      // clean up nodes data. We don't touch other nodes' data so that the test fail if we evaluated
      // more than necessary.
      for (size_t level = 1; level < this->numLevels_; ++level) {
        this->data_[level][dataIndex >> (this->shiftStep_ * level)] = 0;
      }
    }

    result = std::accumulate(this->data_[0].begin(), this->data_[0].end(), size_t(0));

    propagateIncompleteState(this->g_);
    this->executor_(this->mode_, this->g_);
    EXPECT_EQ(this->data_[this->numLevels_ - 1][0], result);
  }
}

TYPED_TEST(BigTree, SubgraphClearAndRebuild) {
  size_t result = std::accumulate(this->data_[0].begin(), this->data_[0].end(), size_t(0));

  setAllNodesIncomplete(this->g_);
  this->executor_(this->mode_, this->g_);
  EXPECT_EQ(this->data_[this->numLevels_ - 1][0], result);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> distLevel(1, this->numLevels_ - 1);

  const size_t numExperiments = 20;
  for (uint32_t i = 0; i < numExperiments; ++i) {
    const size_t level = distLevel(rng);

    this->levelSubgraphs_[level]->clear();
    this->rebuildLevel(level);
    propagateIncompleteState(this->g_);
    // clean up nodes data.
    for (size_t l = level; l < this->numLevels_; ++l) {
      std::fill(this->data_[l].begin(), this->data_[l].end(), size_t(0));
    }

    this->executor_(this->mode_, this->g_);

    EXPECT_EQ(this->data_[this->numLevels_ - 1][0], result);
  }
}
