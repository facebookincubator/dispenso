/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>
#include <dispenso/utils/graph_dot.h>
// Included twice on purpose: the header must be safe to include more than once.
#include <dispenso/utils/graph_dot.h>

#include <gtest/gtest.h>

namespace {

// Reads the file and deletes it.
std::string readFile(const std::string& path) {
  std::stringstream ss;
  {
    std::ifstream in(path);
    ss << in.rdbuf();
  }
  std::remove(path.c_str());
  return ss.str();
}

std::string idOf(const void* p) {
  return std::to_string(reinterpret_cast<uintptr_t>(p));
}

size_t count(const std::string& haystack, const std::string& needle) {
  size_t n = 0;
  for (size_t pos = haystack.find(needle); pos != std::string::npos;
       pos = haystack.find(needle, pos + needle.size())) {
    ++n;
  }
  return n;
}

} // namespace

TEST(GraphDot, WritesNodesEdgesAndClusters) {
  dispenso::Graph graph;
  dispenso::Node& a = graph.addNode([]() {});
  dispenso::Node& b = graph.addNode([]() {});
  dispenso::Subgraph& sub = graph.addSubgraph();
  dispenso::Node& c = sub.addNode([]() {});
  b.dependsOn(a);
  c.dependsOn(b);

  const std::string path = ::testing::TempDir() + "graph_dot_test_plain.dot";
  dispenso::graphsToDot(path.c_str(), graph, nullptr);
  const std::string dot = readFile(path);

  EXPECT_EQ(dot.rfind("digraph {", 0), 0u);
  EXPECT_EQ(dot.back(), '}');
  // Without names, nodes are labelled by their index within their subgraph.
  EXPECT_NE(dot.find(idOf(&a) + " [color = 1 label = \"0\"]"), std::string::npos);
  EXPECT_NE(dot.find(idOf(&b) + " [color = 1 label = \"1\"]"), std::string::npos);
  EXPECT_NE(dot.find(idOf(&c) + " [color = 1 label = \"0\"]"), std::string::npos);
  // Only subgraphs after the first become clusters (the legend is one more).
  EXPECT_EQ(count(dot, "subgraph cluster_"), 2u);
  EXPECT_NE(dot.find("subgraph cluster_1 { label = \"1\""), std::string::npos);
  // One line per dependency edge.
  EXPECT_NE(dot.find(idOf(&a) + " -> " + idOf(&b) + "\n"), std::string::npos);
  EXPECT_NE(dot.find(idOf(&b) + " -> " + idOf(&c) + "\n"), std::string::npos);
  EXPECT_EQ(count(dot, " -> "), 2u + 2u); // two edges, plus two in the legend
}

TEST(GraphDot, UsesNamesAndCompletionState) {
  dispenso::Graph graph;
  dispenso::Node& a = graph.addNode([]() {});
  dispenso::Subgraph& sub = graph.addSubgraph();
  dispenso::Node& b = sub.addNode([]() {});
  b.dependsOn(a);

  setAllNodesIncomplete(graph);
  dispenso::SingleThreadExecutor executor;
  executor(graph);

  std::unordered_map<uintptr_t, std::string> names = {
      {reinterpret_cast<uintptr_t>(&a), "load"},
      {reinterpret_cast<uintptr_t>(&b), "process"},
      {reinterpret_cast<uintptr_t>(&sub), "stage two"},
  };
  const std::string path = ::testing::TempDir() + "graph_dot_test_named.dot";
  dispenso::graphsToDot(path.c_str(), graph, &names);
  const std::string dot = readFile(path);

  // Completed nodes use color 2.
  EXPECT_NE(dot.find(idOf(&a) + " [color = 2 label = \"load\"]"), std::string::npos);
  EXPECT_NE(dot.find(idOf(&b) + " [color = 2 label = \"process\"]"), std::string::npos);
  EXPECT_NE(dot.find("label = \"stage two\""), std::string::npos);
}

TEST(GraphDot, MarksBidirectionalPropagationSets) {
  dispenso::BiPropGraph graph;
  dispenso::BiPropNode& a = graph.addNode([]() {});
  dispenso::BiPropNode& b = graph.addNode([]() {});
  dispenso::BiPropNode& c = graph.addNode([]() {});
  b.biPropDependsOn(a);
  c.dependsOn(b);

  const std::string path = ::testing::TempDir() + "graph_dot_test_biprop.dot";
  dispenso::graphsToDot(path.c_str(), graph, nullptr);
  const std::string dot = readFile(path);

  EXPECT_NE(dot.find(idOf(&a) + " -> " + idOf(&b) + "[arrowhead=onormal]\n"), std::string::npos);
  EXPECT_NE(dot.find(idOf(&b) + " -> " + idOf(&c) + "\n"), std::string::npos);
}
