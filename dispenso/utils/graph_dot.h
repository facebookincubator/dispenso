/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph.h>
#include <fstream>

template <typename G>
void graphsToDot(const char* filename, const G& graph) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;
  std::ofstream datfile(filename);
  datfile << R"dot(
digraph {
  rankdir = LR
  node [shape = rectangle, style = filled, colorscheme=pastel19]
  )dot";

  const std::deque<SubgraphType>& subgraphs = graph.subgraphs();
  printf("subgraphs.size() %lu\n", subgraphs.size()); // DebugCode
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    const SubgraphType& s = subgraphs[i];
    if (i != 0) {
      datfile << "  "
              << "subgraph cluster_" << (uint64_t)(&s) << " { label = \"" << (uint64_t)(&s)
              << "\"\n";
    }
    for (const dispenso::Node& node : s.nodes()) {
      // datfile << "    " << (uint64_t)(&node) << " [color = " << (node.isCompleted() ? 2 : 1)
      datfile << "    " << (uint64_t)(&node) << " [color = " << (i + 1) << " label = \""
              << (uint64_t)(&node) << "\"]\n";
    }
    if (i != 0) {
      datfile << "  }\n";
    }
  }
  for (const SubgraphType& subgraph : graph.subgraphs()) {
    for (const NodeType& node : subgraph.nodes()) {
      for (const dispenso::Node* d : node.dependents()) {
        datfile << "    " << (uint64_t)(&node) << " -> " << (uint64_t)d << '\n';
      }
    }
  }

  datfile << "}";
  datfile.close();
}
