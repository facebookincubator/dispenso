/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph.h>
#include <fstream>
#include <string>
#include <unordered_map>

namespace detail {
inline std::string getName(
    const void* ptr,
    const size_t index,
    const std::unordered_map<uintptr_t, std::string>* nodeNames) {
  const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  if (nodeNames) {
    auto it = nodeNames->find(key);
    if (it != nodeNames->end()) {
      return it->second;
    }
  }
  return std::to_string(index);
}
} // namespace detail

namespace dispenso {
template <typename G>
void graphsToDot(
    const char* filename,
    const G& graph,
    const std::unordered_map<uintptr_t, std::string>* nodeNames) {
  using SubgraphType = typename G::SubgraphType;
  using NodeType = typename G::NodeType;
  std::ofstream datfile(filename);
  datfile << R"dot(digraph {
  rankdir = LR
  node [shape = rectangle, style = filled, colorscheme=pastel19]
  graph [style = filled, color = Gray95]

  subgraph cluster_l { label = "Legend"; style=solid; color=black
    empty1 [style = invis, shape=point]
    empty2 [style = invis, shape=point]
    incomplete [color = 1]
    completed [color = 2]
    incomplete -> empty1 [label = "normal"]
    completed -> empty2 [arrowhead = onormal,label = "bidirectional\lpropagation"]
  }
)dot";

  const size_t numSubgraphs = graph.numSubgraphs();
  for (size_t i = 0; i < numSubgraphs; ++i) {
    const SubgraphType& s = graph.subgraph(i);
    if (i != 0) {
      datfile << "  " << "subgraph cluster_" << i << " { label = \""
              << ::detail::getName(&s, i, nodeNames) << "\"\n";
    }
    const size_t numNodes = s.numNodes();
    for (size_t j = 0; j < numNodes; ++j) {
      const NodeType& node = s.node(j);
      datfile << "    " << reinterpret_cast<uintptr_t>(&node)
              << " [color = " << (node.isCompleted() ? 2 : 1);
      datfile << " label = \"" << ::detail::getName(&node, j, nodeNames) << "\"]\n";
    }

    if (i != 0) {
      datfile << "  }\n";
    }
  }

  graph.forEachNode([&](const NodeType& node) {
    node.forEachDependent([&](const dispenso::Node& d) {
      datfile << "    " << reinterpret_cast<uintptr_t>(&node) << " -> "
              << reinterpret_cast<uintptr_t>(&d);

      if (std::is_same<dispenso::BiPropNode, NodeType>::value) {
        const auto& node1 = static_cast<const dispenso::BiPropNode&>(node);
        const auto& node2 = static_cast<const dispenso::BiPropNode&>(d);
        datfile << (node1.isSameSet(node2) ? "[arrowhead=onormal]" : "");
      }
      datfile << '\n';
    });
  });

  datfile << "}";
  datfile.close();
}
} // namespace dispenso
