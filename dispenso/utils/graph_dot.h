/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file graph_dot.h
 * Utility for writing a <code>Graph</code> or <code>BiPropGraph</code> as a Graphviz DOT file.
 **/

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <dispenso/graph.h>

namespace dispenso {
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

/**
 * Write <code>graph</code> to <code>filename</code> in Graphviz DOT format, for example to render
 * with <code>dot -Tsvg</code>. Each subgraph after the first is drawn as a cluster, completed nodes
 * are coloured differently from incomplete ones, and for a <code>BiPropGraph</code> edges within
 * a bidirectional-propagation set are drawn with open arrowheads.
 *
 * @param filename path of the file to write; it is overwritten.
 * @param graph the graph to write.
 * @param nodeNames optional labels, keyed by the address of a node or subgraph cast to
 *   <code>uintptr_t</code>. Nodes and subgraphs without an entry are labelled by their index.
 **/
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
              << detail::getName(&s, i, nodeNames) << "\"\n";
    }
    const size_t numNodes = s.numNodes();
    for (size_t j = 0; j < numNodes; ++j) {
      const NodeType& node = s.node(j);
      datfile << "    " << reinterpret_cast<uintptr_t>(&node)
              << " [color = " << (node.isCompleted() ? 2 : 1);
      datfile << " label = \"" << detail::getName(&node, j, nodeNames) << "\"]\n";
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
