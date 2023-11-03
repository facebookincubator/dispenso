/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph.h>
#include <fstream>

template <typename... Gs>
void graphsToDot(const char* filename, const Gs&... graphs) {
  std::ofstream datfile(filename);
  datfile << R"dot(
digraph {
  rankdir = LR
  node [shape = rectangle, style = filled, colorscheme=pastel19]
  )dot";

  for (const dispenso::Graph* graph : {&graphs...}) {
    datfile << "  "
            << "subgraph cluster_" << (uint64_t)graph << " { label = \"" << (uint64_t)(graph)
            << "\"\n";
    for (const dispenso::Node& node : graph->nodes()) {
      datfile << "    " << (uint64_t)(&node) << " [color = " << (node.isCompleted() ? 2 : 1)
              << " label = \"" << (uint64_t)(&node) << "\"]\n";
    }
    datfile << "  }\n";

    for (const dispenso::Node& node : graph->nodes()) {
      for (const dispenso::Node* d : node.dependents()) {
        datfile << "    " << (uint64_t)(&node) << " -> " << (uint64_t)d << '\n';
      }
    }
  }
  datfile << "}";
  datfile.close();
}
