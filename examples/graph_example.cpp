/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example graph_example.cpp
 * Demonstrates task graphs with dependencies using dispenso::Graph.
 */

#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>

#include <array>
#include <iostream>

int main() {
  // Example 1: Simple linear dependency chain
  std::cout << "Example 1: Linear dependency chain (A -> B -> C)\n";
  {
    std::array<int, 3> values = {0, 0, 0};

    dispenso::Graph graph;

    dispenso::Node& nodeA = graph.addNode([&]() {
      values[0] = 1;
      std::cout << "  Node A: set values[0] = 1\n";
    });

    dispenso::Node& nodeB = graph.addNode([&]() {
      values[1] = values[0] * 2;
      std::cout << "  Node B: set values[1] = values[0] * 2 = " << values[1] << "\n";
    });

    dispenso::Node& nodeC = graph.addNode([&]() {
      values[2] = values[1] + 10;
      std::cout << "  Node C: set values[2] = values[1] + 10 = " << values[2] << "\n";
    });

    // Set up dependencies: B depends on A, C depends on B
    nodeB.dependsOn(nodeA);
    nodeC.dependsOn(nodeB);

    // Initialize predecessor counts and execute the graph
    setAllNodesIncomplete(graph);
    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
    dispenso::ConcurrentTaskSetExecutor executor;
    executor(taskSet, graph);

    std::cout << "  Final values: " << values[0] << ", " << values[1] << ", " << values[2] << "\n";
  }

  // Example 2: Diamond dependency pattern
  std::cout << "\nExample 2: Diamond dependency pattern\n";
  {
    //         A
    //        / \
    //       B   C
    //        \ /
    //         D
    std::array<float, 5> r = {0, 0, 0, 0, 0};

    dispenso::Graph graph;

    dispenso::Node& A = graph.addNode([&]() {
      r[0] = 1.0f;
      std::cout << "  A: r[0] = " << r[0] << "\n";
    });

    dispenso::Node& B = graph.addNode([&]() {
      r[1] = r[0] * 2.0f;
      std::cout << "  B: r[1] = r[0] * 2 = " << r[1] << "\n";
    });

    dispenso::Node& C = graph.addNode([&]() {
      r[2] = r[0] + 5.0f;
      std::cout << "  C: r[2] = r[0] + 5 = " << r[2] << "\n";
    });

    dispenso::Node& D = graph.addNode([&]() {
      r[3] = r[1] + r[2];
      std::cout << "  D: r[3] = r[1] + r[2] = " << r[3] << "\n";
    });

    // Diamond pattern dependencies
    B.dependsOn(A);
    C.dependsOn(A);
    D.dependsOn(B, C);

    // Initialize predecessor counts and execute
    setAllNodesIncomplete(graph);
    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
    dispenso::ConcurrentTaskSetExecutor executor;
    executor(taskSet, graph);

    std::cout << "  Expected r[3] = 2 + 6 = 8, got: " << r[3] << "\n";
  }

  // Example 3: Graph re-execution with setAllNodesIncomplete
  std::cout << "\nExample 3: Re-executing a graph\n";
  {
    int counter = 0;

    dispenso::Graph graph;

    dispenso::Node& node = graph.addNode([&]() {
      counter++;
      std::cout << "  Executed, counter = " << counter << "\n";
    });
    (void)node;

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
    dispenso::ConcurrentTaskSetExecutor executor;

    // First execution - need to initialize predecessor counts
    std::cout << "  First execution:\n";
    setAllNodesIncomplete(graph);
    executor(taskSet, graph);

    // Reset and execute again
    std::cout << "  Second execution (after setAllNodesIncomplete):\n";
    setAllNodesIncomplete(graph);
    executor(taskSet, graph);

    std::cout << "  Final counter = " << counter << " (expected: 2)\n";
  }

  // Example 4: Using subgraphs
  std::cout << "\nExample 4: Using subgraphs\n";
  {
    std::array<float, 4> r = {0, 0, 0, 0};

    dispenso::Graph graph;

    // Create two subgraphs for different parts of computation
    dispenso::Subgraph& inputSubgraph = graph.addSubgraph();
    dispenso::Subgraph& processSubgraph = graph.addSubgraph();

    // Input nodes
    dispenso::Node& inputA = inputSubgraph.addNode([&]() { r[0] = 10.0f; });
    dispenso::Node& inputB = inputSubgraph.addNode([&]() { r[1] = 20.0f; });

    // Processing nodes
    dispenso::Node& sum = processSubgraph.addNode([&]() { r[2] = r[0] + r[1]; });

    // Output node in main graph
    dispenso::Node& output = graph.addNode([&]() { r[3] = r[2] * 2.0f; });

    // Set dependencies
    sum.dependsOn(inputA, inputB);
    output.dependsOn(sum);

    // Initialize predecessor counts and execute
    setAllNodesIncomplete(graph);
    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
    dispenso::ConcurrentTaskSetExecutor executor;
    executor(taskSet, graph);

    std::cout << "  r[0]=" << r[0] << ", r[1]=" << r[1] << ", r[2]=" << r[2] << ", r[3]=" << r[3]
              << "\n";
    std::cout << "  Expected r[3] = (10+20)*2 = 60\n";
  }

  // Example 5: Partial graph re-evaluation
  std::cout << "\nExample 5: Partial graph re-evaluation\n";
  {
    std::array<int, 3> data = {1, 0, 0};
    std::array<int, 3> execCount = {0, 0, 0};

    dispenso::Graph graph;

    dispenso::Node& A = graph.addNode([&]() {
      data[0] = data[0] * 2;
      execCount[0]++;
    });

    dispenso::Node& B = graph.addNode([&]() {
      data[1] = data[0] + 10;
      execCount[1]++;
    });

    dispenso::Node& C = graph.addNode([&]() {
      data[2] = data[1] + 100;
      execCount[2]++;
    });

    B.dependsOn(A);
    C.dependsOn(B);

    dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());
    dispenso::ConcurrentTaskSetExecutor executor;

    // First execution - initialize predecessor counts
    setAllNodesIncomplete(graph);
    executor(taskSet, graph);
    std::cout << "  After first run: data = [" << data[0] << ", " << data[1] << ", " << data[2]
              << "]\n";

    // Mark only B as incomplete (and propagate to C)
    B.setIncomplete();
    dispenso::ForwardPropagator propagator;
    propagator(graph);

    // Change input to B
    data[0] = 5;

    // Re-execute - only B and C should run
    executor(taskSet, graph);
    std::cout << "  After partial run: data = [" << data[0] << ", " << data[1] << ", " << data[2]
              << "]\n";
    std::cout << "  Execution counts: A=" << execCount[0] << ", B=" << execCount[1]
              << ", C=" << execCount[2] << "\n";
  }

  std::cout << "\nAll Graph examples completed successfully!\n";
  return 0;
}
