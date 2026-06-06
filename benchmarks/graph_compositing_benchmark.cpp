/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark: Multi-layer compositing graph.
//
// Models a scalable rendering compositing pipeline where N independent render
// layers (characters, environment, effects, UI elements) are processed through
// individual chains and merged into a final composite.
//
// Structure (kNumLayers layers, kStagesPerLayer stages each):
//
//   Layer 0: Source → Stage A → Stage B → Stage C ──┐
//   Layer 1: Source → Stage A → Stage B → Stage C ──┤
//   Layer 2: Source → Stage A → Stage B → Stage C ──├→ Merge → Post A → Post B → Output
//   ...                                              │
//   Layer N: Source → Stage A → Stage B → Stage C ──┘
//
// Characteristics:
// - N independent subgraphs (layers) with no inter-layer dependencies
// - Diamond join at the merge node (all layers → merge)
// - Heterogeneous work per stage (source cheap, processing medium/expensive)
// - Meaningful partial re-evaluation: dirtying one layer re-evaluates only
//   that layer's chain + merge + post-process, leaving other layers untouched
// - Scalable: total nodes = N * kStagesPerLayer + kPostStages
//
// Complements the scene_graph benchmark (deep tree, many small nodes) by
// testing wide independent fan-out with meaningful partial evaluation.

#include <benchmark/benchmark.h>
#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>
#include <dispenso/task_set.h>

#include <taskflow/taskflow.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Graph shape parameters
constexpr size_t kNumLayers = 50; // independent render layers
constexpr size_t kStagesPerLayer = 5; // processing stages per layer
constexpr size_t kPostStages = 3; // post-merge sequential stages
// Total nodes = 50 * 5 + 1 (merge) + 3 (post) = 254

// Work per stage (iterations of inner loop). Tuned so total graph takes ~5-15ms
// on 166-thread EPYC — enough for scheduling overhead to be visible.
constexpr size_t kSourceWork = 32; // cheap: generate source data
constexpr size_t kProcessWork = 128; // medium: per-layer processing
constexpr size_t kExpensiveWork = 512; // expensive: one heavy stage per layer
constexpr size_t kMergeWork = 64; // merge all layers
constexpr size_t kPostWork = 64; // post-process

// Per-node data buffer. Small enough to fit in L2 but large enough for
// measurable work.
constexpr size_t kBufferSize = 4096;
using Buffer = std::array<float, kBufferSize>;

struct LayerData {
  std::array<Buffer, kStagesPerLayer> stageBuffers;
};

struct FrameData {
  std::vector<LayerData> layers;
  Buffer mergeBuffer;
  std::array<Buffer, kPostStages> postBuffers;
  Buffer outputBuffer;

  FrameData() : layers(kNumLayers) {}
};

// Simulate compute work: iterative refinement on buffer data.
void processBuffer(Buffer& output, const Buffer& input, size_t iterations) {
  float acc = 0.0f;
  for (size_t iter = 0; iter < iterations; ++iter) {
    for (size_t i = 0; i < 64; ++i) {
      acc += input[i] * 0.99f + 0.01f;
    }
  }
  output[0] = acc;
  for (size_t i = 1; i < kBufferSize; i += 16) {
    output[i] = input[i] * 0.5f + output[i - 1] * 0.5f;
  }
}

// Merge N layer outputs into one buffer
void mergeBuffers(Buffer& output, const std::vector<LayerData>& layers, size_t iterations) {
  output.fill(0.0f);
  for (const auto& layer : layers) {
    const auto& lastStage = layer.stageBuffers[kStagesPerLayer - 1];
    for (size_t i = 0; i < kBufferSize; i += 16) {
      output[i] += lastStage[i];
    }
  }
  float acc = 0.0f;
  for (size_t iter = 0; iter < iterations; ++iter) {
    for (size_t i = 0; i < 64; ++i) {
      acc += output[i] * 0.99f + 0.01f;
    }
  }
  output[0] = acc;
}

// Work cost for each stage within a layer (varies by stage index)
size_t workForStage(size_t stageIdx) {
  if (stageIdx == 0) {
    return kSourceWork; // cheap source generation
  }
  if (stageIdx == kStagesPerLayer / 2) {
    return kExpensiveWork; // one expensive stage per layer
  }
  return kProcessWork; // medium processing
}

// Build the compositing graph (works for both Graph and BiPropGraph)
template <typename GraphT>
void buildCompositingGraph(GraphT& g, FrameData& frame) {
  // Reserve space to avoid reallocation
  size_t totalNodes = kNumLayers * kStagesPerLayer + 1 + kPostStages;
  g.reserve(totalNodes);

  // Per-layer processing chains
  // Store references to last node in each layer for merge dependencies
  using NodeRef = decltype(&g.addNode([]() {}));
  std::vector<NodeRef> layerOutputNodes(kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    NodeRef prevNode = nullptr;
    for (size_t stage = 0; stage < kStagesPerLayer; ++stage) {
      size_t work = workForStage(stage);
      auto& node = g.addNode([&frame, layer, stage, work]() {
        if (stage == 0) {
          // Source: initialize from scratch
          auto& buf = frame.layers[layer].stageBuffers[0];
          for (size_t i = 0; i < kBufferSize; i += 16) {
            buf[i] = static_cast<float>(layer * kStagesPerLayer + stage) * 0.001f + 1.0f;
          }
          // Do some work
          float acc = 0.0f;
          for (size_t iter = 0; iter < work; ++iter) {
            for (size_t i = 0; i < 64; ++i) {
              acc += buf[i] * 0.99f + 0.01f;
            }
          }
          buf[0] = acc;
        } else {
          processBuffer(
              frame.layers[layer].stageBuffers[stage],
              frame.layers[layer].stageBuffers[stage - 1],
              work);
        }
      });

      if (prevNode) {
        // Chain: previous stage → this stage
        node.dependsOn(*prevNode);
      }
      prevNode = &node;
    }
    layerOutputNodes[layer] = prevNode;
  }

  // Merge node: depends on all layer outputs
  auto& mergeNode =
      g.addNode([&frame]() { mergeBuffers(frame.mergeBuffer, frame.layers, kMergeWork); });
  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    mergeNode.dependsOn(*layerOutputNodes[layer]);
  }

  // Post-process chain
  NodeRef prevPost = &mergeNode;
  for (size_t post = 0; post < kPostStages; ++post) {
    auto& postNode = g.addNode([&frame, post]() {
      if (post == 0) {
        processBuffer(frame.postBuffers[0], frame.mergeBuffer, kPostWork);
      } else {
        processBuffer(frame.postBuffers[post], frame.postBuffers[post - 1], kPostWork);
      }
    });
    postNode.dependsOn(*prevPost);
    prevPost = &postNode;
  }
}

// BiPropGraph version uses biPropDependsOn instead of dependsOn
template <>
void buildCompositingGraph<dispenso::BiPropGraph>(dispenso::BiPropGraph& g, FrameData& frame) {
  // Reserve space to avoid reallocation
  size_t totalNodes = kNumLayers * kStagesPerLayer + 1 + kPostStages;
  g.reserve(totalNodes);

  using NodeRef = dispenso::BiPropNode*;
  std::vector<NodeRef> layerOutputNodes(kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    NodeRef prevNode = nullptr;
    for (size_t stage = 0; stage < kStagesPerLayer; ++stage) {
      size_t work = workForStage(stage);
      auto& node = g.addNode([&frame, layer, stage, work]() {
        if (stage == 0) {
          auto& buf = frame.layers[layer].stageBuffers[0];
          for (size_t i = 0; i < kBufferSize; i += 16) {
            buf[i] = static_cast<float>(layer * kStagesPerLayer + stage) * 0.001f + 1.0f;
          }
          float acc = 0.0f;
          for (size_t iter = 0; iter < work; ++iter) {
            for (size_t i = 0; i < 64; ++i) {
              acc += buf[i] * 0.99f + 0.01f;
            }
          }
          buf[0] = acc;
        } else {
          processBuffer(
              frame.layers[layer].stageBuffers[stage],
              frame.layers[layer].stageBuffers[stage - 1],
              work);
        }
      });

      if (prevNode) {
        node.biPropDependsOn(*prevNode);
      }
      prevNode = &node;
    }
    layerOutputNodes[layer] = prevNode;
  }

  auto& mergeNode =
      g.addNode([&frame]() { mergeBuffers(frame.mergeBuffer, frame.layers, kMergeWork); });
  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    mergeNode.biPropDependsOn(*layerOutputNodes[layer]);
  }

  NodeRef prevPost = &mergeNode;
  for (size_t post = 0; post < kPostStages; ++post) {
    auto& postNode = g.addNode([&frame, post]() {
      if (post == 0) {
        processBuffer(frame.postBuffers[0], frame.mergeBuffer, kPostWork);
      } else {
        processBuffer(frame.postBuffers[post], frame.postBuffers[post - 1], kPostWork);
      }
    });
    postNode.biPropDependsOn(*prevPost);
    prevPost = &postNode;
  }
}

// Build equivalent Taskflow graph
void buildCompositingGraphTF(tf::Taskflow& taskflow, FrameData& frame) {
  std::vector<tf::Task> layerOutputTasks(kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    tf::Task prevTask;
    for (size_t stage = 0; stage < kStagesPerLayer; ++stage) {
      size_t work = workForStage(stage);
      auto task = taskflow.emplace([&frame, layer, stage, work]() {
        if (stage == 0) {
          auto& buf = frame.layers[layer].stageBuffers[0];
          for (size_t i = 0; i < kBufferSize; i += 16) {
            buf[i] = static_cast<float>(layer * kStagesPerLayer + stage) * 0.001f + 1.0f;
          }
          float acc = 0.0f;
          for (size_t iter = 0; iter < work; ++iter) {
            for (size_t i = 0; i < 64; ++i) {
              acc += buf[i] * 0.99f + 0.01f;
            }
          }
          buf[0] = acc;
        } else {
          processBuffer(
              frame.layers[layer].stageBuffers[stage],
              frame.layers[layer].stageBuffers[stage - 1],
              work);
        }
      });

      if (stage > 0) {
        prevTask.precede(task);
      }
      prevTask = task;
    }
    layerOutputTasks[layer] = prevTask;
  }

  auto mergeTask =
      taskflow.emplace([&frame]() { mergeBuffers(frame.mergeBuffer, frame.layers, kMergeWork); });
  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    layerOutputTasks[layer].precede(mergeTask);
  }

  tf::Task prevPost = mergeTask;
  for (size_t post = 0; post < kPostStages; ++post) {
    auto postTask = taskflow.emplace([&frame, post]() {
      if (post == 0) {
        processBuffer(frame.postBuffers[0], frame.mergeBuffer, kPostWork);
      } else {
        processBuffer(frame.postBuffers[post], frame.postBuffers[post - 1], kPostWork);
      }
    });
    prevPost.precede(postTask);
    prevPost = postTask;
  }
}

} // anonymous namespace

// --- Benchmarks ---

static void BM_compositing_concurrent_task_set(benchmark::State& state) {
  dispenso::ThreadPool& pool = dispenso::globalThreadPool();
  dispenso::Graph g;
  FrameData frame;
  buildCompositingGraph(g, frame);

  dispenso::ConcurrentTaskSet tasks(pool);
  dispenso::ConcurrentTaskSetExecutor executor;

  for (auto _ : state) {
    state.PauseTiming();
    setAllNodesIncomplete(g);
    state.ResumeTiming();
    executor(tasks, g);
    tasks.wait();
  }
}

static void BM_compositing_partial_reeval(benchmark::State& state) {
  dispenso::ThreadPool& pool = dispenso::globalThreadPool();
  dispenso::BiPropGraph g;
  FrameData frame;
  buildCompositingGraph(g, frame);

  dispenso::ConcurrentTaskSet tasks(pool);
  dispenso::ConcurrentTaskSetExecutor executor;
  dispenso::ForwardPropagator propagator;

  // Initial full evaluation
  setAllNodesIncomplete(g);
  executor(tasks, g);
  tasks.wait();

  for (auto _ : state) {
    state.PauseTiming();
    // Dirty 5 layers (10% of 50). Only those 5 layer chains + merge + post
    // need re-evaluation. The other 45 layers are untouched.
    for (size_t i = 0; i < 5; ++i) {
      size_t layerIdx = i * 10; // layers 0, 10, 20, 30, 40
      size_t sourceNodeIdx = layerIdx * kStagesPerLayer;
      g.node(sourceNodeIdx).setIncomplete();
    }
    state.ResumeTiming();

    propagator(g);
    executor(tasks, g);
    tasks.wait();
  }
}

static void BM_compositing_parallel_for(benchmark::State& state) {
  dispenso::ThreadPool& pool = dispenso::globalThreadPool();
  dispenso::Graph g;
  FrameData frame;
  buildCompositingGraph(g, frame);

  dispenso::ParallelForExecutor executor;
  dispenso::TaskSet tasks(pool);

  for (auto _ : state) {
    state.PauseTiming();
    setAllNodesIncomplete(g);
    state.ResumeTiming();
    executor(tasks, g);
  }
}

static void BM_compositing_taskflow(benchmark::State& state) {
  tf::Taskflow taskflow;
  FrameData frame;
  buildCompositingGraphTF(taskflow, frame);

  tf::Executor executor;
  for (auto _ : state) {
    executor.run(taskflow).wait();
  }
}

BENCHMARK(BM_compositing_parallel_for)->UseRealTime();
BENCHMARK(BM_compositing_concurrent_task_set)->UseRealTime();
BENCHMARK(BM_compositing_partial_reeval)->UseRealTime();
BENCHMARK(BM_compositing_taskflow)->UseRealTime();

BENCHMARK_MAIN();
