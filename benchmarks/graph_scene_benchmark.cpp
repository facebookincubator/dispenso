/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>
#include <taskflow/taskflow.hpp>
#if TF_VERSION > 300000
#include <taskflow/algorithm/for_each.hpp>
#endif // TF_VERSION
#include <array>
#include <numeric>
#include <random>
// For this benchmarks we create set of tasks similar to the scene graph.
// Scene consist of the hierarchy of the Transforms which have local space transformation matrix.
// World matrix should be calculated as multiplication of the parent world matrix and local space
// matrix of the transform. Some transforms have two geometry index. inGeoIdex - is index in source
// geometry array (which generated by functions). This geometry should be transformed by worldMatrix
// and stored into outGeo array.
//                                                    generateGeo─┐
//                                                                │
//                                                             4  │
//     transform 2 ◀──────┐   transform 5                ┌─   ─┬──▼─────┬─   ─┐
//    ┌─────────────┐     │  ┌───────────────┐   inGeo:  │ ... │ points │ ... │
//    │ parent      │     └────parent       2│           └─   ─┴──┬─────┴─   ─┘
//    │ matrix      │   ┌──◀───matrix        │                    │
//    │ worldMatrix─────*────▶ worldMatrix   │───────▶────────────* transformGeo
//    │ inGeoIndex  │        │ inGeoIndex   4│                  9 │
//    │ outGeoIndex │        │ outGeoIndex  9│           ┌──  ─┬──▼─────┬─   ─┐
//    └─────────────┘        └───────────────┘   outGeo: │ ... │ points │ ... │
//             calculateWorldMatrix                      └──  ─┴────────┴─   ─┘

namespace {
namespace params {
// parameters of the scene
constexpr size_t numTransforms = 1000; // number transforms in hierarchy
constexpr size_t numInGeo = 100; // number unique geometries
constexpr size_t numVertexMultiplier = 1024;
constexpr size_t everyNthHasGeo = 2; // probability that transform has geo is 1/everyNthHasGeo
} // namespace params

using Vec3 = std::array<float, 3>;
using Matrix4 = std::array<float, 16>;

constexpr size_t kNoGeometry = std::numeric_limits<size_t>::max();
constexpr size_t kRoot = std::numeric_limits<size_t>::max();
struct Transform {
  Matrix4 matrix;
  Matrix4 worldMatrix;
  size_t parent = kRoot;
  size_t inGeoIndex = kNoGeometry;
  size_t outGeoIndex = kNoGeometry;
};

using Geometry = std::vector<Vec3>;

struct Scene {
  std::vector<Transform> transforms;
  std::vector<Geometry> inGeo;
  std::vector<Geometry> outGeo;
};
//--------------------------------------------------------------------------------
void branchlessONB(const Vec3& n, Vec3& b1, Vec3& b2) {
  const float sign = std::copysign(1.f, n[2]);
  const float a = -1.0f / (sign + n[2]);
  const float b = n[0] * n[1] * a;
  b1 = Vec3{1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]};
  b2 = Vec3{b, sign + n[1] * n[1] * a, -n[1]};
}

Matrix4 getRandomTransformMatrix(std::mt19937& rng) {
  std::uniform_real_distribution<float> thetaDistr(0.f, 2.f * static_cast<float>(M_PI));
  std::uniform_real_distribution<float> uDistr(-1.f, 1.f);
  const float theta = thetaDistr(rng);
  const float u = uDistr(rng);
  const float sq = std::sqrt(1 - u * u);
  const Vec3 dirY = {sq * std::cos(theta), sq * std::sin(theta), u};
  Vec3 dirX, dirZ;
  branchlessONB(dirY, dirZ, dirX);
  const Vec3 pos{uDistr(rng), uDistr(rng), uDistr(rng)};
  // clang-format off
  return {dirX[0], dirX[1], dirX[2], 0.f,
          dirY[0], dirY[1], dirY[2], 0.f,
          dirZ[0], dirZ[1], dirZ[2], 0.f,
          pos[0],  pos[1],  pos[2],  1.f };
  // clang-format on
}

Scene generateTransformsHierarchy(
    size_t numTransforms,
    size_t numInGeo,
    size_t everyNthHasGeo,
    std::mt19937& rng) {
  std::uniform_int_distribution<size_t> distGeom(0, numInGeo - 1);
  std::uniform_int_distribution<size_t> distProb(0, everyNthHasGeo - 1);

  Scene scene;
  scene.inGeo.resize(numInGeo);
  std::vector<Transform>& transforms = scene.transforms;
  transforms.resize(numTransforms);
  transforms[0].matrix = getRandomTransformMatrix(rng);
  transforms[0].worldMatrix = transforms[0].matrix;
  size_t outGeoCounter = 0;
  for (size_t i = 1; i < numTransforms; ++i) {
    Transform& transform = transforms[i];
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    transform.parent = dist(rng);
    transform.matrix = getRandomTransformMatrix(rng);
    if (distProb(rng) == 0) {
      transform.inGeoIndex = distGeom(rng);
      transform.outGeoIndex = outGeoCounter++;
    }
  }
  scene.outGeo.resize(outGeoCounter);
  return scene;
}
//--------------------------------compute functions-------------------------------
Vec3 multiply(const Vec3& v, const Matrix4& m) {
  const float invertW = 1.f / (m[11] * v[2] + m[7] * v[1] + m[3] * v[0] + m[15]);
  return {
      (m[8] * v[2] + m[4] * v[1] + m[0] * v[0] + m[12]) * invertW,
      (m[9] * v[2] + m[5] * v[1] + m[1] * v[0] + m[13]) * invertW,
      (m[10] * v[2] + m[6] * v[1] + m[2] * v[0] + m[14]) * invertW};
}

Matrix4 multiply(const Matrix4& ma, const Matrix4& mb) {
  return {
      ma[2] * mb[8] + ma[1] * mb[4] + ma[3] * mb[12] + ma[0] * mb[0],
      ma[2] * mb[9] + ma[1] * mb[5] + ma[3] * mb[13] + ma[0] * mb[1],
      ma[1] * mb[6] + ma[0] * mb[2] + ma[3] * mb[14] + ma[2] * mb[10],
      ma[1] * mb[7] + ma[0] * mb[3] + ma[3] * mb[15] + ma[2] * mb[11],
      ma[6] * mb[8] + ma[5] * mb[4] + ma[7] * mb[12] + ma[4] * mb[0],
      ma[6] * mb[9] + ma[5] * mb[5] + ma[7] * mb[13] + ma[4] * mb[1],
      ma[5] * mb[6] + ma[4] * mb[2] + ma[7] * mb[14] + ma[6] * mb[10],
      ma[5] * mb[7] + ma[4] * mb[3] + ma[7] * mb[15] + ma[6] * mb[11],
      ma[10] * mb[8] + ma[9] * mb[4] + ma[11] * mb[12] + ma[8] * mb[0],
      ma[10] * mb[9] + ma[9] * mb[5] + ma[11] * mb[13] + ma[8] * mb[1],
      ma[9] * mb[6] + ma[8] * mb[2] + ma[11] * mb[14] + ma[10] * mb[10],
      ma[9] * mb[7] + ma[8] * mb[3] + ma[11] * mb[15] + ma[10] * mb[11],
      ma[14] * mb[8] + ma[13] * mb[4] + ma[15] * mb[12] + ma[12] * mb[0],
      ma[14] * mb[9] + ma[13] * mb[5] + ma[15] * mb[13] + ma[12] * mb[1],
      ma[13] * mb[6] + ma[12] * mb[2] + ma[15] * mb[14] + ma[14] * mb[10],
      ma[13] * mb[7] + ma[12] * mb[3] + ma[15] * mb[15] + ma[14] * mb[11]};
}

void calculateWorldMatrix(std::vector<Transform>& transforms, size_t index) {
  Transform& transform = transforms[index];
  if (transform.parent == kRoot) {
    transform.worldMatrix = transform.matrix;
  } else {
    transform.worldMatrix = multiply(transform.matrix, transforms[transform.parent].worldMatrix);
  }
}

size_t numGeoPoints(size_t inGeoIndex) {
  return (inGeoIndex + 1) * params::numVertexMultiplier;
}

Vec3 calculateGeoPoint(size_t inGeoIndex, size_t pointIndex) {
  size_t numPoints = numGeoPoints(inGeoIndex);
  const float tStep = 2.f * 2.f * static_cast<float>(M_PI) / (static_cast<float>(numPoints) - 1.f);
  const float t = tStep * static_cast<float>(pointIndex);

  const float r = 0.01;
  return {r * t * std::cos(t), r * t * std::sin(t), r * (t + std::sin(16 * t))};
}
//--------------------------------------------------------------------------------
//                                    taskflow
//--------------------------------------------------------------------------------
tf::Task generateGeoTF(tf::Taskflow& taskflow, std::vector<Geometry>& inGeo, size_t inGeoIndex) {
  size_t numPoints = numGeoPoints(inGeoIndex);
  Geometry& g = inGeo[inGeoIndex];
  g.resize(numPoints);

  return taskflow.for_each_index(
      size_t(0), numPoints, size_t(1), [&](size_t i) { g[i] = calculateGeoPoint(inGeoIndex, i); });
}

tf::Task
transformGeoTF(tf::Taskflow& taskflow, Geometry& g, const Geometry& inG, const Matrix4& m) {
  const size_t numPoints = inG.size();

  g.resize(numPoints);

  return taskflow.for_each_index(
      size_t(0), numPoints, size_t(1), [&](size_t i) { g[i] = multiply(inG[i], m); });
}

void prepareGraphTF(tf::Taskflow& taskflow, Scene& scene) {
  std::mt19937 rng(12345);

  scene = generateTransformsHierarchy(
      params::numTransforms, params::numInGeo, params::everyNthHasGeo, rng);

  std::vector<Geometry>& inGeo = scene.inGeo;
  std::vector<Transform>& transforms = scene.transforms;
  std::vector<Geometry>& outGeo = scene.outGeo;
  // calculate inGeo
  std::vector<tf::Task> inGeoTasks(params::numInGeo);

  for (size_t i = 0; i < params::numInGeo; ++i) {
    inGeoTasks[i] = generateGeoTF(taskflow, inGeo, i);
  }

  // calculate transforms
  std::vector<tf::Task> transformTasks(params::numTransforms);
  std::vector<tf::Task> outGeoTasks(outGeo.size());
  for (size_t i = 0; i < params::numTransforms; ++i) {
    transformTasks[i] =
        taskflow.emplace([&transforms, i]() { calculateWorldMatrix(transforms, i); });
    const size_t parentIndex = transforms[i].parent;
    if (parentIndex != kRoot) {
      transformTasks[i].succeed(transformTasks[parentIndex]);
    }

    // calculate outGeo
    const size_t outGeoIndex = transforms[i].outGeoIndex;
    const size_t inGeoIndex = transforms[i].inGeoIndex;
    if (inGeoIndex != kNoGeometry) {
      outGeoTasks[outGeoIndex] = transformGeoTF(
          taskflow, outGeo[outGeoIndex], inGeo[inGeoIndex], transforms[i].worldMatrix);

      outGeoTasks[outGeoIndex].succeed(transformTasks[i], inGeoTasks[inGeoIndex]);
    }
  }
}
//--------------------------------------------------------------------------------
//                                    dispenso
//--------------------------------------------------------------------------------
void generateGeo(
    dispenso::ThreadPool& threadPool,
    std::vector<Geometry>& inGeo,
    size_t inGeoIndex) {
  size_t numPoints = numGeoPoints(inGeoIndex);
  Geometry& g = inGeo[inGeoIndex];
  g.resize(numPoints);

  dispenso::TaskSet taskSet(threadPool);

  dispenso::ParForOptions options;
  options.maxThreads = static_cast<uint32_t>(numPoints);
  options.minItemsPerChunk = 256;

  dispenso::parallel_for(
      taskSet, 0, numPoints, [&](size_t i) { g[i] = calculateGeoPoint(inGeoIndex, i); }, options);
}

void transformGeo(
    dispenso::ThreadPool& threadPool,
    Geometry& g,
    const Geometry& inG,
    const Matrix4& m) {
  dispenso::TaskSet taskSet(threadPool);
  const size_t numPoints = inG.size();
  dispenso::ParForOptions options;
  options.maxThreads = static_cast<uint32_t>(numPoints);
  options.minItemsPerChunk = 256;

  g.resize(numPoints);

  dispenso::parallel_for(
      taskSet, 0, numPoints, [&](size_t i) { g[i] = multiply(inG[i], m); }, options);
}

struct Subgraphs {
  dispenso::Subgraph* inGeo;
  dispenso::Subgraph* transforms;
  dispenso::Subgraph* outGeo;
};
Subgraphs prepareGraph(dispenso::ThreadPool& threadPool, Scene& scene, dispenso::Graph& g) {
  std::mt19937 rng(12345);

  scene = generateTransformsHierarchy(
      params::numTransforms, params::numInGeo, params::everyNthHasGeo, rng);

  std::vector<Geometry>& inGeo = scene.inGeo;
  std::vector<Transform>& transforms = scene.transforms;
  std::vector<Geometry>& outGeo = scene.outGeo;
  // calculate inGeo
  dispenso::Subgraph& transformsSub = g.addSubgraph();
  dispenso::Subgraph& inGeoSub = g.addSubgraph();
  dispenso::Subgraph& outGeoSub = g.addSubgraph();

  for (size_t i = 0; i < params::numInGeo; ++i) {
    inGeoSub.addNode([&threadPool, &inGeo, i]() { generateGeo(threadPool, inGeo, i); });
  }

  // calculate transforms
  for (size_t i = 0; i < params::numTransforms; ++i) {
    transformsSub.addNode([&transforms, i]() { calculateWorldMatrix(transforms, i); });
    const size_t parentIndex = transforms[i].parent;
    if (parentIndex != kRoot) {
      transformsSub.node(i).dependsOn(transformsSub.node(parentIndex));
    }

    // calculate outGeo
    const size_t outGeoIndex = transforms[i].outGeoIndex;
    const size_t inGeoIndex = transforms[i].inGeoIndex;
    if (inGeoIndex != kNoGeometry) {
      outGeoSub.addNode([&, i, inGeoIndex, outGeoIndex]() {
        transformGeo(threadPool, outGeo[outGeoIndex], inGeo[inGeoIndex], transforms[i].worldMatrix);
      });
      outGeoSub.node(outGeoIndex).dependsOn(transformsSub.node(i), inGeoSub.node(inGeoIndex));
    }
  }

  return {&inGeoSub, &transformsSub, &outGeoSub};
}

//----------------------------------test results----------------------------------
#ifndef NDEBUG
template <size_t N>
bool compare(const std::array<float, N>& ma, const std::array<float, N>& mb) {
  bool result = true;
  for (size_t i = 0; i < N; ++i) {
    result = result && std::abs(ma[i] - mb[i]) < std::numeric_limits<float>::epsilon();
  }
  return result;
}

bool testScene(const Scene& scene) {
  bool result = true;
  for (const Transform& transform : scene.transforms) {
    const Matrix4 worldMatrix = transform.parent == kRoot
        ? transform.matrix
        : multiply(transform.matrix, scene.transforms[transform.parent].worldMatrix);
    result = result && compare(transform.worldMatrix, worldMatrix);
  }

  for (const Transform& transform : scene.transforms) {
    const size_t outGeoIndex = transform.outGeoIndex;
    const size_t inGeoIndex = transform.inGeoIndex;
    if (inGeoIndex != kNoGeometry) {
      const size_t numVertices = scene.inGeo[inGeoIndex].size();
      assert(numVertices == scene.outGeo[outGeoIndex].size());
      for (size_t i = 0; i < numVertices; ++i) {
        const Vec3 worldPos = multiply(scene.inGeo[inGeoIndex][i], transform.worldMatrix);
        result = result && compare(scene.outGeo[outGeoIndex][i], worldPos);
      }
    }
  }
  return result;
}
void cleanScene(Scene& s) {
  for (Geometry& g : s.inGeo) {
    g.clear();
  }
  for (Geometry& g : s.outGeo) {
    g.clear();
  }
  for (Transform& t : s.transforms) {
    std::fill(t.worldMatrix.begin(), t.worldMatrix.end(), 0.f);
  }
}

#endif
} // anonymous namespace

static void BM_scene_graph_parallel_for(benchmark::State& state) {
  // transform inGeo
  dispenso::ThreadPool& threadPool = dispenso::globalThreadPool();

  dispenso::Graph g;
  Scene scene;
  prepareGraph(threadPool, scene, g);

  dispenso::ParallelForExecutor parallelForExecutor;

  dispenso::TaskSet taskSet(threadPool);

  for (auto _ : state) {
    state.PauseTiming();
    setAllNodesIncomplete(g);
    state.ResumeTiming();

    parallelForExecutor(taskSet, g);

#ifndef NDEBUG
    state.PauseTiming();
    assert(testScene(scene));
    cleanScene(scene);
    state.ResumeTiming();
#endif
  }
}

static void BM_scene_graph_concurrent_task_set(benchmark::State& state) {
  // transform inGeo
  dispenso::ThreadPool& threadPool = dispenso::globalThreadPool();

  dispenso::Graph g;
  Scene scene;
  prepareGraph(threadPool, scene, g);

  dispenso::ConcurrentTaskSet concurrentTaskSet(threadPool);
  dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;

  for (auto _ : state) {
    state.PauseTiming();
    setAllNodesIncomplete(g);
    state.ResumeTiming();

    concurrentTaskSetExecutor(concurrentTaskSet, g);
    concurrentTaskSet.wait();

#ifndef NDEBUG
    state.PauseTiming();
    assert(testScene(scene));
    cleanScene(scene);
    state.ResumeTiming();
#endif
  }
}

static void BM_scene_graph_partial_revaluation(benchmark::State& state) {
  dispenso::ThreadPool& threadPool = dispenso::globalThreadPool();

  std::mt19937 rng(123456);

  dispenso::Graph g;
  Scene scene;
  Subgraphs subgraphs = prepareGraph(threadPool, scene, g);

  const size_t numTransforms = scene.transforms.size();
  std::uniform_real_distribution<float> transformIndexDistr(0, numTransforms - 1);

  dispenso::ConcurrentTaskSet concurrentTaskSet(threadPool);
  dispenso::ConcurrentTaskSetExecutor concurrentTaskSetExecutor;
  dispenso::ForwardPropagator forwardPropagator;

  setAllNodesIncomplete(g);
  concurrentTaskSetExecutor(concurrentTaskSet, g);
  concurrentTaskSet.wait();

  assert(testScene(scene));

  for (auto _ : state) {
    state.PauseTiming();
    // change several transforms
    rng.seed(123456);
    for (size_t i = 0; i < 10; ++i) {
      const size_t index = transformIndexDistr(rng);
      Transform& t = scene.transforms[index];
      t.matrix = getRandomTransformMatrix(rng);
      subgraphs.transforms->node(index).setIncomplete();
    }
    state.ResumeTiming();
    // The graph automatically recalculates all children of modified transforms. If these transforms
    // have geometry this geometry will be recomputed as well.
    forwardPropagator(g);
    concurrentTaskSetExecutor(concurrentTaskSet, g);
    concurrentTaskSet.wait();
#ifndef NDEBUG
    state.PauseTiming();
    assert(testScene(scene));
    state.ResumeTiming();
#endif
  }
}
// TODO(roman fedotov): Add partial evaluation variant for taskflow (possible implementation:
// conditional tasks)

static void BM_scene_graph_taskflow(benchmark::State& state) {
  tf::Taskflow taskflow;

  Scene scene;
  prepareGraphTF(taskflow, scene);

  tf::Executor executor;
  for (auto _ : state) {
    executor.run(taskflow).wait();

#ifndef NDEBUG
    state.PauseTiming();
    assert(testScene(scene));
    cleanScene(scene);
    state.ResumeTiming();
#endif
  }
}

BENCHMARK(BM_scene_graph_parallel_for)->UseRealTime();
BENCHMARK(BM_scene_graph_concurrent_task_set)->UseRealTime();
BENCHMARK(BM_scene_graph_taskflow)->UseRealTime();

#ifndef NDEBUG
BENCHMARK(BM_scene_graph_partial_revaluation)->UseRealTime()->Iterations(50);
#else
BENCHMARK(BM_scene_graph_partial_revaluation)->UseRealTime();
#endif

BENCHMARK_MAIN();
