/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include <dispenso/future.h>
#include <dispenso/parallel_for.h>
#include <dispenso/parallel_invoke.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include <tbb/parallel_invoke.h>
#include "tbb_compat.h"
#endif // !BENCHMARK_WITHOUT_TBB

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

#include "thread_benchmark_common.h"

constexpr size_t kSmallSize = 13;
constexpr size_t kMediumSize = 16;
constexpr size_t kLargeSize = 19;

// Note that there are many optimizations that could be made for these tree build routines.  The
// goal was to make these as apples-to-apples as possible.

uint32_t busyWork(uint32_t seed, size_t iterations) {
  uint32_t h = seed;
  for (size_t i = 0; i < iterations; ++i) {
    h *= 2654435761u;
    h ^= h >> 16;
  }
  return h;
}

struct Node {
  Node* left;
  Node* right;
  uint32_t value;
  uint32_t workResult;

  void setValue(uint32_t unique_bitset, uint32_t modulo) {
    value = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      value += unique_bitset % modulo;
      unique_bitset /= modulo;
    }
  }

  void setValueWithWork(uint32_t unique_bitset, uint32_t modulo, size_t workIters) {
    setValue(unique_bitset, modulo);
    workResult = busyWork(unique_bitset, workIters);
  }
};

class Allocator {
 public:
  void reset(size_t depth) {
    nodes_.resize(std::pow(2, depth) - 1);
    next_.store(0, std::memory_order_release);
  }

  Node* alloc() {
    size_t cur = next_.fetch_add(1, std::memory_order_relaxed);
    return &nodes_[cur];
  }

 private:
  std::vector<Node> nodes_;
  std::atomic<size_t> next_{0};
};

const std::vector<uint32_t>& getModulos() {
  static const std::vector<uint32_t> modulos = []() {
    std::mt19937 mt;
    std::uniform_int_distribution<> dis(2, 55);
    std::vector<uint32_t> m;
    for (size_t i = 0; i < 64; ++i) {
      m.emplace_back(dis(mt));
    }
    return m;
  }();
  return modulos;
}

uint64_t sumTree(Node* root) {
  if (!root) {
    return 0;
  }
  return root->value + sumTree(root->left) + sumTree(root->right);
}

void checkTree(Node* root, uint32_t depth, uint32_t modulo) {
  uint64_t expectedSum = 0;

  uint32_t num = std::pow(2, depth);
  for (uint32_t i = 0; i < num; ++i) {
    auto bitset = i;
    while (bitset) {
      expectedSum += bitset % modulo;
      bitset /= modulo;
    }
  }

  uint64_t actual = sumTree(root);
  if (actual != expectedSum) {
    std::cerr << "Mismatch! " << expectedSum << " vs " << actual << std::endl;
    std::abort();
  }
}

Node* serialTree(Allocator& allocator, uint32_t depth, uint32_t bitset, uint32_t modulo) {
  --depth;
  Node* node = allocator.alloc();
  node->setValue(bitset, modulo);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  node->left = serialTree(allocator, depth, (bitset << 1), modulo);
  node->right = serialTree(allocator, depth, (bitset << 1) | 1, modulo);

  return node;
}

template <size_t depth>
void BM_serial_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;

  Node* root;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = serialTree(alloc, depth, 1, modulo);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(root, depth, modulo);
}

Node* dispensoTree(Allocator& allocator, uint32_t depth, uint32_t bitset, uint32_t modulo) {
  --depth;
  Node* node = allocator.alloc();
  node->setValue(bitset, modulo);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }

  auto left = dispenso::async(
      [=, &allocator]() { return dispensoTree(allocator, depth, (bitset << 1), modulo); });
  auto right = dispenso::async(
      [=, &allocator]() { return dispensoTree(allocator, depth, (bitset << 1) | 1, modulo); });
  node->left = left.get();
  node->right = right.get();

  return node;
}

// Naive baseline: forks both children with dispenso::async and blocks on each
// child's .get() at every recursion level, which starves the pool (real_time
// dwarfs cpu_time). Kept only as a cautionary "how not to do it" reference --
// not a representative dispenso fork-join result. See the taskset/parallel_invoke
// variants for the real idioms.
template <size_t depth>
void BM_dispenso_tree_naive(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();

  uint32_t modulo;

  Node* root;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = dispensoTree(alloc, depth, 1, modulo);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(root, depth, modulo);
}

#if !defined(BENCHMARK_WITHOUT_FOLLY)
folly::SemiFuture<folly::Unit> follyTree(
    folly::Executor* exec,
    Node* node,
    Allocator* allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo) {
  --depth;
  node->setValue(bitset, modulo);

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return folly::Unit{};
  }

  node->left = allocator->alloc();
  node->right = allocator->alloc();

  return folly::via(
             exec,
             [=]() {
               return folly::collectAll(
                          follyTree(exec, node->left, allocator, depth, bitset << 1, modulo),
                          follyTree(exec, node->right, allocator, depth, bitset << 1 | 1, modulo))
                   .unit();
             })
      .semi();
}

template <size_t depth>
void BM_folly_tree(benchmark::State& state) {
  folly::CPUThreadPoolExecutor follyExec{std::thread::hardware_concurrency()};
  Allocator alloc;
  alloc.reset(depth);

  uint32_t modulo;

  Node root;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    follyTree(&follyExec, &root, &alloc, depth, 1, modulo).via(&follyExec).get();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(&root, depth, modulo);
}
#endif // !BENCHMARK_WITHOUT_FOLLY

void dispensoTaskSetTree(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo) {
  node->setValue(bitset, modulo);
  --depth;

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }

  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo]() {
        node->left = allocator.alloc();
        dispensoTaskSetTree(tasks, node->left, allocator, depth, (bitset << 1), modulo);
      },
      true);
  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo]() {
        node->right = allocator.alloc();
        dispensoTaskSetTree(tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo);
      },
      true);
}

template <size_t depth>
void BM_dispenso_taskset_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2);

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTree(tasks, &root, alloc, depth, 1, modulo);
    tasks.wait();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeBulk(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo) {
  node->setValue(bitset, modulo);
  --depth;

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }

  std::array<Node**, 2> children = {&node->left, &node->right};
  tasks.scheduleBulk(2, [&tasks, &allocator, children, depth, bitset, modulo](size_t i) {
    return [&tasks,
            &allocator,
            child = children[i],
            depth,
            bitset = (bitset << 1) | static_cast<uint32_t>(i),
            modulo]() {
      *child = allocator.alloc();
      dispensoTaskSetTreeBulk(tasks, *child, allocator, depth, bitset, modulo);
    };
  });
}

template <size_t depth>
void BM_dispenso_taskset_tree_bulk(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2);

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeBulk(tasks, &root, alloc, depth, 1, modulo);
    tasks.wait();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeHybrid(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    uint32_t serialThreshold) {
  node->setValue(bitset, modulo);
  --depth;

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }

  if (depth <= serialThreshold) {
    node->left = serialTree(allocator, depth, (bitset << 1), modulo);
    node->right = serialTree(allocator, depth, (bitset << 1) | 1, modulo);
    return;
  }

  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo, serialThreshold]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeHybrid(
            tasks, node->left, allocator, depth, (bitset << 1), modulo, serialThreshold);
      },
      true);
  node->right = allocator.alloc();
  dispensoTaskSetTreeHybrid(
      tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo, serialThreshold);
}

template <size_t depth>
void BM_dispenso_taskset_tree_hybrid(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  auto& pool = dispenso::globalThreadPool();
  dispenso::ConcurrentTaskSet tasks(pool, dispenso::ParentCascadeCancel::kOff, 2);

  uint32_t serialThreshold = (depth > 10) ? depth - 10 : 0;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeHybrid(tasks, &root, alloc, depth, 1, modulo, serialThreshold);
    tasks.wait();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeInline(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo) {
  node->setValue(bitset, modulo);
  --depth;

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }

  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeInline(tasks, node->left, allocator, depth, (bitset << 1), modulo);
      },
      true);
  node->right = allocator.alloc();
  dispensoTaskSetTreeInline(tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo);
}

template <size_t depth>
void BM_dispenso_taskset_tree_inline(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2);

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeInline(tasks, &root, alloc, depth, 1, modulo);
    tasks.wait();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

dispenso::Future<Node*>
dispensoTreeWhenAll(Allocator& allocator, uint32_t depth, uint32_t bitset, uint32_t modulo) {
  --depth;
  Node* node = allocator.alloc();
  node->setValue(bitset, modulo);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return dispenso::make_ready_future(node);
  }

  auto left = dispenso::async([depth, bitset, modulo, &allocator]() {
    return dispensoTreeWhenAll(allocator, depth, (bitset << 1), modulo);
  });
  auto right = dispenso::async([depth, bitset, modulo, &allocator]() {
    return dispensoTreeWhenAll(allocator, depth, (bitset << 1) | 1, modulo);
  });
  return dispenso::when_all(left, right).then([node](auto&& both) {
    auto& tuple = both.get();
    node->left = std::get<0>(tuple).get().get();
    node->right = std::get<1>(tuple).get().get();
    return node;
  });
}

template <size_t depth>
void BM_dispenso_tree_when_all(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();

  uint32_t modulo;

  Node* root;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = dispensoTreeWhenAll(alloc, depth, 1, modulo).get();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(root, depth, modulo);
}

#if !defined(BENCHMARK_WITHOUT_TBB)
Node* tbbTree(Allocator& allocator, uint32_t depth, uint32_t bitset, uint32_t modulo) {
  --depth;
  Node* node = allocator.alloc();
  node->setValue(bitset, modulo);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  tbb::parallel_invoke(
      [&]() { node->left = tbbTree(allocator, depth, (bitset << 1), modulo); },
      [&]() { node->right = tbbTree(allocator, depth, (bitset << 1) | 1, modulo); });
  return node;
}

template <size_t depth>
void BM_tbb_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node* root;
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = tbbTree(alloc, depth, 1, modulo);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(root, depth, modulo);
}

Node* tbbTreeWork(
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  --depth;
  Node* node = allocator.alloc();
  node->setValueWithWork(bitset, modulo, workIters);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  tbb::parallel_invoke(
      [&]() { node->left = tbbTreeWork(allocator, depth, (bitset << 1), modulo, workIters); },
      [&]() { node->right = tbbTreeWork(allocator, depth, (bitset << 1) | 1, modulo, workIters); });
  return node;
}

template <size_t depth, size_t workIters>
void BM_tbb_tree_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node* root;
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = tbbTreeWork(alloc, depth, 1, modulo, workIters);
    benchmark::DoNotOptimize(root->workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(root, depth, modulo);
}
#endif // !BENCHMARK_WITHOUT_TBB

// ============================================================================
// Graduated-work tree benchmarks
//
// Same fork-join tree structure, but with tunable per-node work. This reveals
// whether scheduling overhead regressions are visible when nodes do real work.
// ============================================================================

constexpr size_t kLightWork = 100;
constexpr size_t kMediumWork = 1000;
constexpr size_t kHeavyWork = 10000;

// Pick the dispenso TaskCost hint that best matches a given per-task work
// budget.  Heavy workloads benefit from locality-aware scheduling; light/medium
// workloads are dominated by submission overhead and prefer the cheaper path.
constexpr dispenso::TaskCost taskCostFor(size_t workIters) {
  return workIters >= kHeavyWork ? dispenso::TaskCost::kHeavy : dispenso::TaskCost::kLightweight;
}

Node* serialTreeWork(
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  --depth;
  Node* node = allocator.alloc();
  node->setValueWithWork(bitset, modulo, workIters);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  node->left = serialTreeWork(allocator, depth, (bitset << 1), modulo, workIters);
  node->right = serialTreeWork(allocator, depth, (bitset << 1) | 1, modulo, workIters);
  return node;
}

template <size_t depth, size_t workIters>
void BM_serial_tree_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node* root;
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = serialTreeWork(alloc, depth, 1, modulo, workIters);
    benchmark::DoNotOptimize(root->workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(root, depth, modulo);
}

Node* dispensoTreeWork(
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  --depth;
  Node* node = allocator.alloc();
  node->setValueWithWork(bitset, modulo, workIters);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  auto left = dispenso::async([=, &allocator]() {
    return dispensoTreeWork(allocator, depth, (bitset << 1), modulo, workIters);
  });
  auto right = dispenso::async([=, &allocator]() {
    return dispensoTreeWork(allocator, depth, (bitset << 1) | 1, modulo, workIters);
  });
  node->left = left.get();
  node->right = right.get();
  return node;
}

template <size_t depth, size_t workIters>
void BM_dispenso_tree_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();
  uint32_t modulo;
  Node* root;
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = dispensoTreeWork(alloc, depth, 1, modulo, workIters);
    benchmark::DoNotOptimize(root->workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(root, depth, modulo);
}

void dispensoTaskSetTreeWork(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo, workIters]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeWork(
            tasks, node->left, allocator, depth, (bitset << 1), modulo, workIters);
      },
      true);
  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo, workIters]() {
        node->right = allocator.alloc();
        dispensoTaskSetTreeWork(
            tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo, workIters);
      },
      true);
}

template <size_t depth, size_t workIters>
void BM_dispenso_taskset_tree_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node root;
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2, taskCostFor(workIters));
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeWork(tasks, &root, alloc, depth, 1, modulo, workIters);
    tasks.wait();
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeBulkWork(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  std::array<Node**, 2> children = {&node->left, &node->right};
  tasks.scheduleBulk(2, [&tasks, &allocator, children, depth, bitset, modulo, workIters](size_t i) {
    return [&tasks,
            &allocator,
            child = children[i],
            depth,
            bitset = (bitset << 1) | static_cast<uint32_t>(i),
            modulo,
            workIters]() {
      *child = allocator.alloc();
      dispensoTaskSetTreeBulkWork(tasks, *child, allocator, depth, bitset, modulo, workIters);
    };
  });
}

template <size_t depth, size_t workIters>
void BM_dispenso_taskset_tree_bulk_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node root;
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2, taskCostFor(workIters));
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeBulkWork(tasks, &root, alloc, depth, 1, modulo, workIters);
    tasks.wait();
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(&root, depth, modulo);
}

// dispenso::parallel_invoke (binary) — fork-join idiom: schedule one sibling,
// run the other inline on the calling thread.
void dispensoTaskSetTreeParInvokeWork(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  dispenso::parallel_invoke(
      tasks,
      [&tasks, &allocator, node, depth, bitset, modulo, workIters]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeParInvokeWork(
            tasks, node->left, allocator, depth, (bitset << 1), modulo, workIters);
      },
      [&tasks, &allocator, node, depth, bitset, modulo, workIters]() {
        node->right = allocator.alloc();
        dispensoTaskSetTreeParInvokeWork(
            tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo, workIters);
      });
}

template <size_t depth, size_t workIters>
void BM_dispenso_taskset_tree_parallel_invoke_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node root;
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2, taskCostFor(workIters));
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeParInvokeWork(tasks, &root, alloc, depth, 1, modulo, workIters);
    tasks.wait();
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeHybridWork(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    uint32_t serialThreshold,
    size_t workIters) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;

  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }

  if (depth <= serialThreshold) {
    node->left = serialTreeWork(allocator, depth, (bitset << 1), modulo, workIters);
    node->right = serialTreeWork(allocator, depth, (bitset << 1) | 1, modulo, workIters);
    return;
  }

  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo, serialThreshold, workIters]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeHybridWork(
            tasks, node->left, allocator, depth, (bitset << 1), modulo, serialThreshold, workIters);
      },
      true);
  node->right = allocator.alloc();
  dispensoTaskSetTreeHybridWork(
      tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo, serialThreshold, workIters);
}

template <size_t depth, size_t workIters>
void BM_dispenso_taskset_tree_hybrid_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node root;

  auto& pool = dispenso::globalThreadPool();
  dispenso::ConcurrentTaskSet tasks(
      pool, dispenso::ParentCascadeCancel::kOff, 2, taskCostFor(workIters));

  uint32_t serialThreshold = (depth > 10) ? depth - 10 : 0;

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeHybridWork(
        tasks, &root, alloc, depth, 1, modulo, serialThreshold, workIters);
    tasks.wait();
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeInlineWork(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  tasks.schedule(
      [&tasks, &allocator, node, depth, bitset, modulo, workIters]() {
        node->left = allocator.alloc();
        dispensoTaskSetTreeInlineWork(
            tasks, node->left, allocator, depth, (bitset << 1), modulo, workIters);
      },
      true);
  node->right = allocator.alloc();
  dispensoTaskSetTreeInlineWork(
      tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo, workIters);
}

template <size_t depth, size_t workIters>
void BM_dispenso_taskset_tree_inline_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  uint32_t modulo;
  Node root;
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(), dispenso::ParentCascadeCancel::kOff, 2, taskCostFor(workIters));
  size_t m = 0;
  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeInlineWork(tasks, &root, alloc, depth, 1, modulo, workIters);
    tasks.wait();
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }
  checkTree(&root, depth, modulo);
}

// ============================================================================
// Parallel-for tree experiment
//
// Serially build the top levels of the tree, collecting frontier nodes into
// a vector.  Then dispenso::parallel_for over those entries, calling
// serialTree (or serialTreeWork) to complete each subtree.  This is the
// optimal scheduling pattern for this workload and establishes a lower bound.
// ============================================================================

struct ParForLeaf {
  Node** dest;
  uint32_t depth;
  uint32_t bitset;
  uint32_t modulo;
};

void collectTreeLeaves(
    Allocator& allocator,
    Node* node,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    uint32_t parallelDepth,
    std::vector<ParForLeaf>& leaves) {
  node->setValue(bitset, modulo);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  if (depth <= parallelDepth) {
    leaves.push_back({&node->left, depth, (bitset << 1), modulo});
    leaves.push_back({&node->right, depth, (bitset << 1) | 1, modulo});
    return;
  }
  node->left = allocator.alloc();
  node->right = allocator.alloc();
  collectTreeLeaves(allocator, node->left, depth, (bitset << 1), modulo, parallelDepth, leaves);
  collectTreeLeaves(
      allocator, node->right, depth, (bitset << 1) | 1, modulo, parallelDepth, leaves);
}

void collectTreeLeavesWork(
    Allocator& allocator,
    Node* node,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo,
    size_t workIters,
    uint32_t parallelDepth,
    std::vector<ParForLeaf>& leaves) {
  node->setValueWithWork(bitset, modulo, workIters);
  --depth;
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return;
  }
  if (depth <= parallelDepth) {
    leaves.push_back({&node->left, depth, (bitset << 1), modulo});
    leaves.push_back({&node->right, depth, (bitset << 1) | 1, modulo});
    return;
  }
  node->left = allocator.alloc();
  node->right = allocator.alloc();
  collectTreeLeavesWork(
      allocator, node->left, depth, (bitset << 1), modulo, workIters, parallelDepth, leaves);
  collectTreeLeavesWork(
      allocator, node->right, depth, (bitset << 1) | 1, modulo, workIters, parallelDepth, leaves);
}

template <size_t depth>
void BM_dispenso_parfor_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();

  uint32_t modulo;
  Node root;

  constexpr uint32_t kSerialLevels = 9;
  constexpr uint32_t parallelDepth = (depth > kSerialLevels) ? depth - kSerialLevels : 0;

  std::vector<ParForLeaf> leaves;
  leaves.reserve(size_t{1} << kSerialLevels);

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    leaves.clear();
    collectTreeLeaves(alloc, &root, depth, 1, modulo, parallelDepth, leaves);
    dispenso::parallel_for(size_t{0}, leaves.size(), [&alloc, &leaves](size_t i) {
      auto& task = leaves[i];
      *task.dest = serialTree(alloc, task.depth, task.bitset, task.modulo);
    });
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

template <size_t depth, size_t workIters>
void BM_dispenso_parfor_tree_work(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();

  uint32_t modulo;
  Node root;

  constexpr uint32_t kSerialLevels = 9;
  constexpr uint32_t parallelDepth = (depth > kSerialLevels) ? depth - kSerialLevels : 0;

  std::vector<ParForLeaf> leaves;
  leaves.reserve(size_t{1} << kSerialLevels);

  size_t m = 0;

  for (auto UNUSED_VAR : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    leaves.clear();
    collectTreeLeavesWork(alloc, &root, depth, 1, modulo, workIters, parallelDepth, leaves);
    dispenso::parallel_for(size_t{0}, leaves.size(), [&alloc, &leaves, wi = workIters](size_t i) {
      auto& task = leaves[i];
      *task.dest = serialTreeWork(alloc, task.depth, task.bitset, task.modulo, wi);
    });
    benchmark::DoNotOptimize(root.workResult);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

BENCHMARK_TEMPLATE(BM_serial_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree, kLargeSize)->UseRealTime();

#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK_TEMPLATE(BM_folly_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_folly_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_folly_tree, kLargeSize)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_FOLLY

BENCHMARK_TEMPLATE(BM_dispenso_tree_naive, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_naive, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_naive, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kLargeSize)->UseRealTime();

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK_TEMPLATE(BM_tbb_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_tree, kLargeSize)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

// Graduated-work benchmarks at kMediumSize (depth 16, ~65K nodes).
// Shows whether scheduling overhead matters as per-node work grows.
BENCHMARK_TEMPLATE(BM_serial_tree_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree_work, kMediumSize, kHeavyWork)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_tree_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_work, kMediumSize, kHeavyWork)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_work, kMediumSize, kHeavyWork)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_bulk_work, kMediumSize, kHeavyWork)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_parallel_invoke_work, kMediumSize, kLightWork)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_parallel_invoke_work, kMediumSize, kMediumWork)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_parallel_invoke_work, kMediumSize, kHeavyWork)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_inline_work, kMediumSize, kHeavyWork)->UseRealTime();

// Parallel-for tree experiment — base tree
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree, kLargeSize)->UseRealTime();

// Parallel-for tree experiment — graduated work
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_parfor_tree_work, kMediumSize, kHeavyWork)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_taskset_tree_hybrid_work, kMediumSize, kHeavyWork)->UseRealTime();

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK_TEMPLATE(BM_tbb_tree_work, kMediumSize, kLightWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_tree_work, kMediumSize, kMediumWork)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tbb_tree_work, kMediumSize, kHeavyWork)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_TBB

// ============================================================================
// KD-tree-style top-heavy fork-join benchmark.
//
// Real KD-tree partition: at each node, std::nth_element along a cycling
// dimension (x → y → z → x …) finds the median, then recurse on the two
// halves until chunk size <= kKdMinChunk. Total work O(N log N) — heavy at
// the root (full N partition), halving at each level. Counterpoint to piTree
// (leaf-heavy) and representative of real workloads where the root operation
// dominates.
//
// Per iteration, the working buffer is reset from a precomputed master so
// every variant sees identical input. The reset (memcpy of N * sizeof(Point3))
// is included in measured time but is small relative to the partition work.
// ============================================================================

struct KdPoint {
  float x, y, z;
};

DISPENSO_INLINE float kdDim(const KdPoint& p, int dim) {
  return dim == 0 ? p.x : (dim == 1 ? p.y : p.z);
}

constexpr size_t kKdN = 1u << 20; // 1M points
constexpr size_t kKdMinChunk = 1024; // serial below this size

const std::vector<KdPoint>& kdMaster() {
  static const std::vector<KdPoint> v = [] {
    std::vector<KdPoint> tmp(kKdN);
    std::mt19937 rng(0xc0ffeeULL);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& p : tmp) {
      p.x = dist(rng);
      p.y = dist(rng);
      p.z = dist(rng);
    }
    return tmp;
  }();
  return v;
}

DISPENSO_INLINE void kdNodePartition(KdPoint* begin, KdPoint* mid, KdPoint* end, int dim) {
  std::nth_element(begin, mid, end, [dim](const KdPoint& a, const KdPoint& b) {
    return kdDim(a, dim) < kdDim(b, dim);
  });
}

void kdSerial(KdPoint* begin, KdPoint* end, int depth) {
  size_t n = static_cast<size_t>(end - begin);
  if (n <= kKdMinChunk) {
    return;
  }
  KdPoint* mid = begin + n / 2;
  kdNodePartition(begin, mid, end, depth % 3);
  kdSerial(begin, mid, depth + 1);
  kdSerial(mid + 1, end, depth + 1);
}

void BM_kdtree_serial(benchmark::State& state) {
  const auto& master = kdMaster();
  std::vector<KdPoint> work(master.size());
  for (auto UNUSED_VAR : state) {
    std::copy(master.begin(), master.end(), work.begin());
    kdSerial(work.data(), work.data() + work.size(), 0);
    benchmark::DoNotOptimize(work.data());
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void kdTbb(KdPoint* begin, KdPoint* end, int depth) {
  size_t n = static_cast<size_t>(end - begin);
  if (n <= kKdMinChunk) {
    return;
  }
  KdPoint* mid = begin + n / 2;
  kdNodePartition(begin, mid, end, depth % 3);
  tbb::parallel_invoke(
      [=] { kdTbb(begin, mid, depth + 1); }, [=] { kdTbb(mid + 1, end, depth + 1); });
}

void BM_kdtree_tbb(benchmark::State& state) {
  const auto& master = kdMaster();
  std::vector<KdPoint> work(master.size());
  for (auto UNUSED_VAR : state) {
    std::copy(master.begin(), master.end(), work.begin());
    kdTbb(work.data(), work.data() + work.size(), 0);
    benchmark::DoNotOptimize(work.data());
  }
}
#endif

void kdTaskSetInline(dispenso::ConcurrentTaskSet& ts, KdPoint* begin, KdPoint* end, int depth) {
  size_t n = static_cast<size_t>(end - begin);
  if (n <= kKdMinChunk) {
    return;
  }
  KdPoint* mid = begin + n / 2;
  kdNodePartition(begin, mid, end, depth % 3);
  ts.schedule([&ts, begin, mid, depth] { kdTaskSetInline(ts, begin, mid, depth + 1); });
  kdTaskSetInline(ts, mid + 1, end, depth + 1);
}

void BM_kdtree_dispenso_taskset_inline(benchmark::State& state) {
  const auto& master = kdMaster();
  std::vector<KdPoint> work(master.size());
  dispenso::ConcurrentTaskSet ts(dispenso::globalThreadPool());
  for (auto UNUSED_VAR : state) {
    std::copy(master.begin(), master.end(), work.begin());
    kdTaskSetInline(ts, work.data(), work.data() + work.size(), 0);
    ts.wait();
    benchmark::DoNotOptimize(work.data());
  }
}

void kdTaskSetBulk(dispenso::ConcurrentTaskSet& ts, KdPoint* begin, KdPoint* end, int depth) {
  size_t n = static_cast<size_t>(end - begin);
  if (n <= kKdMinChunk) {
    return;
  }
  KdPoint* mid = begin + n / 2;
  kdNodePartition(begin, mid, end, depth % 3);
  std::array<std::pair<KdPoint*, KdPoint*>, 2> ranges = {
      std::make_pair(begin, mid), std::make_pair(mid + 1, end)};
  ts.scheduleBulk(2, [&ts, ranges, depth](size_t i) {
    return [&ts, ranges, i, depth]() {
      kdTaskSetBulk(ts, ranges[i].first, ranges[i].second, depth + 1);
    };
  });
}

void BM_kdtree_dispenso_taskset_bulk(benchmark::State& state) {
  const auto& master = kdMaster();
  std::vector<KdPoint> work(master.size());
  dispenso::ConcurrentTaskSet ts(dispenso::globalThreadPool());
  for (auto UNUSED_VAR : state) {
    std::copy(master.begin(), master.end(), work.begin());
    kdTaskSetBulk(ts, work.data(), work.data() + work.size(), 0);
    ts.wait();
    benchmark::DoNotOptimize(work.data());
  }
}

BENCHMARK(BM_kdtree_serial)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_kdtree_tbb)->UseRealTime();
#endif
BENCHMARK(BM_kdtree_dispenso_taskset_inline)->UseRealTime();
BENCHMARK(BM_kdtree_dispenso_taskset_bulk)->UseRealTime();

// ============================================================================
// 4-ary tree benchmark (heavy work).
//
// Counterpoint to the binary tree benchmarks: each internal node has 4
// children rather than 2, producing the same total leaf count (4^8 = 65536)
// at half the depth.  The 4-arity is where parallel_invoke's variadic
// atomic-batching is expected to pay off vs. four individual schedule()
// calls or scheduleBulk(4).
// ============================================================================

constexpr size_t k4AryDepth = 8;

DISPENSO_INLINE uint32_t fourAryKernel(uint32_t bitset, size_t workIters) {
  return busyWork(bitset, workIters);
}

uint32_t serialFourAry(uint32_t depth, uint32_t bitset, size_t workIters) {
  uint32_t v = fourAryKernel(bitset, workIters);
  if (depth == 0) {
    return v;
  }
  --depth;
  uint32_t s = v;
  for (uint32_t i = 0; i < 4; ++i) {
    s += serialFourAry(depth, (bitset << 2) | i, workIters);
  }
  return s;
}

void BM_4ary_serial(benchmark::State& state) {
  uint32_t result = 0;
  for (auto UNUSED_VAR : state) {
    result = serialFourAry(k4AryDepth, 1, kHeavyWork);
    benchmark::DoNotOptimize(result);
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
uint32_t tbbFourAry(uint32_t depth, uint32_t bitset, size_t workIters) {
  uint32_t v = fourAryKernel(bitset, workIters);
  if (depth == 0) {
    return v;
  }
  --depth;
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  tbb::parallel_invoke(
      [&] { s0 = tbbFourAry(depth, (bitset << 2) | 0, workIters); },
      [&] { s1 = tbbFourAry(depth, (bitset << 2) | 1, workIters); },
      [&] { s2 = tbbFourAry(depth, (bitset << 2) | 2, workIters); },
      [&] { s3 = tbbFourAry(depth, (bitset << 2) | 3, workIters); });
  return v + s0 + s1 + s2 + s3;
}

void BM_4ary_tbb(benchmark::State& state) {
  uint32_t result = 0;
  for (auto UNUSED_VAR : state) {
    result = tbbFourAry(k4AryDepth, 1, kHeavyWork);
    benchmark::DoNotOptimize(result);
  }
}
#endif

// dispenso 4-ary via parallel_invoke (variadic).  Each call peels the last
// functor to run inline; the other three siblings are queued via the bulk
// path for atomic batching.  parallel_invoke does NOT wait — completion is
// joined by tasks.wait() at the top of the algorithm.  No aggregation: each
// leaf does its work and feeds the result through DoNotOptimize.
void dispensoFourAryParInvoke(
    dispenso::ConcurrentTaskSet& tasks,
    uint32_t depth,
    uint32_t bitset,
    size_t workIters) {
  uint32_t v = fourAryKernel(bitset, workIters);
  benchmark::DoNotOptimize(v);
  if (depth == 0) {
    return;
  }
  --depth;
  dispenso::parallel_invoke(
      tasks,
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryParInvoke(tasks, depth, (bitset << 2) | 0, workIters);
      },
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryParInvoke(tasks, depth, (bitset << 2) | 1, workIters);
      },
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryParInvoke(tasks, depth, (bitset << 2) | 2, workIters);
      },
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryParInvoke(tasks, depth, (bitset << 2) | 3, workIters);
      });
}

void BM_4ary_dispenso_parallel_invoke(benchmark::State& state) {
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(),
      dispenso::ParentCascadeCancel::kOff,
      2,
      taskCostFor(kHeavyWork));
  for (auto UNUSED_VAR : state) {
    dispensoFourAryParInvoke(tasks, k4AryDepth, 1, kHeavyWork);
    tasks.wait();
  }
}

// dispenso 4-ary via manual schedule()×3 + inline recurse: equivalent of
// the binary "inline" idiom, generalized.  Mirrors what parallel_invoke does
// internally but without the OnceFunction array indirection.
void dispensoFourAryManualInline(
    dispenso::ConcurrentTaskSet& tasks,
    uint32_t depth,
    uint32_t bitset,
    size_t workIters) {
  uint32_t v = fourAryKernel(bitset, workIters);
  benchmark::DoNotOptimize(v);
  if (depth == 0) {
    return;
  }
  --depth;
  tasks.schedule(
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryManualInline(tasks, depth, (bitset << 2) | 0, workIters);
      },
      /*skipRecheck=*/true);
  tasks.schedule(
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryManualInline(tasks, depth, (bitset << 2) | 1, workIters);
      },
      /*skipRecheck=*/true);
  tasks.schedule(
      [&tasks, depth, bitset, workIters]() {
        dispensoFourAryManualInline(tasks, depth, (bitset << 2) | 2, workIters);
      },
      /*skipRecheck=*/true);
  dispensoFourAryManualInline(tasks, depth, (bitset << 2) | 3, workIters);
}

void BM_4ary_dispenso_manual_inline(benchmark::State& state) {
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(),
      dispenso::ParentCascadeCancel::kOff,
      2,
      taskCostFor(kHeavyWork));
  for (auto UNUSED_VAR : state) {
    dispensoFourAryManualInline(tasks, k4AryDepth, 1, kHeavyWork);
    tasks.wait();
  }
}

// dispenso 4-ary via scheduleBulk(4).  Same workload, but all 4 siblings are
// queued (no inline peel), so the calling thread is idle until tasks.wait().
void dispensoFourAryBulk(
    dispenso::ConcurrentTaskSet& tasks,
    uint32_t depth,
    uint32_t bitset,
    size_t workIters) {
  uint32_t v = fourAryKernel(bitset, workIters);
  benchmark::DoNotOptimize(v);
  if (depth == 0) {
    return;
  }
  --depth;
  tasks.scheduleBulk(4, [&tasks, depth, bitset, workIters](size_t i) {
    return [&tasks, depth, bitset, workIters, i]() {
      dispensoFourAryBulk(tasks, depth, (bitset << 2) | static_cast<uint32_t>(i), workIters);
    };
  });
}

void BM_4ary_dispenso_bulk(benchmark::State& state) {
  dispenso::ConcurrentTaskSet tasks(
      dispenso::globalThreadPool(),
      dispenso::ParentCascadeCancel::kOff,
      2,
      taskCostFor(kHeavyWork));
  for (auto UNUSED_VAR : state) {
    dispensoFourAryBulk(tasks, k4AryDepth, 1, kHeavyWork);
    tasks.wait();
  }
}

BENCHMARK(BM_4ary_serial)->UseRealTime();
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_4ary_tbb)->UseRealTime();
#endif
BENCHMARK(BM_4ary_dispenso_parallel_invoke)->UseRealTime();
BENCHMARK(BM_4ary_dispenso_manual_inline)->UseRealTime();
BENCHMARK(BM_4ary_dispenso_bulk)->UseRealTime();

BENCHMARK_MAIN();
