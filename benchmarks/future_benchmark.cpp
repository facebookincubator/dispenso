// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <cmath>
#include <future>
#include <iostream>
#include <random>

#include <dispenso/future.h>

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

#include "thread_benchmark_common.h"

constexpr size_t kSmallSize = 13;
constexpr size_t kMediumSize = 16;
constexpr size_t kLargeSize = 19;

struct Node {
  Node* left;
  Node* right;
  uint32_t value;

  void setValue(uint32_t unique_bitset, uint32_t modulo) {
    value = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      value += unique_bitset % modulo;
      unique_bitset /= modulo;
    }
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

  for (auto _ : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = serialTree(alloc, depth, 1, modulo);
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(root, depth, modulo);
}

Node* stdTree(Allocator& allocator, uint32_t depth, uint32_t bitset, uint32_t modulo) {
  --depth;
  Node* node = allocator.alloc();
  node->setValue(bitset, modulo);
  if (!depth) {
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }
  auto left = std::async([&]() { return stdTree(allocator, depth, (bitset << 1), modulo); });
  auto right = std::async([&]() { return stdTree(allocator, depth, (bitset << 1) | 1, modulo); });
  node->left = left.get();
  node->right = right.get();

  return node;
}

template <size_t depth>
void BM_std_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;

  Node* root;

  size_t m = 0;

  for (auto _ : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = stdTree(alloc, depth, 1, modulo);
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

  auto left =
      dispenso::async([&]() { return dispensoTree(allocator, depth, (bitset << 1), modulo); });
  auto right =
      dispenso::async([&]() { return dispensoTree(allocator, depth, (bitset << 1) | 1, modulo); });
  node->left = left.get();
  node->right = right.get();

  return node;
}

template <size_t depth>
void BM_dispenso_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();
  dispenso::globalThreadPool();

  uint32_t modulo;

  Node* root;

  size_t m = 0;

  for (auto _ : state) {
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

  for (auto _ : state) {
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

  tasks.schedule([&tasks, &allocator, node, depth, bitset, modulo]() {
    node->left = allocator.alloc();
    dispensoTaskSetTree(tasks, node->left, allocator, depth, (bitset << 1), modulo);
  });
  tasks.schedule([&tasks, &allocator, node, depth, bitset, modulo]() {
    node->right = allocator.alloc();
    dispensoTaskSetTree(tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo);
  });
}

template <size_t depth>
void BM_taskset_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());

  size_t m = 0;

  for (auto _ : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTree(tasks, &root, alloc, depth, 1, modulo);
    tasks.wait();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(&root, depth, modulo);
}

void dispensoTaskSetTreeOpt(
    dispenso::ConcurrentTaskSet& tasks,
    Node* node,
    Allocator& allocator,
    uint32_t depth,
    uint32_t bitset,
    uint32_t modulo) {
  node->setValue(bitset, modulo);
  --depth;

  if (depth < 5) {
    node->left = serialTree(allocator, depth, (bitset << 1), modulo);
    node->right = serialTree(allocator, depth, (bitset << 1) | 1, modulo);
    return;
  }

  tasks.schedule([&tasks, &allocator, node, depth, bitset, modulo]() {
    node->left = allocator.alloc();
    dispensoTaskSetTreeOpt(tasks, node->left, allocator, depth, (bitset << 1), modulo);
  });
  tasks.schedule([&tasks, &allocator, node, depth, bitset, modulo]() {
    node->right = allocator.alloc();
    dispensoTaskSetTreeOpt(tasks, node->right, allocator, depth, (bitset << 1) | 1, modulo);
  });
}

template <size_t depth>
void BM_tasksetopt_tree(benchmark::State& state) {
  Allocator alloc;
  alloc.reset(depth);
  getModulos();

  uint32_t modulo;
  Node root;

  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());

  size_t m = 0;

  for (auto _ : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    dispensoTaskSetTreeOpt(tasks, &root, alloc, depth, 1, modulo);
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

  for (auto _ : state) {
    alloc.reset(depth);
    modulo = getModulos()[m];
    root = dispensoTreeWhenAll(alloc, depth, 1, modulo).get();
    m = (m + 1 == getModulos().size()) ? 0 : m + 1;
  }

  checkTree(root, depth, modulo);
}

BENCHMARK_TEMPLATE(BM_serial_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_serial_tree, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_std_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_std_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_std_tree, kLargeSize)->UseRealTime();

#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK_TEMPLATE(BM_folly_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_folly_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_folly_tree, kLargeSize)->UseRealTime();
#endif // !BENCHMARK_WITHOUT_FOLLY

BENCHMARK_TEMPLATE(BM_dispenso_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_taskset_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_taskset_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_taskset_tree, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_tasksetopt_tree, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tasksetopt_tree, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_tasksetopt_tree, kLargeSize)->UseRealTime();

BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kSmallSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kMediumSize)->UseRealTime();
BENCHMARK_TEMPLATE(BM_dispenso_tree_when_all, kLargeSize)->UseRealTime();

BENCHMARK_MAIN();
