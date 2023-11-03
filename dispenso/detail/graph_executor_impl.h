/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <dispenso/graph.h>
#include <dispenso/task_set.h>

namespace detail {

class ExecutorBase {
 protected:
  inline static bool hasNoIncompletePredecessors(const dispenso::Node& node) {
    return node.numIncompletePredecessors_.load(std::memory_order_relaxed) == 0;
  }

  inline static void addIncompletePredecessor(const dispenso::Node& node) {
    if (node.isCompleted()) {
      node.numIncompletePredecessors_.store(1, std::memory_order_relaxed);
    } else {
      node.numIncompletePredecessors_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  inline static void ifIncompleteAddIncompletePredecessor(const dispenso::Node& node) {
    if (!node.isCompleted()) {
      node.numIncompletePredecessors_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  inline static bool decNumIncompletePredecessors(
      const dispenso::Node& node,
      std::memory_order order) {
    return node.numIncompletePredecessors_.fetch_sub(1, order) == 1;
  }

  inline static bool decNumIncompletePredecessors(
      const dispenso::BiPropNode& node,
      std::memory_order order) {
    const std::memory_order loadOrder =
        order == std::memory_order_relaxed ? std::memory_order_relaxed : std::memory_order_acquire;
    if (node.numIncompletePredecessors_.load(loadOrder) == dispenso::Node::kCompleted) {
      return false;
    }

    return node.numIncompletePredecessors_.fetch_sub(1, order) == 1;
  }

  template <class N>
  inline static void evaluateNodeConcurrently(dispenso::ConcurrentTaskSet& tasks, const N* node) {
    node->run();
    for (const dispenso::Node* const d : node->dependents()) {
      if (decNumIncompletePredecessors(static_cast<const N&>(*d), std::memory_order_acq_rel)) {
        tasks.schedule(
            [&tasks, d]() { evaluateNodeConcurrently(tasks, static_cast<const N*>(d)); });
      }
    }
  }
};

} // namespace detail
