/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <dispenso/detail/per_thread_info.h>
#include <dispenso/graph.h>
#include <dispenso/platform.h>
#include <dispenso/task_set.h>

#include <unordered_set>

namespace dispenso {
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
  inline static void evaluateNodeConcurrently(
      dispenso::ConcurrentTaskSet& tasks,
      const N* node,
      float poolRecursiveLoadFactor = 3.0f) {
    InlineDepthGuard dGuard;

    // Process nodes in a loop, continuing inline with first ready dependent
    // to avoid task scheduling overhead on the critical path
    while (node != nullptr) {
      node->run();

      const N* inlineNext = nullptr;
      for (const dispenso::Node* const d : node->dependents_) {
        if (decNumIncompletePredecessors(static_cast<const N&>(*d), std::memory_order_acq_rel)) {
          const N* dep = static_cast<const N*>(d);
          if (inlineNext == nullptr) {
            // First ready dependent: continue with it inline
            inlineNext = dep;
          } else if (PerPoolPerThreadInfo::canInlineSchedule()) {
            // Additional ready dependents: schedule to task queue
            tasks.schedule(
                [&tasks, dep, poolRecursiveLoadFactor]() {
                  evaluateNodeConcurrently(tasks, dep, poolRecursiveLoadFactor);
                },
                false,
                poolRecursiveLoadFactor);
          } else {
            // Depth limit reached: force enqueue to prevent stack overflow
            tasks.schedule(
                [&tasks, dep, poolRecursiveLoadFactor]() {
                  evaluateNodeConcurrently(tasks, dep, poolRecursiveLoadFactor);
                },
                dispenso::ForceQueuingTag());
          }
        }
      }
      node = inlineNext;
    }
  }

  static void appendGroup(
      const dispenso::Node* /* node */,
      std::unordered_set<const std::vector<const dispenso::BiPropNode*>*>& /* groups */) {}

  static void appendGroup(
      const dispenso::BiPropNode* node,
      std::unordered_set<const std::vector<const dispenso::BiPropNode*>*>& groups) {
    const std::vector<const dispenso::BiPropNode*>* group = node->biPropSet_.get();
    if (group != nullptr) {
      groups.insert(group);
    }
  }
};

} // namespace detail
} // namespace dispenso
