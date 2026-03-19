/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <dispenso/graph.h>
#include <dispenso/platform.h>
#include <dispenso/task_set.h>

#include <unordered_set>

namespace detail {

// Maximum recursion depth for inline graph node scheduling before forcing work
// through the thread pool queue. Prevents unbounded stack growth when CTS inlines
// evaluateNodeConcurrently calls for additional ready dependents.
static constexpr int kMaxGraphInlineDepth = 32;

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
    // Track recursion depth to prevent stack overflow. CTS may inline scheduled
    // lambdas when load is high, causing evaluateNodeConcurrently to recurse
    // through the schedule→inline→evaluate path. When depth exceeds the limit,
    // force work through the pool queue to bound stack growth.
    static DISPENSO_THREAD_LOCAL int depth = 0;
    struct DepthGuard {
      DISPENSO_INLINE DepthGuard(int& d) : d_(d) {
        ++d_;
      }
      DISPENSO_INLINE ~DepthGuard() {
        --d_;
      }
      int& d_;
    };
    DepthGuard dGuard(depth);

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
          } else if (depth < kMaxGraphInlineDepth) {
            // Additional ready dependents: schedule to task queue
            tasks.schedule([&tasks, dep]() { evaluateNodeConcurrently(tasks, dep); });
          } else {
            // Depth limit reached: force enqueue to prevent stack overflow
            tasks.schedule(
                [&tasks, dep]() { evaluateNodeConcurrently(tasks, dep); },
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
