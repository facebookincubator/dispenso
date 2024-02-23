/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/graph.h>

#include <iterator>
#include <mutex>

namespace {
constexpr size_t kToDelete = std::numeric_limits<size_t>::max();

void set_union(
    std::vector<const dispenso::BiPropNode*>& s1,
    const std::vector<const dispenso::BiPropNode*>& s2) {
  std::vector<const dispenso::BiPropNode*> tmp(s1);
  s1.clear();
  std::set_union(tmp.cbegin(), tmp.cend(), s2.cbegin(), s2.cend(), std::back_inserter(s1));
}

void set_insert(std::vector<const dispenso::BiPropNode*>& s, const dispenso::BiPropNode* node) {
  auto it = std::upper_bound(s.begin(), s.end(), node);
  if (it == s.begin() || *(it - 1) != node) {
    s.insert(it, node);
  }
}
} // anonymous namespace

namespace dispenso {

void BiPropNode::biPropDependsOnOneNode(BiPropNode& node) {
  Node::dependsOnOneNode(node);
  if (node.biPropSet_ == nullptr && biPropSet_ == nullptr) {
    biPropSet_ = std::make_shared<std::vector<const BiPropNode*>>();
    set_insert(*biPropSet_, this);
    set_insert(*biPropSet_, &node);
    node.biPropSet_ = biPropSet_;
  } else if (node.biPropSet_ != nullptr && biPropSet_ != nullptr) {
    set_union(*biPropSet_, *node.biPropSet_);
    node.biPropSet_ = biPropSet_;
  } else if (biPropSet_ == nullptr) {
    biPropSet_ = node.biPropSet_;
    set_insert(*biPropSet_, this);
  } else {
    node.biPropSet_ = biPropSet_;
    set_insert(*biPropSet_, &node);
  }
}

template <class N>
void SubgraphT<N>::clear() {
  decrementDependentCounters();
  const size_t numGraphPredecessors = markNodesWithPredicessors();
  if (numGraphPredecessors != 0) {
    removePredecessorDependencies(numGraphPredecessors);
  }
  destroyNodes();
}

template <class N>
void SubgraphT<N>::destroyNodes() {
  for (NodeType* n : nodes_) {
    n->~NodeType();
  }
  allocator_->clear();
  nodes_.clear();
}

template <class N>
SubgraphT<N>::~SubgraphT() {
  for (NodeType* n : nodes_) {
    n->~NodeType();
  }
}

template <class N>
void SubgraphT<N>::decrementDependentCounters() {
  for (N* node : nodes_) {
    for (Node* const dependent : node->dependents_) {
      dependent->numPredecessors_--;
    }
    removeNodeFromBiPropSet(node);
  }
}

template <class N>
size_t SubgraphT<N>::markNodesWithPredicessors() {
  size_t numGraphPredecessors = 0;
  for (N* node : nodes_) {
    if (node->numPredecessors_ != 0) {
      numGraphPredecessors += node->numPredecessors_;
      node->numPredecessors_ = kToDelete;
    }
  }
  return numGraphPredecessors;
}

template <class N>
void SubgraphT<N>::removePredecessorDependencies(size_t numGraphPredecessors) {
  for (SubgraphT<N>& subgraph : graph_->subgraphs_) {
    if (&subgraph == this) {
      continue;
    }
    for (N* node : subgraph.nodes_) {
      std::vector<Node*>& dependents = node->dependents_;
      size_t num = dependents.size();
      for (size_t i = 0; i < num;) {
        if (dependents[i]->numPredecessors_ == kToDelete) {
          dependents[i] = dependents[num - 1];
          --num;
          if (--numGraphPredecessors == 0) {
            dependents.resize(num);
            return;
          }
        } else {
          i++;
        }
      }
      dependents.resize(num);
    }
  }
}

namespace {
constexpr size_t kMaxCache = 8;
// Don't cache too-large allocators.  This way we will have at most 8*(2**16) = 512K outstanding
// nodes worth of memory per node type.
// TODO(bbudge): Make these caching values macro configurable for lightweight platforms.
constexpr size_t kMaxChunkCapacity = 1 << 16;

using AlignedNodePoolPtr =
    std::unique_ptr<NoLockPoolAllocator, detail::AlignedFreeDeleter<NoLockPoolAllocator>>;

std::vector<AlignedNodePoolPtr> g_sgcache[2];
std::mutex g_sgcacheMtx;

template <class T>
constexpr size_t kCacheIndex = size_t{std::is_same<T, BiPropNode>::value};

} // namespace

template <class N>
typename SubgraphT<N>::PoolPtr SubgraphT<N>::getAllocator() {
  AlignedNodePoolPtr ptr;

  auto& cache = g_sgcache[kCacheIndex<N>];

  {
    std::lock_guard<std::mutex> lk(g_sgcacheMtx);
    if (cache.empty()) {
      void* alloc =
          detail::alignedMalloc(sizeof(NoLockPoolAllocator), alignof(NoLockPoolAllocator));
      auto* pool = new (alloc)
          NoLockPoolAllocator(sizeof(NodeType), 128 * sizeof(NodeType), ::malloc, ::free);
      ptr.reset(pool);
    } else {
      ptr = std::move(cache.back());
      ptr->clear();
      cache.pop_back();
    }
  }
  return PoolPtr(ptr.release(), releaseAllocator);
}

template <class N>
void SubgraphT<N>::releaseAllocator(NoLockPoolAllocator* ptr) {
  if (!ptr) {
    return;
  }
  if (ptr->totalChunkCapacity() < kMaxChunkCapacity) {
    auto& cache = g_sgcache[kCacheIndex<N>];
    {
      std::lock_guard<std::mutex> lk(g_sgcacheMtx);
      if (cache.size() < kMaxCache) {
        cache.emplace_back(ptr);
        return;
      }
    }
  }
  detail::AlignedFreeDeleter<NoLockPoolAllocator>()(ptr);
}

template <class N>
GraphT<N>::GraphT(GraphT<N>&& other) : subgraphs_(std::move(other.subgraphs_)) {
  for (SubgraphT<N>& subgraph : subgraphs_) {
    subgraph.graph_ = this;
  }
}

template <class N>
GraphT<N>& GraphT<N>::operator=(GraphT&& other) noexcept {
  subgraphs_ = std::move(other.subgraphs_);
  for (SubgraphT<N>& subgraph : subgraphs_) {
    subgraph.graph_ = this;
  }
  return *this;
}

template <class N>
SubgraphT<N>& GraphT<N>::addSubgraph() {
  subgraphs_.push_back(SubgraphType(this));
  return subgraphs_.back();
}

template class DISPENSO_DLL_ACCESS SubgraphT<Node>;
template class DISPENSO_DLL_ACCESS SubgraphT<BiPropNode>;
template class DISPENSO_DLL_ACCESS GraphT<Node>;
template class DISPENSO_DLL_ACCESS GraphT<BiPropNode>;
} // namespace dispenso
