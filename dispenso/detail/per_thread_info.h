/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/platform.h>

namespace dispenso {
namespace detail {

struct alignas(kCacheLineSize) PerThreadInfo {
  void* pool = nullptr;
  void* producer = nullptr;
  int parForRecursionLevel = 0;
  // Index into ThreadPool's per-thread ring array, or -1 if not a pool thread.
  int32_t ringIndex = -1;
};

class ParForRecursion {
 public:
  ~ParForRecursion() {
    --parForRecursionLevel_;
  }

 private:
  ParForRecursion(int& parForRecursionLevel) : parForRecursionLevel_(parForRecursionLevel) {
    ++parForRecursionLevel_;
  }

  int& parForRecursionLevel_;
  friend class PerPoolPerThreadInfo;
};

class PerPoolPerThreadInfo {
 public:
  static void registerPool(void* pool, void* producer, int32_t ringIndex = -1) {
    auto& i = info();
    i.pool = pool;
    i.producer = producer;
    i.ringIndex = ringIndex;
  }

  static void* producer(void* pool) {
    auto& i = info();
    return i.pool == pool ? i.producer : nullptr;
  }

  static int32_t ringIndex(void* pool) {
    auto& i = info();
    return i.pool == pool ? i.ringIndex : -1;
  }

  static bool isParForRecursive(void* pool) {
    auto& i = info();
    return (!i.pool || i.pool == pool) && i.parForRecursionLevel > 0;
  }

  static bool isPoolRecursive(void* pool) {
    return info().pool == pool;
  }

  static ParForRecursion parForRecurse() {
    return ParForRecursion(info().parForRecursionLevel);
  }

 private:
  DISPENSO_DLL_ACCESS static PerThreadInfo& info();
};

} // namespace detail
} // namespace dispenso
