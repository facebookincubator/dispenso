// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#pragma once

#include <dispenso/platform.h>

namespace dispenso {
namespace detail {

struct alignas(kCacheLineSize) PerThreadInfo {
  void* pool = nullptr;
  int parForRecursionLevel = 0;
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
  static void registerPool(void* pool) {
    info_.pool = pool;
  }

  static bool isParForRecursive(void* pool) {
    return (!info_.pool || info_.pool == pool) && info_.parForRecursionLevel > 0;
  }

  static bool isPoolRecursive(void* pool) {
    return info_.pool == pool;
  }

  static ParForRecursion parForRecurse() {
    return ParForRecursion(info_.parForRecursionLevel);
  }

 private:
  static DISPENSO_THREAD_LOCAL PerThreadInfo info_;
};

} // namespace detail
} // namespace dispenso
