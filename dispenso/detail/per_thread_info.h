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
    info().pool = pool;
  }

  static bool isParForRecursive(void* pool) {
    return (!info().pool || info().pool == pool) && info().parForRecursionLevel > 0;
  }

  static bool isPoolRecursive(void* pool) {
    return info().pool == pool;
  }

  static ParForRecursion parForRecurse() {
    return ParForRecursion(info().parForRecursionLevel);
  }

 private:
  static PerThreadInfo& info();
};

} // namespace detail
} // namespace dispenso
