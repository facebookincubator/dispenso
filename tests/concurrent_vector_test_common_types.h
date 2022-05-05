/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/concurrent_vector.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <dispenso/parallel_for.h>
#include <gtest/gtest.h>

template <typename Traits>
class ConcurrentVectorTest : public testing::Test {
 public:
};

using dispenso::ConcurrentVectorReallocStrategy;

struct TestTraitsA {
  static constexpr bool kPreferBuffersInline = false;
  static constexpr ConcurrentVectorReallocStrategy kReallocStrategy =
      ConcurrentVectorReallocStrategy::kHalfBufferAhead;
  static constexpr bool kIteratorPreferSpeed = false;
};

struct TestTraitsB {
  static constexpr bool kPreferBuffersInline = true;
  static constexpr ConcurrentVectorReallocStrategy kReallocStrategy =
      ConcurrentVectorReallocStrategy::kFullBufferAhead;
  static constexpr bool kIteratorPreferSpeed = true;
};
