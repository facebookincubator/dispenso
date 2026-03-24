/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Force the opposite cache-pointer mode for this platform so CI exercises
// whichever code path is normally skipped.
#if !defined(__aarch64__) && !defined(_M_ARM64)
#define DISPENSO_HAS_CACHED_PTRS 0
#else
#define DISPENSO_HAS_CACHED_PTRS 1
#endif

#include "concurrent_vector_test_common_types.h"

using TestTraitsTypes = ::testing::Types<dispenso::DefaultConcurrentVectorTraits>;
DISPENSO_DISABLE_WARNING_PUSH
DISPENSO_DISABLE_WARNING_ZERO_VARIADIC_MACRO_ARGUMENTS
TYPED_TEST_SUITE(ConcurrentVectorTest, TestTraitsTypes);
DISPENSO_DISABLE_WARNING_POP

#include "concurrent_vector_test_common.h"
