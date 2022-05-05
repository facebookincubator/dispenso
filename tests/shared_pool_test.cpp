/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef _WIN32
__declspec(dllimport) void* sharedPoolA();
__declspec(dllimport) void* sharedPoolB();
#else
__attribute__((visibility("default"))) void* sharedPoolA();
__attribute__((visibility("default"))) void* sharedPoolB();
#endif

TEST(ThreadPool, SharedPool) {
  EXPECT_EQ(sharedPoolA(), sharedPoolB());
}
