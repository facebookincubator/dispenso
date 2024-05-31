/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/platform.h>

// Note that this header is intended for direct inclusion into test cpps that require the
// functionality.  tids are essentially independent for tranlation units (don't expect
// coordinated/sane behavior if used from multiple cpps in the same binary).

namespace {

std::atomic<int> g_nextTid(0);
DISPENSO_THREAD_LOCAL int g_tid = -1;

inline void resetTestTid() {
  g_tid = -1;
  g_nextTid.store(0);
}

inline int getTestTid() {
  if (g_tid < 0) {
    g_tid = g_nextTid.fetch_add(1, std::memory_order_relaxed);
  }
  return g_tid;
}

} // namespace
