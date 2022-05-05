/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/thread_pool.h>

#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__ ((visibility ("default")))
#endif
    void* DISPENSO_EXPORT_NAME() {
  return &dispenso::globalThreadPool();
}
