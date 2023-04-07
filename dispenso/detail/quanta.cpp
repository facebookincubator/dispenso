/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/detail/quanta.h>

namespace dispenso {
#ifdef _WIN32
#include <Windows.h>
#include <timeapi.h>

namespace {
struct OsQuantaSetter {
  OsQuantaSetter() {
    timeBeginPeriod(1);
  }
  ~OsQuantaSetter() {
    timeEndPeriod(1);
  }
};
} // namespace
#else
namespace {
struct OsQuantaSetter {};
} // namespace

#endif // _WIN32

namespace detail {
void registerFineSchedulerQuanta() {
  static OsQuantaSetter setter;
  (void)setter;
}
} // namespace detail
} // namespace dispenso
