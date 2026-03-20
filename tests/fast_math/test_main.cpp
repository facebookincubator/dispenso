/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Rosetta detection for fast_math SIMD tests.
// When running under Rosetta (x86 emulation on ARM Mac), Cody-Waite range
// reduction accuracy breaks because FMA is emulated as separate multiply+add.
// This file registers a GoogleTest environment that disables all tests when
// Rosetta is detected.

#if defined(__APPLE__) && defined(__x86_64__)
#include <sys/sysctl.h>
#include <cstdio>

#include <gtest/gtest.h>

static bool isRosetta() {
  int ret = 0;
  size_t size = sizeof(ret);
  return sysctlbyname("sysctl.proc_translated", &ret, &size, nullptr, 0) == 0 && ret == 1;
}

class RosettaGuard : public testing::Environment {
 public:
  void SetUp() override {
    if (isRosetta()) {
      printf("Rosetta detected: skipping fast_math SIMD tests.\n");
      printf("FMA emulation breaks Cody-Waite range reduction accuracy.\n");
      printf("Run on native ARM or native x86 hardware.\n");
      // Set filter to match no tests, effectively skipping all.
      testing::GTEST_FLAG(filter) = "DISABLED_RosettaSkip";
    }
  }
};

// Register before main() runs via global constructor.
static auto* rosetta_env [[maybe_unused]] = testing::AddGlobalTestEnvironment(new RosettaGuard());
#endif // __APPLE__ && __x86_64__
