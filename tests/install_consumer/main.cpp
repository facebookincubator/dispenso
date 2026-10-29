/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Built against an installed dispenso by CMakeLists.txt in this directory.
// Including the umbrella header checks that every public header was installed
// and resolves from the exported include directories alone (including the
// bundled moodycamel headers); the rest checks that the library links and runs.

#include <dispenso/dispenso.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define CONSUMER_STR2(x) #x
#define CONSUMER_STR(x) CONSUMER_STR2(x)

int main() {
  int failures = 0;

  const std::string headerVersion = CONSUMER_STR(DISPENSO_MAJOR_VERSION) "." CONSUMER_STR(
      DISPENSO_MINOR_VERSION) "." CONSUMER_STR(DISPENSO_PATCH_VERSION);
  if (headerVersion != CONSUMER_PACKAGE_VERSION) {
    std::printf(
        "Installed headers say %s, package version file says %s\n",
        headerVersion.c_str(),
        CONSUMER_PACKAGE_VERSION);
    ++failures;
  }

  constexpr int64_t kN = 100000;
  std::vector<int64_t> values(static_cast<size_t>(kN));
  dispenso::parallel_for(0, kN, [&values](int64_t i) { values[static_cast<size_t>(i)] = i; });
  int64_t sum = 0;
  for (int64_t v : values) {
    sum += v;
  }
  if (sum != kN * (kN - 1) / 2) {
    std::printf(
        "parallel_for sum %lld, expected %lld\n", (long long)sum, (long long)(kN * (kN - 1) / 2));
    ++failures;
  }

  std::atomic<int> count(0);
  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  for (int i = 0; i < 100; ++i) {
    tasks.schedule([&count]() { count.fetch_add(1, std::memory_order_relaxed); });
  }
  tasks.wait();
  if (count.load() != 100) {
    std::printf("TaskSet ran %d tasks, expected 100\n", count.load());
    ++failures;
  }

  dispenso::Future<int> answer = dispenso::async([]() { return 42; });
  if (answer.get() != 42) {
    std::printf("Future returned %d, expected 42\n", answer.get());
    ++failures;
  }

  if (failures == 0) {
    std::printf("dispenso %s install consumer OK\n", CONSUMER_PACKAGE_VERSION);
  }
  return failures == 0 ? 0 : 1;
}
