/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @example cpu_set_example.cpp
 * Demonstrates CPU topology queries and thread affinity with dispenso::CpuSet.
 */

#include <dispenso/cpu_set.h>

#include <iostream>

int main() {
  // Example 1: Query available hardware threads
  std::cout << "Example 1: Hardware topology overview\n";
  {
    int32_t available = dispenso::CpuSet::availableCount();
    int32_t numaNodes = dispenso::CpuSet::totalNumaNodes();
    std::cout << "  Available CPUs: " << available << "\n";
    std::cout << "  NUMA nodes: " << numaNodes << "\n";
  }

  // Example 2: Inspect NUMA node CPU sets
  std::cout << "\nExample 2: CPUs per NUMA node\n";
  {
    int32_t nodes = dispenso::CpuSet::totalNumaNodes();
    for (int32_t i = 0; i < nodes; ++i) {
      const dispenso::CpuSet& nodeSet = dispenso::CpuSet::node(i);
      std::cout << "  Node " << i << ": " << nodeSet.count() << " CPUs\n";
    }
  }

  // Example 3: Query cache sharing groups
  std::cout << "\nExample 3: Cache topology\n";
  {
    const auto& l2Groups = dispenso::CpuSet::l2CacheGroups();
    const auto& l3Groups = dispenso::CpuSet::l3CacheGroups();
    std::cout << "  L2 cache groups: " << l2Groups.size() << "\n";
    std::cout << "  L3 cache groups: " << l3Groups.size() << "\n";

    // Show the first L3 group's CPUs (if available)
    if (!l3Groups.empty()) {
      const auto& group = l3Groups[0];
      std::cout << "  L3 group 0 (cache ID " << group.cacheId << "): ";
      for (int32_t cpu : group.cpus) {
        std::cout << cpu << " ";
      }
      std::cout << "\n";
    }
  }

  // Example 4: Build and display scheduling thread groups
  std::cout << "\nExample 4: Scheduling thread groups\n";
  {
    auto groups = dispenso::CpuSet::buildThreadGroups();
    std::cout << "  Thread groups (max " << dispenso::kDefaultMaxGroupSize
              << " CPUs each): " << groups.size() << "\n";
    for (size_t i = 0; i < groups.size() && i < 4; ++i) {
      std::cout << "  Group " << i << ": " << groups[i].cpus.size() << " CPUs [";
      for (size_t j = 0; j < groups[i].cpus.size() && j < 8; ++j) {
        if (j > 0) {
          std::cout << ", ";
        }
        std::cout << groups[i].cpus[j];
      }
      if (groups[i].cpus.size() > 8) {
        std::cout << ", ...";
      }
      std::cout << "]\n";
    }
    if (groups.size() > 4) {
      std::cout << "  ... (" << groups.size() - 4 << " more groups)\n";
    }
  }

  // Example 5: Create a CpuSet and bind the current thread
  std::cout << "\nExample 5: Manual CpuSet creation and thread binding\n";
  {
    dispenso::CpuSet set;
    set.addRange(0, 4); // CPUs 0, 1, 2, 3
    std::cout << "  CpuSet count: " << set.count() << "\n";
    std::cout << "  Contains CPU 2: " << (set.contains(2) ? "yes" : "no") << "\n";
    std::cout << "  Contains CPU 5: " << (set.contains(5) ? "yes" : "no") << "\n";

    // Bind the calling thread (succeeds on Linux, no-op on macOS)
    bool bound = set.bindCurrentThread();
    std::cout << "  Bind result: " << (bound ? "success" : "unsupported or failed") << "\n";
  }

  // Example 6: Query current hardware thread
  std::cout << "\nExample 6: Current hardware thread\n";
  {
    int32_t cpu = dispenso::CpuSet::currentHardwareThread();
    if (cpu >= 0) {
      std::cout << "  Running on CPU " << cpu << "\n";
    } else {
      std::cout << "  CPU query not supported on this platform\n";
    }
  }

  std::cout << "\nAll CpuSet examples completed successfully!\n";
  return 0;
}
