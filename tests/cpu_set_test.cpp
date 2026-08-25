/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/cpu_set.h>

#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>

#include <cstring>
#endif // __linux__

#include <gtest/gtest.h>

using dispenso::CpuSet;

#if defined(__linux__)
namespace {
// Whether the kernel reports any cache topology at all.
//
// /sys/devices/system/cpu/cpu0/cache is populated only when the kernel has
// cache information to expose, and plenty of virtual machines have none --
// GitHub's aarch64 runners among them. There, an empty group list is the
// kernel's answer rather than a parsing failure. Probing for the directory
// keeps the distinction: skipping merely because the list came back empty
// would also swallow a real regression on hardware that does report topology,
// which for a locality-aware scheduler is precisely the bug worth catching.
bool sysfsHasCacheTopology() {
  DIR* dir = ::opendir("/sys/devices/system/cpu/cpu0/cache");
  if (dir == nullptr) {
    return false;
  }
  bool found = false;
  while (const dirent* entry = ::readdir(dir)) {
    if (std::strncmp(entry->d_name, "index", 5) == 0) {
      found = true;
      break;
    }
  }
  ::closedir(dir);
  return found;
}
} // namespace
#endif // __linux__

// =============================================================================
// CPU List Parsing Tests
// =============================================================================

TEST(CpuSet, ParseSingleCpu) {
  std::string line = "5";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  EXPECT_TRUE(set.contains(5));
  EXPECT_FALSE(set.contains(0));
  EXPECT_FALSE(set.contains(4));
  EXPECT_FALSE(set.contains(6));
  EXPECT_EQ(set.count(), 1);
}

TEST(CpuSet, ParseSimpleRange) {
  std::string line = "0-3";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  for (int i = 0; i <= 3; ++i) {
    EXPECT_TRUE(set.contains(i)) << "Missing CPU " << i;
  }
  EXPECT_FALSE(set.contains(4));
  EXPECT_EQ(set.count(), 4);
}

TEST(CpuSet, ParseCommaSeparatedValues) {
  std::string line = "1,2";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  EXPECT_TRUE(set.contains(1));
  EXPECT_TRUE(set.contains(2));
  EXPECT_FALSE(set.contains(0));
  EXPECT_FALSE(set.contains(3));
  EXPECT_EQ(set.count(), 2);
}

TEST(CpuSet, ParseMixedRangesAndValues) {
  std::string line = "0-3,8-11,16";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  EXPECT_EQ(set.count(), 9);
  for (int i = 0; i <= 3; ++i) {
    EXPECT_TRUE(set.contains(i));
  }
  for (int i = 4; i <= 7; ++i) {
    EXPECT_FALSE(set.contains(i));
  }
  for (int i = 8; i <= 11; ++i) {
    EXPECT_TRUE(set.contains(i));
  }
  EXPECT_FALSE(set.contains(15));
  EXPECT_TRUE(set.contains(16));
  EXPECT_FALSE(set.contains(17));
}

TEST(CpuSet, ParseNonContiguousRanges) {
  // Typical dual-socket with SMT: node0 = 0-63,128-191
  std::string line = "0-63,128-191";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  EXPECT_EQ(set.count(), 128);
  for (int i = 0; i <= 63; ++i) {
    EXPECT_TRUE(set.contains(i));
  }
  for (int i = 64; i <= 127; ++i) {
    EXPECT_FALSE(set.contains(i));
  }
  for (int i = 128; i <= 191; ++i) {
    EXPECT_TRUE(set.contains(i));
  }
}

TEST(CpuSet, ParseWithSpaces) {
  // sysfs files sometimes have trailing whitespace
  std::string line = "1, 2";
  auto set = dispenso::detail::parseLinuxCpuList(line.data());
  EXPECT_TRUE(set.contains(1));
  // Note: atoi(" 2") returns 2, so this should work
  EXPECT_TRUE(set.contains(2));
}

TEST(CpuSet, ParseEmptyString) {
  auto set = dispenso::detail::parseLinuxCpuList("");
  EXPECT_EQ(set.count(), 0);
}

TEST(CpuSet, ParseNonNumericInput) {
  auto set = dispenso::detail::parseLinuxCpuList("abc");
  EXPECT_EQ(set.count(), 0);
}

TEST(CpuSet, ParseTrailingComma) {
  auto set = dispenso::detail::parseLinuxCpuList("1,");
  EXPECT_EQ(set.count(), 1);
  EXPECT_TRUE(set.contains(1));
}

// =============================================================================
// FreeBSD topology_spec Parsing Tests
//
// These drive the parser directly with synthetic kern.sched.topology_spec XML,
// so nested and multi-socket paths are covered on every platform.
// =============================================================================

using dispenso::detail::parseCacheGroupsFromTopologySpec;

// Flat single L3 and no L2, the shape a 4-core FreeBSD 15.1 arm64 machine emits.
TEST(CpuSet, TopologySpecFlatL3) {
  const std::string xml = R"(<groups>
 <group level="1" cache-level="3">
  <cpu count="4" mask="f,0,0,0">0, 1, 2, 3</cpu>
 </group>
</groups>)";
  auto l3 = parseCacheGroupsFromTopologySpec(xml, 3);
  ASSERT_EQ(l3.size(), 1u);
  EXPECT_EQ(l3[0].cpus, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec(xml, 2).empty());
}

// L3 over two L2 sibling groups: the matching group's own <cpu> must be taken,
// not a nested child's, and both nested L2 groups must be found.
TEST(CpuSet, TopologySpecNestedL3OverL2) {
  const std::string xml = R"(<groups>
 <group level="1" cache-level="3">
  <cpu count="4" mask="f">0, 1, 2, 3</cpu>
  <children>
   <group level="2" cache-level="2">
    <cpu count="2" mask="3">0, 1</cpu>
   </group>
   <group level="2" cache-level="2">
    <cpu count="2" mask="c">2, 3</cpu>
   </group>
  </children>
 </group>
</groups>)";
  auto l3 = parseCacheGroupsFromTopologySpec(xml, 3);
  ASSERT_EQ(l3.size(), 1u);
  EXPECT_EQ(l3[0].cpus, (std::vector<int32_t>{0, 1, 2, 3}));

  auto l2 = parseCacheGroupsFromTopologySpec(xml, 2);
  ASSERT_EQ(l2.size(), 2u);
  EXPECT_EQ(l2[0].cpus, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(l2[1].cpus, (std::vector<int32_t>{2, 3}));
}

// Two sockets: top group shares no cache (cache-level=0), each socket an L3.
TEST(CpuSet, TopologySpecMultiSocketL3) {
  const std::string xml = R"(<groups>
 <group level="1" cache-level="0">
  <cpu count="8" mask="ff">0, 1, 2, 3, 4, 5, 6, 7</cpu>
  <children>
   <group level="2" cache-level="3">
    <cpu count="4" mask="f">0, 1, 2, 3</cpu>
   </group>
   <group level="2" cache-level="3">
    <cpu count="4" mask="f0">4, 5, 6, 7</cpu>
   </group>
  </children>
 </group>
</groups>)";
  auto l3 = parseCacheGroupsFromTopologySpec(xml, 3);
  ASSERT_EQ(l3.size(), 2u);
  EXPECT_EQ(l3[0].cpus, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(l3[1].cpus, (std::vector<int32_t>{4, 5, 6, 7}));
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec(xml, 2).empty());
}

// Groups emitted out of CPU order must come back sorted by first CPU id.
TEST(CpuSet, TopologySpecSortsByFirstCpu) {
  const std::string xml = R"(<groups>
 <group level="1" cache-level="2">
  <cpu count="2" mask="30">4, 5</cpu>
 </group>
 <group level="1" cache-level="2">
  <cpu count="2" mask="3">0, 1</cpu>
 </group>
</groups>)";
  auto l2 = parseCacheGroupsFromTopologySpec(xml, 2);
  ASSERT_EQ(l2.size(), 2u);
  EXPECT_EQ(l2[0].cpus, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(l2[1].cpus, (std::vector<int32_t>{4, 5}));
}

// Kernel reports no cache levels (cf. the SMP(4) example): both queries empty.
TEST(CpuSet, TopologySpecNoMatchingCacheLevel) {
  const std::string xml = R"(<groups>
 <group level="1" cache-level="0">
  <cpu count="4" mask="f">0, 1, 2, 3</cpu>
  <children>
   <group level="2" cache-level="0">
    <cpu count="2" mask="3">0, 1</cpu>
   </group>
   <group level="2" cache-level="0">
    <cpu count="2" mask="c">2, 3</cpu>
   </group>
  </children>
 </group>
</groups>)";
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec(xml, 2).empty());
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec(xml, 3).empty());
}

// Empty, non-XML, and truncated inputs must not crash and yield no groups.
TEST(CpuSet, TopologySpecMalformedIsSafe) {
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("", 3).empty());
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("not xml at all", 3).empty());
  // Matching group but the CPU list is never closed with </cpu>.
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"><cpu count=\"2\">0, 1", 3)
                  .empty());
  // Group tag itself never closed.
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"", 3).empty());
}

// One input per defensive early-out in the parser.
TEST(CpuSet, TopologySpecDefensiveBranches) {
  // cache-level attribute quote is never closed.
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("<group cache-level=\"3><cpu>1</cpu>", 3).empty());
  // Matching group with no <cpu> element at all.
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"></group>", 3).empty());
  // The only <cpu> belongs to a nested child group, not the matching parent.
  {
    const std::string xml = R"(<group cache-level="3">
 <children>
  <group cache-level="2">
   <cpu count="1" mask="1">0</cpu>
  </group>
 </children>
</group>)";
    EXPECT_TRUE(parseCacheGroupsFromTopologySpec(xml, 3).empty());
    auto l2 = parseCacheGroupsFromTopologySpec(xml, 2);
    ASSERT_EQ(l2.size(), 1u);
    EXPECT_EQ(l2[0].cpus, (std::vector<int32_t>{0}));
  }
  // Matching group closes before a stray <cpu> that is not its own.
  EXPECT_TRUE(
      parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"></group><cpu>5</cpu>", 3).empty());
  // <cpu tag never closed with '>'.
  EXPECT_TRUE(parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"><cpu", 3).empty());
  // Trailing separators after the last id.
  {
    auto g = parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"><cpu>1, </cpu>", 3);
    ASSERT_EQ(g.size(), 1u);
    EXPECT_EQ(g[0].cpus, (std::vector<int32_t>{1}));
  }
  // Non-numeric id list.
  EXPECT_TRUE(
      parseCacheGroupsFromTopologySpec("<group cache-level=\"3\"><cpu>abc</cpu>", 3).empty());
}

// A zero-padded cache-level value parses by numeric value, regardless of width.
TEST(CpuSet, TopologySpecPaddedCacheLevel) {
  auto g = parseCacheGroupsFromTopologySpec(
      "<group cache-level=\"00000003\"><cpu count=\"1\" mask=\"2\">1</cpu></group>", 3);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(g[0].cpus, (std::vector<int32_t>{1}));
}

// =============================================================================
// buildGroupsFromCacheTopology (cache-aware thread grouping)
//
// Driven directly with synthetic L2/L3 cache groups so the grouping algorithm
// is exercised on every platform, including CI sandboxes without real topology
// or affinity permissions. The live-topology test below runs the same algorithm
// on real hardware when it is available.
// =============================================================================

using dispenso::CacheGroup;
using dispenso::ThreadGroup;
using dispenso::detail::buildGroupsFromCacheTopology;

// Two L3 caches, each holding two L2 atoms. With room for a full L3 worth of
// CPUs, each thread group should gather one L3's atoms and stop at the boundary.
TEST(CpuSet, BuildGroupsFromTopologyRespectsL3Boundaries) {
  const std::vector<CacheGroup> l2{{{0, 1}, 0}, {{2, 3}, 1}, {{4, 5}, 2}, {{6, 7}, 3}};
  const std::vector<CacheGroup> l3{{{0, 1, 2, 3}, 0}, {{4, 5, 6, 7}, 1}};

  const auto groups = buildGroupsFromCacheTopology(l2, l3, /*maxGroupSize=*/4);

  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].cpus, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(groups[1].cpus, (std::vector<int32_t>{4, 5, 6, 7}));
}

// maxGroupSize caps how many L2 atoms merge: with room for only one atom, each
// L2 becomes its own thread group.
TEST(CpuSet, BuildGroupsFromTopologyRespectsMaxGroupSize) {
  const std::vector<CacheGroup> l2{{{0, 1}, 0}, {{2, 3}, 1}, {{4, 5}, 2}, {{6, 7}, 3}};
  const std::vector<CacheGroup> l3{{{0, 1, 2, 3}, 0}, {{4, 5, 6, 7}, 1}};

  const auto groups = buildGroupsFromCacheTopology(l2, l3, /*maxGroupSize=*/2);

  ASSERT_EQ(groups.size(), 4u);
  EXPECT_EQ(groups[0].cpus, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(groups[1].cpus, (std::vector<int32_t>{2, 3}));
  EXPECT_EQ(groups[2].cpus, (std::vector<int32_t>{4, 5}));
  EXPECT_EQ(groups[3].cpus, (std::vector<int32_t>{6, 7}));
}

// maxGroupSize is clamped up to the largest L2 atom, so cores sharing an L2 are
// never split even when the caller asks for a smaller group.
TEST(CpuSet, BuildGroupsFromTopologyNeverSplitsAnL2Atom) {
  const std::vector<CacheGroup> l2{{{0, 1, 2, 3}, 0}};
  const std::vector<CacheGroup> l3{{{0, 1, 2, 3}, 0}};

  const auto groups = buildGroupsFromCacheTopology(l2, l3, /*maxGroupSize=*/2);

  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].cpus, (std::vector<int32_t>{0, 1, 2, 3}));
}

// When real cache topology is available, run the same algorithm on it and check
// the core invariant: no thread group spans more than one L3 cache. Skips on
// hosts (e.g. CI sandboxes) that expose no topology.
TEST(CpuSet, BuildGroupsFromRealTopologyStaysWithinL3) {
  const auto& l2 = CpuSet::l2CacheGroups();
  const auto& l3 = CpuSet::l3CacheGroups();
  if (l2.empty() || l3.empty()) {
    GTEST_SKIP() << "No L2/L3 cache topology on this host";
  }

  const auto groups = buildGroupsFromCacheTopology(l2, l3, /*maxGroupSize=*/1 << 20);

  for (const auto& tg : groups) {
    ASSERT_FALSE(tg.cpus.empty());
    const bool withinOneL3 = std::any_of(l3.begin(), l3.end(), [&](const CacheGroup& c) {
      const std::set<int32_t> l3cpus(c.cpus.begin(), c.cpus.end());
      return std::all_of(
          tg.cpus.begin(), tg.cpus.end(), [&](int32_t cpu) { return l3cpus.count(cpu) > 0; });
    });
    EXPECT_TRUE(withinOneL3) << "thread group starting at CPU " << tg.cpus[0]
                             << " spans multiple L3 caches";
  }
}

// =============================================================================
// CpuSet Manipulation Tests
// =============================================================================

TEST(CpuSet, DefaultConstructionIsEmpty) {
  CpuSet set;
  EXPECT_EQ(set.count(), 0);
  EXPECT_FALSE(set.contains(0));
}

TEST(CpuSet, AddAndContains) {
  CpuSet set;
  set.add(5);
  EXPECT_TRUE(set.contains(5));
  EXPECT_FALSE(set.contains(4));
  EXPECT_FALSE(set.contains(6));
  EXPECT_EQ(set.count(), 1);
}

TEST(CpuSet, AddRange) {
  CpuSet set;
  set.addRange(4, 8);
  EXPECT_EQ(set.count(), 4);
  EXPECT_FALSE(set.contains(3));
  EXPECT_TRUE(set.contains(4));
  EXPECT_TRUE(set.contains(5));
  EXPECT_TRUE(set.contains(6));
  EXPECT_TRUE(set.contains(7));
  EXPECT_FALSE(set.contains(8));
}

TEST(CpuSet, Remove) {
  CpuSet set;
  set.addRange(0, 4);
  set.remove(2);
  EXPECT_EQ(set.count(), 3);
  EXPECT_TRUE(set.contains(0));
  EXPECT_TRUE(set.contains(1));
  EXPECT_FALSE(set.contains(2));
  EXPECT_TRUE(set.contains(3));
}

TEST(CpuSet, RemoveRange) {
  CpuSet set;
  set.addRange(0, 8);
  set.removeRange(2, 6);
  EXPECT_EQ(set.count(), 4);
  EXPECT_TRUE(set.contains(0));
  EXPECT_TRUE(set.contains(1));
  EXPECT_FALSE(set.contains(2));
  EXPECT_FALSE(set.contains(5));
  EXPECT_TRUE(set.contains(6));
  EXPECT_TRUE(set.contains(7));
}

TEST(CpuSet, Clear) {
  CpuSet set;
  set.addRange(0, 16);
  EXPECT_EQ(set.count(), 16);
  set.clear();
  EXPECT_EQ(set.count(), 0);
  EXPECT_FALSE(set.contains(0));
}

TEST(CpuSet, AddDuplicate) {
  CpuSet set;
  set.add(5);
  set.add(5);
  EXPECT_EQ(set.count(), 1);
  EXPECT_TRUE(set.contains(5));
}

TEST(CpuSet, RemoveAbsent) {
  CpuSet set;
  set.add(5);
  set.remove(3); // Not in set — should be harmless
  EXPECT_EQ(set.count(), 1);
  EXPECT_TRUE(set.contains(5));
}

TEST(CpuSet, WordBoundary) {
  // CPU 63 and 64 cross the uint64_t word boundary on non-Linux platforms.
  // On Linux cpu_set_t this also crosses an internal word boundary.
  CpuSet set;
  set.add(63);
  set.add(64);
  EXPECT_EQ(set.count(), 2);
  EXPECT_TRUE(set.contains(63));
  EXPECT_TRUE(set.contains(64));
  EXPECT_FALSE(set.contains(62));
  EXPECT_FALSE(set.contains(65));

  set.remove(63);
  EXPECT_EQ(set.count(), 1);
  EXPECT_FALSE(set.contains(63));
  EXPECT_TRUE(set.contains(64));
}

TEST(CpuSet, CountAfterMixedOperations) {
  CpuSet set;
  set.addRange(0, 10);
  EXPECT_EQ(set.count(), 10);
  set.removeRange(3, 7);
  EXPECT_EQ(set.count(), 6);
  set.add(5);
  EXPECT_EQ(set.count(), 7);
  set.add(5); // Duplicate
  EXPECT_EQ(set.count(), 7);
  set.remove(0);
  EXPECT_EQ(set.count(), 6);
}

// =============================================================================
// Out-of-Range CPU ID Tests
// =============================================================================

TEST(CpuSet, OutOfRangeCpuIdsAreIgnored) {
  CpuSet set;
  set.add(-1);
  EXPECT_EQ(set.count(), 0);
  EXPECT_FALSE(set.contains(-1));

  set.add(1024);
  EXPECT_EQ(set.count(), 0);
  EXPECT_FALSE(set.contains(1024));

  set.add(99999);
  EXPECT_EQ(set.count(), 0);

  set.addRange(-5, 3);
  EXPECT_EQ(set.count(), 3);
  for (int32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(set.contains(i));
  }

  set.remove(-1);
  set.remove(1024);
  EXPECT_EQ(set.count(), 3);

  set.clear();

  set.addRange(1020, 1030);
  EXPECT_EQ(set.count(), 4);
  for (int32_t i = 1020; i < 1024; ++i) {
    EXPECT_TRUE(set.contains(i));
  }
  EXPECT_FALSE(set.contains(1024));
}

// =============================================================================
// availableCount Tests
// =============================================================================

TEST(CpuSet, AvailableCountIsPositive) {
  int32_t available = CpuSet::availableCount();
  EXPECT_GE(available, 1);
  EXPECT_LE(available, CpuSet::all().count());
}

// =============================================================================
// Topology Query Tests
// =============================================================================

TEST(CpuSet, TotalNumaNodesIsPositive) {
  EXPECT_GE(CpuSet::totalNumaNodes(), 1);
}

TEST(CpuSet, NodeSetsAreNonEmpty) {
  int32_t numNodes = CpuSet::totalNumaNodes();
  for (int32_t i = 0; i < numNodes; ++i) {
    const auto& nodeSet = CpuSet::node(i);
    EXPECT_GT(nodeSet.count(), 0) << "NUMA node " << i << " has no CPUs";
  }
}

TEST(CpuSet, AllSetCoversAllNodeSets) {
  const auto& allSet = CpuSet::all();
  int32_t totalCpus = 0;
  int32_t numNodes = CpuSet::totalNumaNodes();
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());

  for (int32_t n = 0; n < numNodes; ++n) {
    const auto& nodeSet = CpuSet::node(n);
    for (int32_t i = 0; i < maxCpu; ++i) {
      if (nodeSet.contains(i)) {
        EXPECT_TRUE(allSet.contains(i)) << "CPU " << i << " in node " << n << " but not in all()";
        ++totalCpus;
      }
    }
  }
  EXPECT_EQ(allSet.count(), totalCpus);
}

TEST(CpuSet, CurrentHardwareThreadIsValid) {
  int32_t cpu = CpuSet::currentHardwareThread();
#if defined(__linux__) || defined(_WIN32) || (defined(__FreeBSD__) && __FreeBSD_version >= 1301000)
  // Linux (sched_getcpu), Windows (GetCurrentProcessorNumberEx), and FreeBSD
  // (sched_getcpu, 13.1+) report a valid hardware thread that must be a member
  // of the full CPU set.
  EXPECT_GE(cpu, 0);
  EXPECT_TRUE(CpuSet::all().contains(cpu));
#elif defined(__APPLE__)
  // macOS may return a valid CPU via _os_cpu_number, or -1 if unavailable
  EXPECT_GE(cpu, -1);
  if (cpu >= 0) {
    EXPECT_TRUE(CpuSet::all().contains(cpu));
  }
#else
  // On unsupported platforms, currentHardwareThread() returns -1
  EXPECT_EQ(cpu, -1);
#endif
}

// =============================================================================
// Cache Topology Tests
// =============================================================================

TEST(CpuSet, L2GroupsAreNonEmpty) {
  const auto& l2Groups = CpuSet::l2CacheGroups();
#if defined(__linux__)
  if (!sysfsHasCacheTopology()) {
    GTEST_SKIP() << "Kernel exposes no cache topology under /sys/devices/system/cpu";
  }
#endif
#if defined(__linux__) || defined(_WIN32)
  EXPECT_GT(l2Groups.size(), 0u) << "No L2 cache groups detected";

  // Each group should have at least 1 CPU
  for (const auto& group : l2Groups) {
    EXPECT_GT(group.cpus.size(), 0u) << "L2 group with cacheId=" << group.cacheId << " has no CPUs";
  }
#elif defined(__FreeBSD__)
  if (l2Groups.empty()) {
    GTEST_SKIP() << "No L2 cache groups detected on this FreeBSD machine";
  }
  for (const auto& group : l2Groups) {
    EXPECT_GT(group.cpus.size(), 0u) << "L2 group with cacheId=" << group.cacheId << " has no CPUs";
  }
#else
  // On unsupported platforms (e.g. macOS), l2CacheGroups() returns empty
  EXPECT_TRUE(l2Groups.empty());
#endif
}

TEST(CpuSet, L3GroupsAreNonEmpty) {
  const auto& l3Groups = CpuSet::l3CacheGroups();
#if defined(__linux__)
  if (!sysfsHasCacheTopology()) {
    GTEST_SKIP() << "Kernel exposes no cache topology under /sys/devices/system/cpu";
  }
#endif
#if defined(__linux__) || defined(_WIN32)
  EXPECT_GT(l3Groups.size(), 0u) << "No L3 cache groups detected";

  for (const auto& group : l3Groups) {
    EXPECT_GT(group.cpus.size(), 0u) << "L3 group with cacheId=" << group.cacheId << " has no CPUs";
  }
#elif defined(__FreeBSD__)
  if (l3Groups.empty()) {
    GTEST_SKIP() << "No L3 cache groups detected on this FreeBSD machine";
  }
  for (const auto& group : l3Groups) {
    EXPECT_GT(group.cpus.size(), 0u) << "L3 group with cacheId=" << group.cacheId << " has no CPUs";
  }
#else
  // On unsupported platforms (e.g. macOS), l3CacheGroups() returns empty
  EXPECT_TRUE(l3Groups.empty());
#endif
}

TEST(CpuSet, L2GroupsCoverAllCpus) {
  const auto& l2Groups = CpuSet::l2CacheGroups();
  if (l2Groups.empty()) {
    GTEST_SKIP() << "No L2 cache groups (unsupported platform)";
  }

  std::set<int32_t> coveredCpus;
  for (const auto& group : l2Groups) {
    for (int32_t cpu : group.cpus) {
      EXPECT_TRUE(coveredCpus.insert(cpu).second)
          << "CPU " << cpu << " appears in multiple L2 groups";
    }
  }

  // All online CPUs should be covered
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  const auto& allSet = CpuSet::all();
  for (int32_t i = 0; i < maxCpu; ++i) {
    if (allSet.contains(i)) {
      EXPECT_TRUE(coveredCpus.count(i) > 0) << "CPU " << i << " not covered by any L2 group";
    }
  }
}

TEST(CpuSet, L3GroupsCoverAllCpus) {
  const auto& l3Groups = CpuSet::l3CacheGroups();
  if (l3Groups.empty()) {
    GTEST_SKIP() << "No L3 cache groups (unsupported platform)";
  }

  std::set<int32_t> coveredCpus;
  for (const auto& group : l3Groups) {
    for (int32_t cpu : group.cpus) {
      EXPECT_TRUE(coveredCpus.insert(cpu).second)
          << "CPU " << cpu << " appears in multiple L3 groups";
    }
  }

  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  const auto& allSet = CpuSet::all();
  for (int32_t i = 0; i < maxCpu; ++i) {
    if (allSet.contains(i)) {
      EXPECT_TRUE(coveredCpus.count(i) > 0) << "CPU " << i << " not covered by any L3 group";
    }
  }
}

TEST(CpuSet, L2GroupsAreSubsetsOfL3Groups) {
  const auto& l2Groups = CpuSet::l2CacheGroups();
  const auto& l3Groups = CpuSet::l3CacheGroups();
  if (l2Groups.empty() || l3Groups.empty()) {
    GTEST_SKIP() << "No L2 or L3 cache groups (unsupported platform)";
  }

  // Build L3 membership map: CPU -> L3 group index. Size by the actual max CPU
  // ID across both group sets — CPU IDs can be sparse or exceed the online
  // count, so hardware_concurrency() alone can under-size the vector.
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  for (const auto& g : l3Groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  for (const auto& g : l2Groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  std::vector<int32_t> cpuToL3(static_cast<size_t>(maxCpu), -1);
  for (size_t g = 0; g < l3Groups.size(); ++g) {
    for (int32_t cpu : l3Groups[g].cpus) {
      cpuToL3[static_cast<size_t>(cpu)] = static_cast<int32_t>(g);
    }
  }

  // Each L2 group's CPUs should all belong to the same L3 group
  for (const auto& l2 : l2Groups) {
    if (l2.cpus.empty()) {
      continue;
    }
    int32_t expectedL3 = cpuToL3[static_cast<size_t>(l2.cpus[0])];
    ASSERT_GE(expectedL3, 0) << "CPU " << l2.cpus[0] << " has no L3 group";
    for (int32_t cpu : l2.cpus) {
      EXPECT_EQ(cpuToL3[static_cast<size_t>(cpu)], expectedL3)
          << "L2 group (cacheId=" << l2.cacheId << ") spans multiple L3 groups: "
          << "CPU " << l2.cpus[0] << " is in L3 group " << expectedL3 << " but CPU " << cpu
          << " is in L3 group " << cpuToL3[static_cast<size_t>(cpu)];
    }
  }
}

TEST(CpuSet, L2GroupsSortedByFirstCpu) {
  const auto& l2Groups = CpuSet::l2CacheGroups();
  for (size_t i = 1; i < l2Groups.size(); ++i) {
    EXPECT_LT(l2Groups[i - 1].cpus.front(), l2Groups[i].cpus.front())
        << "L2 groups not sorted by first CPU ID";
  }
}

// =============================================================================
// Thread Binding Tests
// =============================================================================

// Fixture that restores full CPU affinity after each test, even if an
// assertion aborts early (ASSERT_*, unhandled exception, etc.).
class CpuSetBindTest : public ::testing::Test {
 protected:
  void TearDown() override {
    CpuSet::all().bindCurrentThread();
  }
};

// Best-effort probe for whether this environment can actually pin the current
// thread. There is no pure permission query for CPU affinity on Linux/FreeBSD --
// the only way to know is to attempt a bind and observe the result -- so this
// binds to the full (non-restrictive) set and reports success. Returns false on
// platforms without pinning (macOS) or when a sandbox denies the syscall (EPERM).
// CpuSetBindTest::TearDown restores full affinity regardless.
namespace {
bool bindingPermitted() {
  if (CpuSet::currentHardwareThread() < 0) {
    return false;
  }
  return CpuSet::all().bindCurrentThread();
}
} // namespace

TEST_F(CpuSetBindTest, BindCurrentThread) {
  const int32_t currentCpu = CpuSet::currentHardwareThread();

  if (!bindingPermitted()) {
    // Unsupported platform or sandboxed environment: exercise the graceful
    // degradation path (a clean false, no crash) instead of skipping, so CI
    // still runs the test.
    CpuSet set;
    set.add(currentCpu >= 0 ? currentCpu : 0);
    EXPECT_FALSE(set.bindCurrentThread())
        << "bindCurrentThread() should fail cleanly when binding is not permitted";
    return;
  }

  // Permitted: binding to the current CPU must keep us on it.
  CpuSet set;
  set.add(currentCpu);
  ASSERT_TRUE(set.bindCurrentThread());
  EXPECT_EQ(CpuSet::currentHardwareThread(), currentCpu);
}

TEST_F(CpuSetBindTest, BindToRange) {
  const auto& allSet = CpuSet::all();

  // Find the first two online CPUs.
  int32_t cpu0 = -1, cpu1 = -1;
  const int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  for (int32_t i = 0; i < maxCpu && cpu1 < 0; ++i) {
    if (allSet.contains(i)) {
      if (cpu0 < 0) {
        cpu0 = i;
      } else {
        cpu1 = i;
      }
    }
  }

  if (!bindingPermitted()) {
    // Unsupported platform or sandboxed environment: verify graceful failure
    // instead of skipping.
    CpuSet set;
    set.add(cpu0 >= 0 ? cpu0 : 0);
    if (cpu1 >= 0) {
      set.add(cpu1);
    }
    EXPECT_FALSE(set.bindCurrentThread())
        << "bindCurrentThread() should fail cleanly when binding is not permitted";
    return;
  }
  if (cpu1 < 0) {
    GTEST_SKIP() << "Need at least 2 online CPUs to test range binding";
  }

  CpuSet set;
  set.add(cpu0);
  set.add(cpu1);
  ASSERT_TRUE(set.bindCurrentThread());

  // After binding, we should be on one of the two CPUs.
  const int32_t current = CpuSet::currentHardwareThread();
  EXPECT_TRUE(current == cpu0 || current == cpu1)
      << "After binding to {" << cpu0 << ", " << cpu1 << "}, running on CPU " << current;
}

// =============================================================================
// Thread Group Building Tests
// =============================================================================

TEST(CpuSet, BuildThreadGroupsProducesGroups) {
  auto groups = CpuSet::buildThreadGroups();
  EXPECT_GT(groups.size(), 0u);

  // Every group should have at least 1 CPU
  for (const auto& group : groups) {
    EXPECT_GT(group.cpus.size(), 0u);
  }
}

TEST(CpuSet, BuildThreadGroupsRespectsMaxSize) {
  auto groups = CpuSet::buildThreadGroups(dispenso::kDefaultMaxGroupSize);

  // L2 atom clamping may make some groups larger than maxGroupSize on
  // architectures with large SMT (e.g. Power SMT8 with maxGroupSize=4).
  // But on this machine (SMT2), no group should exceed the limit.
  const auto& l2Groups = CpuSet::l2CacheGroups();
  int32_t maxL2 = 0;
  for (const auto& l2 : l2Groups) {
    int32_t sz = static_cast<int32_t>(l2.cpus.size());
    if (sz > maxL2) {
      maxL2 = sz;
    }
  }
  int32_t effectiveMax = std::max(static_cast<int32_t>(dispenso::kDefaultMaxGroupSize), maxL2);

  for (const auto& group : groups) {
    EXPECT_LE(static_cast<int32_t>(group.cpus.size()), effectiveMax)
        << "Group starting at CPU " << group.cpus.front() << " exceeds max size";
  }
}

TEST(CpuSet, BuildThreadGroupsCoversAllCpus) {
  auto groups = CpuSet::buildThreadGroups();
  const auto& allSet = CpuSet::all();

  // CPU IDs can be sparse and exceed hardware_concurrency() on some platforms
  // (e.g. Windows processor groups), so derive the scan limit from both the
  // group data and hardware_concurrency() to cover all possible IDs.
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  std::set<int32_t> covered;
  for (const auto& group : groups) {
    for (int32_t cpu : group.cpus) {
      EXPECT_TRUE(covered.insert(cpu).second)
          << "CPU " << cpu << " appears in multiple thread groups";
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }

  // Also check beyond the groups — allSet may contain CPUs that
  // buildThreadGroups failed to include, which is the bug we want to catch.
  for (int32_t i = 0; i < maxCpu; ++i) {
    if (allSet.contains(i)) {
      EXPECT_TRUE(covered.count(i) > 0) << "CPU " << i << " not in any thread group";
    }
  }
}

TEST(CpuSet, BuildThreadGroupsNeverSplitsL2) {
  const auto& l2Groups = CpuSet::l2CacheGroups();
  if (l2Groups.empty()) {
    GTEST_SKIP() << "No L2 cache groups (unsupported platform)";
  }

  auto groups = CpuSet::buildThreadGroups();

  // Build group membership: CPU -> thread group index. Size by the actual max
  // CPU ID across both group sets — CPU IDs can be sparse or exceed the online
  // count, so hardware_concurrency() alone can under-size the vector.
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  for (const auto& g : groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  for (const auto& g : l2Groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  std::vector<int32_t> cpuToGroup(static_cast<size_t>(maxCpu), -1);
  for (size_t g = 0; g < groups.size(); ++g) {
    for (int32_t cpu : groups[g].cpus) {
      cpuToGroup[static_cast<size_t>(cpu)] = static_cast<int32_t>(g);
    }
  }

  // Every L2 group's CPUs must be in the same thread group
  for (const auto& l2 : l2Groups) {
    if (l2.cpus.size() <= 1) {
      continue;
    }
    int32_t expectedGroup = cpuToGroup[static_cast<size_t>(l2.cpus[0])];
    for (int32_t cpu : l2.cpus) {
      EXPECT_EQ(cpuToGroup[static_cast<size_t>(cpu)], expectedGroup)
          << "L2 group (cacheId=" << l2.cacheId << ") split across thread groups: "
          << "CPU " << l2.cpus[0] << " in group " << expectedGroup << ", CPU " << cpu
          << " in group " << cpuToGroup[static_cast<size_t>(cpu)];
    }
  }
}

TEST(CpuSet, BuildThreadGroupsNeverCrossesL3) {
  const auto& l3Groups = CpuSet::l3CacheGroups();
  if (l3Groups.empty()) {
    GTEST_SKIP() << "No L3 cache groups (unsupported platform)";
  }

  auto groups = CpuSet::buildThreadGroups();

  // Build L3 membership. Size by the actual max CPU ID across both group sets —
  // CPU IDs can be sparse or exceed the online count, so hardware_concurrency()
  // alone can under-size the vector.
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  for (const auto& g : l3Groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  for (const auto& g : groups) {
    for (int32_t cpu : g.cpus) {
      maxCpu = std::max(maxCpu, cpu + 1);
    }
  }
  std::vector<int32_t> cpuToL3(static_cast<size_t>(maxCpu), -1);
  for (size_t g = 0; g < l3Groups.size(); ++g) {
    for (int32_t cpu : l3Groups[g].cpus) {
      cpuToL3[static_cast<size_t>(cpu)] = static_cast<int32_t>(g);
    }
  }

  // Every thread group's CPUs must be in the same L3 group
  for (const auto& group : groups) {
    if (group.cpus.empty()) {
      continue;
    }
    int32_t expectedL3 = cpuToL3[static_cast<size_t>(group.cpus[0])];
    for (int32_t cpu : group.cpus) {
      EXPECT_EQ(cpuToL3[static_cast<size_t>(cpu)], expectedL3)
          << "Thread group crosses L3 boundary: CPU " << group.cpus[0] << " in L3 " << expectedL3
          << ", CPU " << cpu << " in L3 " << cpuToL3[static_cast<size_t>(cpu)];
    }
  }
}

TEST(CpuSet, BuildThreadGroupsSortedByFirstCpu) {
  auto groups = CpuSet::buildThreadGroups();
  for (size_t i = 1; i < groups.size(); ++i) {
    EXPECT_LT(groups[i - 1].cpus.front(), groups[i].cpus.front())
        << "Thread groups not sorted by first CPU ID";
  }
}

TEST(CpuSet, BuildThreadGroupsAffinityMaskMatchesCpus) {
  auto groups = CpuSet::buildThreadGroups();
  for (const auto& group : groups) {
    EXPECT_EQ(group.affinityMask.count(), static_cast<int32_t>(group.cpus.size()));
    for (int32_t cpu : group.cpus) {
      EXPECT_TRUE(group.affinityMask.contains(cpu))
          << "CPU " << cpu << " in group but not in affinity mask";
    }
  }
}

TEST(CpuSet, BuildThreadGroupsCustomSize) {
  // Build with a smaller group size
  auto groups8 = CpuSet::buildThreadGroups(8);
  auto groups16 = CpuSet::buildThreadGroups(16);

  // Smaller max should produce at least as many groups
  EXPECT_GE(groups8.size(), groups16.size());

  // Both should cover all CPUs
  std::set<int32_t> covered8, covered16;
  for (const auto& g : groups8) {
    covered8.insert(g.cpus.begin(), g.cpus.end());
  }
  for (const auto& g : groups16) {
    covered16.insert(g.cpus.begin(), g.cpus.end());
  }
  EXPECT_EQ(covered8, covered16);
}

TEST(CpuSet, BuildThreadGroupsClampsBelowL2Atom) {
  // maxGroupSize=1 should be clamped up to the L2 atom size
  auto groups = CpuSet::buildThreadGroups(1);
  EXPECT_GT(groups.size(), 0u);

  // On any machine with SMT, L2 atoms are >= 2, so groups should be >= 2.
  // Without topology (fallback path), groups of 1 are valid.
  const auto& l2Groups = CpuSet::l2CacheGroups();
  if (!l2Groups.empty()) {
    // Production clamps maxGroupSize *up* to the largest L2 atom so it never
    // splits an atom — but a group flushed at an L3 boundary can still be
    // smaller than the largest atom. The guaranteed invariant is that every
    // group is built by packing whole L2 atoms, so each group contains at least
    // one complete atom and is therefore no smaller than the *smallest* L2 atom.
    int32_t minL2Atom = static_cast<int32_t>(l2Groups.front().cpus.size());
    for (const auto& l2 : l2Groups) {
      minL2Atom = std::min(minL2Atom, static_cast<int32_t>(l2.cpus.size()));
    }
    for (const auto& group : groups) {
      EXPECT_GE(static_cast<int32_t>(group.cpus.size()), minL2Atom)
          << "Group smaller than the smallest L2 atom (every group holds >= 1 full L2 atom)";
    }
  }

  // Coverage should still be complete
  std::set<int32_t> covered;
  for (const auto& g : groups) {
    covered.insert(g.cpus.begin(), g.cpus.end());
  }
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  const auto& allSet = CpuSet::all();
  for (int32_t i = 0; i < maxCpu; ++i) {
    if (allSet.contains(i)) {
      EXPECT_TRUE(covered.count(i) > 0);
    }
  }
}

TEST(CpuSet, BuildThreadGroupsLargeMaxSize) {
  // maxGroupSize larger than total CPUs — should still work, fewer groups
  auto groups = CpuSet::buildThreadGroups(4096);
  EXPECT_GT(groups.size(), 0u);

  // Should produce fewer groups than default
  auto defaultGroups = CpuSet::buildThreadGroups();
  EXPECT_LE(groups.size(), defaultGroups.size());

  // Coverage must be complete
  std::set<int32_t> covered;
  for (const auto& g : groups) {
    covered.insert(g.cpus.begin(), g.cpus.end());
  }
  std::set<int32_t> coveredDefault;
  for (const auto& g : defaultGroups) {
    coveredDefault.insert(g.cpus.begin(), g.cpus.end());
  }
  EXPECT_EQ(covered, coveredDefault);
}

TEST(CpuSet, BuildThreadGroupsCpusSortedWithinGroup) {
  auto groups = CpuSet::buildThreadGroups();
  for (const auto& group : groups) {
    for (size_t i = 1; i < group.cpus.size(); ++i) {
      EXPECT_LT(group.cpus[i - 1], group.cpus[i]) << "CPUs within group not sorted";
    }
  }
}
