/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file cpu_set.h
 * @ingroup group_util
 * A portable CPU affinity and NUMA topology facility.
 *
 * Provides CPU set manipulation, thread-to-core binding, and NUMA/cache topology detection.
 * The primary use case is cache-aware thread group assignment for fork-join scheduling.
 *
 * Platform support:
 * - **Linux**: Full support (binding via pthread_setaffinity_np, topology via sysfs)
 * - **FreeBSD**: Binding supported (cpuset_setaffinity), topology detection deferred
 * - **macOS**: Topology query only (binding is not supported by the OS)
 * - **Windows**: Full support (binding via SetThreadGroupAffinity, topology via
 *   GetLogicalProcessorInformationEx)
 *
 * @see docs/design/three_tier_scheduling.md for the design context.
 **/

#pragma once

#include <cstdint>
#include <thread>
#include <vector>

#include <dispenso/platform.h>

#if defined(__linux__) || defined(__FreeBSD__)
#define DISPENSO_CPUSET_LINUXY
#endif // linuxy

#if defined(DISPENSO_CPUSET_LINUXY)
#include <pthread.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <sys/param.h>
#include <sys/cpuset.h>
#include <pthread_np.h>
typedef cpuset_t cpu_set_t;
#endif
#elif defined(_WIN32)
#define DISPENSO_CPUSET_WINDOWS
#endif // supported os

// Note: Windows uses SetThreadGroupAffinity for binding and
// GetLogicalProcessorInformationEx for NUMA/cache topology, behind the same public API.
//
// macOS does not support explicit CPU pinning. bindCurrentThread() is a no-op and
// returns false. Topology queries return a single-node fallback.
//
// FreeBSD binding support is present (cpuset_setaffinity uses a similar interface to
// Linux), but topology detection requires test hardware and is deferred.

/**
 * @brief Compile-time override for the default maximum threads per scheduling group.
 *
 * Define DISPENSO_MAX_GROUP_SIZE before including this header (or via -D) to change
 * the default. This only affects the default; ThreadPool can override at construction.
 */
#ifndef DISPENSO_MAX_GROUP_SIZE
#define DISPENSO_MAX_GROUP_SIZE 16
#endif

namespace dispenso {

/**
 * @brief Default maximum number of threads per scheduling group.
 *
 * 16 matches the AMD CCD boundary (8 cores x 2 SMT = 16 threads) and provides
 * a good balance between wake granularity and scheduling overhead across
 * architectures. See docs/design/three_tier_scheduling.md for rationale.
 *
 * This is a runtime default, not a compile-time constraint. ThreadPool accepts
 * maxGroupSize as a constructor parameter. Override at compile time via
 * DISPENSO_MAX_GROUP_SIZE.
 */
constexpr int32_t kDefaultMaxGroupSize = DISPENSO_MAX_GROUP_SIZE;

/**
 * @brief Describes a group of CPUs that share a cache level.
 *
 * Used by the topology detection API to report L2 and L3 cache sharing groups.
 */
struct CacheGroup {
  std::vector<int32_t> cpus; ///< CPU IDs in this group
  int32_t cacheId; ///< Unique cache instance ID from sysfs
};

// Forward declaration; full definition follows CpuSet (ThreadGroup has a CpuSet member).
struct ThreadGroup;

/**
 * @class CpuSet
 * @brief A set of CPU IDs for affinity manipulation and topology queries.
 *
 * CpuSet provides:
 * - Set manipulation: add, remove, query individual CPUs or ranges
 * - Thread binding: pin the calling thread to the CPUs in the set
 * - Topology queries: NUMA node enumeration, cache sharing groups
 *
 * ## Thread Safety
 *
 * CpuSet instances are NOT thread-safe. Static query methods (totalNumaNodes(),
 * node(), all(), etc.) are safe to call from any thread after static initialization.
 *
 * ## Capacity
 *
 * A CpuSet represents CPU IDs in [0, 1024): the Linux/FreeBSD backing is a fixed
 * cpu_set_t (CPU_SETSIZE, typically 1024) and the portable Windows/macOS backing is
 * a 1024-bit array. CPU IDs at or beyond that bound are silently ignored by add() /
 * addRange() / contains() (never undefined behavior), so the library degrades
 * gracefully on machines with >= 1024 logical CPUs or sparse IDs that high: such
 * CPUs simply are not bound or grouped. CpuSets are allocated only at startup
 * (topology singletons, per-ThreadGroup masks), so raising the limit later
 * (dynamic cpu_set_t on Linux via CPU_ALLOC, a wider bitset elsewhere) would be
 * cheap; it is tracked in the roadmap rather than done now since no current
 * hardware approaches it.
 *
 * ## Example
 *
 * @code
 * // Pin current thread to CPUs 0-7
 * dispenso::CpuSet set;
 * set.addRange(0, 8);
 * set.bindCurrentThread();
 *
 * // Query NUMA topology
 * int nodes = dispenso::CpuSet::totalNumaNodes();
 * for (int i = 0; i < nodes; ++i) {
 *     const auto& nodeSet = dispenso::CpuSet::node(i);
 *     // nodeSet contains CPUs on NUMA node i
 * }
 *
 * // Query L3 cache groups for thread group assignment
 * auto l3Groups = dispenso::CpuSet::l3CacheGroups();
 * // Each group contains CPUs sharing an L3 cache (CCD on AMD, tile on Intel)
 * @endcode
 */
class CpuSet {
 public:
  /**
   * @brief Constructs an empty CPU set.
   */
  DISPENSO_DLL_ACCESS CpuSet();

  /**
   * @brief Removes all CPUs from the set.
   */
  DISPENSO_DLL_ACCESS void clear();

  /**
   * @brief Adds a single CPU to the set.
   * @param hardwareThread The CPU ID to add.
   */
  DISPENSO_DLL_ACCESS void add(int32_t hardwareThread);

  /**
   * @brief Adds a range of CPUs to the set.
   * @param start First CPU ID (inclusive).
   * @param end One past the last CPU ID (exclusive).
   */
  DISPENSO_DLL_ACCESS void addRange(int32_t start, int32_t end);

  /**
   * @brief Removes a single CPU from the set.
   * @param hardwareThread The CPU ID to remove.
   */
  DISPENSO_DLL_ACCESS void remove(int32_t hardwareThread);

  /**
   * @brief Removes a range of CPUs from the set.
   * @param start First CPU ID (inclusive).
   * @param end One past the last CPU ID (exclusive).
   */
  DISPENSO_DLL_ACCESS void removeRange(int32_t start, int32_t end);

  /**
   * @brief Tests whether a CPU is in the set.
   * @param hardwareThread The CPU ID to test.
   * @return true if the CPU is in the set.
   */
  DISPENSO_DLL_ACCESS bool contains(int32_t hardwareThread) const;

  /**
   * @brief Returns the number of CPUs in the set.
   */
  DISPENSO_DLL_ACCESS int32_t count() const;

  /**
   * @brief Binds the calling thread to the CPUs in this set.
   *
   * On Linux/FreeBSD, calls pthread_setaffinity_np. On unsupported platforms
   * (macOS, Windows without GROUP_AFFINITY support), returns false.
   *
   * @return true if binding succeeded, false on failure or unsupported platform.
   */
  DISPENSO_DLL_ACCESS bool bindCurrentThread() const;

  /**
   * @brief Returns the total number of NUMA nodes detected.
   *
   * Always returns at least 1 (single-node fallback when detection fails
   * or the platform has no NUMA).
   */
  DISPENSO_DLL_ACCESS static int32_t totalNumaNodes();

  /**
   * @brief Returns the CPU ID of the calling thread's current core.
   *
   * On Linux, uses the vDSO-accelerated getcpu(). On unsupported platforms,
   * returns -1.
   *
   * @note The result is instantaneous and may be stale by the time it is used
   *       (the OS may migrate the thread). Useful as a hint for scheduling
   *       decisions, not as a hard guarantee.
   */
  DISPENSO_DLL_ACCESS static int32_t currentHardwareThread();

  /**
   * @brief Approximate CPU ID for the calling thread, refreshed periodically.
   *
   * Caches the result of currentHardwareThread() in a thread-local and
   * re-queries every kRefreshPeriod calls. Designed for hot paths that
   * want locality hints without paying the ~15ns vDSO cost per call: the
   * cached path is a single TLS read + increment + branch (~1-2ns).
   *
   * Trade-off: between refreshes the returned CPU is stale if the OS has
   * migrated the thread. With kRefreshPeriod=32 and a typical OS time-slice
   * of 1-10ms, staleness is bounded to a small fraction of a slice.
   *
   * @return A possibly-stale CPU ID, or -1 if the platform doesn't support
   *         CPU queries.
   */
  static int32_t currentHardwareThreadApprox() {
    static constexpr uint32_t kRefreshPeriod = 32;
    static DISPENSO_THREAD_LOCAL int32_t cachedCpu = -1;
    static DISPENSO_THREAD_LOCAL uint32_t counter = 0;
    if ((counter++ & (kRefreshPeriod - 1)) == 0) {
      cachedCpu = currentHardwareThread();
    }
    return cachedCpu;
  }

  /**
   * @brief Returns the CPU set for a specific NUMA node.
   * @param numaNode The NUMA node index (0 to totalNumaNodes()-1).
   * @return A reference to the cached CpuSet for that node.
   */
  DISPENSO_DLL_ACCESS static const CpuSet& node(int32_t numaNode);

  /**
   * @brief Returns a CPU set containing all online CPUs.
   *
   * On Linux, derived from the union of all NUMA node sets (which correctly
   * handles non-contiguous CPU ID ranges). Falls back to
   * [0, hardware_concurrency()) on unsupported platforms.
   */
  DISPENSO_DLL_ACCESS static const CpuSet& all();

  /**
   * @brief Returns the number of hardware threads available to this process.
   *
   * On Linux, queries the process CPU affinity mask (respects taskset/cgroup).
   * On Windows single-group systems, queries the process affinity mask.
   * On Windows multi-group systems, sums active processors per group (does not
   * reflect process-level restrictions).
   * On other platforms, falls back to std::thread::hardware_concurrency().
   */
  DISPENSO_DLL_ACCESS static int32_t availableCount();

  /**
   * @brief Returns the L2 cache sharing groups.
   *
   * Each group contains the CPU IDs that share an L2 cache instance. On
   * x86 with SMT, each group is typically an SMT sibling pair. On Power
   * (SMT8), each group is 8 threads.
   *
   * The groups are sorted by their first CPU ID.
   *
   * @return A reference to the cached vector of L2 groups. Empty if
   *         detection is not supported on the current platform.
   */
  DISPENSO_DLL_ACCESS static const std::vector<CacheGroup>& l2CacheGroups();

  /**
   * @brief Returns the L3 cache sharing groups.
   *
   * Each group contains the CPU IDs that share an L3 cache instance. On
   * AMD EPYC, each group corresponds to a CCD (Core Complex Die). On Intel,
   * each group corresponds to a tile or SNC cluster.
   *
   * The groups are sorted by their first CPU ID.
   *
   * @return A reference to the cached vector of L3 groups. Empty if
   *         detection is not supported on the current platform.
   */
  DISPENSO_DLL_ACCESS static const std::vector<CacheGroup>& l3CacheGroups();

  /**
   * @brief Builds scheduling thread groups from cache topology.
   *
   * Groups are built bottom-up:
   * 1. L2 cache groups are the atoms (SMT siblings are never split)
   * 2. L2 atoms are packed into groups within L3 boundaries
   * 3. Each group has at most @p maxGroupSize CPUs
   * 4. Groups are L3-coherent (never cross L3 cache boundaries)
   *
   * On systems without cache topology detection (macOS, Windows currently),
   * falls back to contiguous chunking of all online CPUs.
   *
   * @param maxGroupSize Maximum CPUs per group. Defaults to DISPENSO_MAX_GROUP_SIZE.
   *                     Clamped to at least the largest L2 atom size (e.g. 8 on
   *                     Power10 with SMT8) to avoid splitting SMT siblings.
   * @return Vector of ThreadGroups, sorted by first CPU ID in each group.
   *
   * ## Example
   * @code
   * auto groups = dispenso::CpuSet::buildThreadGroups();
   * // On a 192-thread dual-socket EPYC (12 CCDs x 16 threads):
   * // groups.size() == 12, each with 16 CPUs matching a CCD
   *
   * // Custom group size for finer-grained waking:
   * auto smallGroups = dispenso::CpuSet::buildThreadGroups(8);
   * // groups.size() == 24, each with 8 CPUs (4 cores)
   * @endcode
   */
  DISPENSO_DLL_ACCESS static std::vector<ThreadGroup> buildThreadGroups(
      int32_t maxGroupSize = kDefaultMaxGroupSize);

 private:
#if defined(DISPENSO_CPUSET_LINUXY)
  cpu_set_t set_;
#else
  // Portable bitset supporting up to kMaxCpus logical processors.
  // 1024 matches Linux cpu_set_t and covers all practical configurations.
  // Windows Server 2025 supports up to 12,288 (192 groups x 64) but
  // no known hardware approaches that limit. Can be increased later;
  // dispenso does not guarantee ABI stability.
  static constexpr int32_t kMaxCpus = 1024;
  static constexpr int32_t kBitsPerWord = 64;
  static constexpr int32_t kNumWords = kMaxCpus / kBitsPerWord;
  uint64_t words_[kNumWords];
#endif
};

/**
 * @brief A scheduling group of CPUs for fork-join thread pool assignment.
 *
 * Thread groups are the fundamental scheduling and wake unit. Each group:
 * - Contains at most DISPENSO_MAX_GROUP_SIZE threads (configurable)
 * - Is L3-coherent (all CPUs share an L3 cache)
 * - Never splits an L2 cache group (SMT siblings stay together)
 * - Stays within L3 cache boundaries when possible
 *
 * Groups are built bottom-up from cache topology: L2 atoms are packed into
 * groups within L3 boundaries. On systems without cache topology detection,
 * CPUs are chunked contiguously by maxGroupSize.
 *
 * @see CpuSet::buildThreadGroups()
 */
struct ThreadGroup {
  std::vector<int32_t> cpus; ///< CPU IDs in this group (sorted)
  CpuSet affinityMask; ///< Pre-built CpuSet for binding threads in this group
};

namespace detail {
/**
 * @brief Parses a Linux-style CPU list string (e.g. "0-3,8-11") into a CpuSet.
 *
 * This is an internal helper exposed for testing.
 */
DISPENSO_DLL_ACCESS CpuSet parseLinuxCpuList(const char* input);
} // namespace detail

} // namespace dispenso
