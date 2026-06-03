/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/cpu_set.h>

#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif // __linux__

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace dispenso {

namespace detail {

namespace {
constexpr long kMaxReasonableCpuId = 1 << 20;

int32_t parseIntClamped(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || v < 0 || v > kMaxReasonableCpuId) {
    return -1;
  }
  return static_cast<int32_t>(v);
}

void parseAndAddRange(char* buf, CpuSet& set) {
  if (*buf == '\0') {
    return;
  }
  char* sep = strchr(buf, '-');
  if (sep) {
    *sep = '\0';
    int32_t lo = parseIntClamped(buf);
    int32_t hi = parseIntClamped(sep + 1);
    if (lo >= 0 && hi >= 0) {
      set.addRange(lo, hi + 1);
    }
  } else {
    int32_t v = parseIntClamped(buf);
    if (v >= 0) {
      set.add(v);
    }
  }
}
} // namespace

CpuSet parseLinuxCpuList(const char* input) {
  // Copy because parsing mutates the buffer (NUL-terminates at commas).
  std::string copy(input);
  char* buf = &copy[0];
  CpuSet set;
  while (char* commaSep = strchr(buf, ',')) {
    *commaSep = '\0';
    parseAndAddRange(buf, set);
    buf = commaSep + 1;
  }
  parseAndAddRange(buf, set);
  return set;
}

} // namespace detail

// =============================================================================
// CpuSet core methods
// =============================================================================

CpuSet::CpuSet() {
  clear();
}

#if defined(DISPENSO_CPUSET_LINUXY)

void CpuSet::clear() {
  CPU_ZERO(&set_);
}

// cpu_set_t holds exactly CPU_SETSIZE (typically 1024) bits; CPU_SET/CPU_CLR/
// CPU_ISSET with an index outside [0, CPU_SETSIZE) is undefined behavior (it
// reads/writes past the fixed-size set member). CPU IDs can legitimately reach
// or exceed CPU_SETSIZE on very large machines (>=1024 logical CPUs) or with
// sparse/hot-plugged numbering, and sysfs parsing can surface such IDs. Rather
// than corrupt memory, we silently ignore out-of-range IDs: such CPUs are simply
// not representable in this set (graceful degradation — they won't be bound or
// grouped, but nothing is corrupted).
void CpuSet::add(int32_t hardwareThread) {
  if (hardwareThread < 0 || hardwareThread >= CPU_SETSIZE) {
    return;
  }
  CPU_SET(hardwareThread, &set_);
}

void CpuSet::addRange(int32_t start, int32_t end) {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int32_t>(CPU_SETSIZE));
  for (int32_t i = start; i < end; ++i) {
    CPU_SET(i, &set_);
  }
}

void CpuSet::remove(int32_t hardwareThread) {
  if (hardwareThread < 0 || hardwareThread >= CPU_SETSIZE) {
    return;
  }
  CPU_CLR(hardwareThread, &set_);
}

void CpuSet::removeRange(int32_t start, int32_t end) {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int32_t>(CPU_SETSIZE));
  for (int32_t i = start; i < end; ++i) {
    CPU_CLR(i, &set_);
  }
}

bool CpuSet::contains(int32_t hardwareThread) const {
  if (hardwareThread < 0 || hardwareThread >= CPU_SETSIZE) {
    return false;
  }
  return CPU_ISSET(hardwareThread, &set_);
}

int32_t CpuSet::count() const {
  return static_cast<int32_t>(CPU_COUNT(&set_));
}

bool CpuSet::bindCurrentThread() const {
#if defined(__ANDROID__)
  // Android NDK lacks pthread_setaffinity_np; sched_setaffinity with pid=0 binds the caller.
  return sched_setaffinity(0, sizeof(cpu_set_t), &set_) == 0;
#else
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set_) == 0;
#endif
}

int32_t CpuSet::currentHardwareThread() {
#if defined(__linux__)
  // sched_getcpu() uses the vDSO on modern kernels (~15 ns, no syscall).
  int cpu = sched_getcpu();
  return (cpu >= 0) ? static_cast<int32_t>(cpu) : -1;
#else
  return -1;
#endif
}

// =============================================================================
// NUMA topology detection (Linux)
// =============================================================================

namespace {

// Read a small file into a NUL-terminated buffer. Returns empty string on failure.
std::vector<char> readSmallFile(int dirFd, const char* path) {
  int fd = ::openat(dirFd, path, O_RDONLY);
  if (fd < 0) {
    return {};
  }

  constexpr ssize_t kChunkSize = 256;
  std::vector<char> buf(kChunkSize + 1);
  ssize_t totalRead = 0;

  while (true) {
    ssize_t n = ::read(fd, buf.data() + totalRead, kChunkSize);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      break;
    }
    totalRead += n;
    buf.resize(static_cast<size_t>(totalRead) + kChunkSize + 1);
  }
  ::close(fd);

  if (totalRead == 0) {
    return {};
  }

  // Trim trailing whitespace/newline
  while (totalRead > 0 &&
         (buf[static_cast<size_t>(totalRead) - 1] == '\n' ||
          buf[static_cast<size_t>(totalRead) - 1] == ' ')) {
    --totalRead;
  }
  buf[static_cast<size_t>(totalRead)] = '\0';
  buf.resize(static_cast<size_t>(totalRead) + 1);
  return buf;
}

// Parse a "nodeN" sysfs directory name, returning the node index, or -1.
int32_t parseNodeDirName(const char* name) {
  if (name[0] != 'n' || name[1] != 'o' || name[2] != 'd' || name[3] != 'e' || name[4] == '\0') {
    return -1;
  }
  return detail::parseIntClamped(name + 4);
}

const std::vector<CpuSet>& getNumaSets() {
  static const std::vector<CpuSet> numaSets = []() {
    std::vector<CpuSet> sets;
#if defined(__linux__)
    DIR* nodeDir = ::opendir("/sys/devices/system/node");
    if (nodeDir != nullptr) {
      int dirFd = ::dirfd(nodeDir);
      // Collect node indices via readdir to handle sparse NUMA IDs (e.g., after
      // hot-unplug or in virtualized environments).
      std::vector<int32_t> nodeIndices;
      struct dirent* entry;
      while ((entry = ::readdir(nodeDir)) != nullptr) {
        int32_t idx = parseNodeDirName(entry->d_name);
        if (idx >= 0) {
          nodeIndices.push_back(idx);
        }
      }
      std::sort(nodeIndices.begin(), nodeIndices.end());
      for (int32_t nodeIdx : nodeIndices) {
        char path[32];
        snprintf(path, sizeof(path), "node%d/cpulist", nodeIdx);
        auto buf = readSmallFile(dirFd, path);
        if (!buf.empty()) {
          sets.emplace_back(detail::parseLinuxCpuList(buf.data()));
        }
      }
      ::closedir(nodeDir);
    }
#endif // __linux__
    // Fallback: if no NUMA nodes were found (missing sysfs, container, etc.),
    // return a single node containing all CPUs so that node(0) and all() work.
    // If hardware_concurrency() is 0 (rare — uncomputable on the platform),
    // node(0) returns an empty set; affinity binding becomes a no-op but
    // ThreadPool still functions normally since it does not derive its
    // thread count from CpuSet.
    if (sets.empty()) {
      CpuSet s;
      s.addRange(0, static_cast<int32_t>(std::thread::hardware_concurrency()));
      sets.push_back(std::move(s));
    }
    return sets;
  }();
  return numaSets;
}

#if defined(__linux__)

// Parse a "cpuN" sysfs entry name, returning the CPU index, or -1 if `name` is
// not of the form cpu<digits>.
int32_t parseCpuDirName(const char* name) {
  if (name[0] != 'c' || name[1] != 'p' || name[2] != 'u' || name[3] == '\0') {
    return -1;
  }
  int32_t cpu = 0;
  for (const char* p = name + 3; *p; ++p) {
    if (*p < '0' || *p > '9') {
      return -1;
    }
    cpu = cpu * 10 + (*p - '0');
  }
  return cpu;
}

// Enumerate online CPU indices from /sys/devices/system/cpu, sorted ascending.
// Iterating the directory rather than a range bounded on hardware_concurrency()
// correctly handles sparse online CPU IDs (gaps or numbering beyond the count).
std::vector<int32_t> enumerateOnlineCpus(DIR* cpuDir) {
  std::vector<int32_t> cpuIndices;
  struct dirent* entry;
  while ((entry = ::readdir(cpuDir)) != nullptr) {
    int32_t cpu = parseCpuDirName(entry->d_name);
    if (cpu >= 0) {
      cpuIndices.push_back(cpu);
    }
  }
  // Stable ascending order so byId's first-seen cacheId is the lowest CPU's.
  std::sort(cpuIndices.begin(), cpuIndices.end());
  return cpuIndices;
}

// Read the cache ID for a CPU at the given cache index, or -1 if unavailable.
int32_t readCacheId(int cpuDirFd, int32_t cpu, const char* indexStr) {
  char idPath[64];
  snprintf(idPath, sizeof(idPath), "cpu%d/cache/%s/id", cpu, indexStr);
  auto idBuf = readSmallFile(cpuDirFd, idPath);
  if (idBuf.empty()) {
    return -1;
  }
  return detail::parseIntClamped(idBuf.data());
}

// Read the shared_cpu_list for a CPU/cache into `outCpus` (CPU IDs in [0, maxCpu]).
// Returns false if the sysfs entry is missing/empty.
bool readCacheCpuList(
    int cpuDirFd,
    int32_t cpu,
    const char* indexStr,
    int32_t maxCpu,
    std::vector<int32_t>& outCpus) {
  char listPath[64];
  snprintf(listPath, sizeof(listPath), "cpu%d/cache/%s/shared_cpu_list", cpu, indexStr);
  auto listBuf = readSmallFile(cpuDirFd, listPath);
  if (listBuf.empty()) {
    return false;
  }
  CpuSet cpuSet = detail::parseLinuxCpuList(listBuf.data());
  for (int32_t i = 0; i <= maxCpu; ++i) {
    if (cpuSet.contains(i)) {
      outCpus.push_back(i);
    }
  }
  return true;
}

#endif // __linux__

// Parses L2 or L3 cache sharing groups from sysfs.
// cacheIndex: 2 for L2, 3 for L3
std::vector<CacheGroup> parseCacheGroups(int cacheIndex) {
  std::vector<CacheGroup> groups;
#if defined(__linux__)
  char indexStr[8];
  snprintf(indexStr, sizeof(indexStr), "index%d", cacheIndex);

  DIR* cpuDir = ::opendir("/sys/devices/system/cpu");
  if (!cpuDir) {
    return groups;
  }
  int cpuDirFd = ::dirfd(cpuDir);

  const std::vector<int32_t> cpuIndices = enumerateOnlineCpus(cpuDir);
  const int32_t maxCpuSeen = cpuIndices.empty() ? 0 : cpuIndices.back(); // sorted ascending

  // Map from cache ID to CacheGroup, to deduplicate across CPUs sharing the same cache.
  std::map<int32_t, CacheGroup> byId;
  for (int32_t cpu : cpuIndices) {
    const int32_t cacheId = readCacheId(cpuDirFd, cpu, indexStr);
    if (cacheId < 0 || byId.count(cacheId)) {
      continue;
    }
    CacheGroup group;
    group.cacheId = cacheId;
    if (readCacheCpuList(cpuDirFd, cpu, indexStr, maxCpuSeen, group.cpus) && !group.cpus.empty()) {
      byId.emplace(cacheId, std::move(group));
    }
  }
  ::closedir(cpuDir);

  // Extract groups sorted by first CPU ID
  groups.reserve(byId.size());
  for (auto& kv : byId) {
    groups.push_back(std::move(kv.second));
  }
  std::sort(groups.begin(), groups.end(), [](const CacheGroup& a, const CacheGroup& b) {
    return a.cpus.front() < b.cpus.front();
  });
#endif // __linux__
  return groups;
}

const std::vector<CacheGroup>& getL2Groups() {
  static const std::vector<CacheGroup> groups = parseCacheGroups(2);
  return groups;
}

const std::vector<CacheGroup>& getL3Groups() {
  static const std::vector<CacheGroup> groups = parseCacheGroups(3);
  return groups;
}

int32_t getNumNumaNodes() {
  static const int32_t n = []() {
    auto sz = static_cast<int32_t>(getNumaSets().size());
    return sz > 0 ? sz : 1;
  }();
  return n;
}

} // namespace

#elif defined(DISPENSO_CPUSET_WINDOWS)

// =============================================================================
// CpuSet core methods (Windows)
// =============================================================================

void CpuSet::clear() {
  memset(words_, 0, sizeof(words_));
}

void CpuSet::add(int32_t hardwareThread) {
  // Real runtime bounds check (not just a debug assert): out-of-range IDs are
  // silently ignored rather than writing past words_, which would be undefined
  // behavior in release builds. CPU IDs can exceed kMaxCpus on very large
  // machines (e.g. Windows hosts with >16 processor groups); such CPUs are
  // simply not representable here (graceful degradation, no corruption).
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return;
  }
  words_[hardwareThread / kBitsPerWord] |= uint64_t{1} << (hardwareThread % kBitsPerWord);
}

void CpuSet::addRange(int32_t start, int32_t end) {
  for (int32_t i = start; i < end; ++i) {
    add(i);
  }
}

void CpuSet::remove(int32_t hardwareThread) {
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return;
  }
  words_[hardwareThread / kBitsPerWord] &= ~(uint64_t{1} << (hardwareThread % kBitsPerWord));
}

void CpuSet::removeRange(int32_t start, int32_t end) {
  for (int32_t i = start; i < end; ++i) {
    remove(i);
  }
}

bool CpuSet::contains(int32_t hardwareThread) const {
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return false;
  }
  return (words_[hardwareThread / kBitsPerWord] >> (hardwareThread % kBitsPerWord)) & 1;
}

namespace {
inline int32_t popcount64(uint64_t v) {
#if defined(_M_X64) || defined(_M_AMD64)
  return static_cast<int32_t>(__popcnt64(v));
#else
  return static_cast<int32_t>(__popcnt(static_cast<uint32_t>(v))) +
      static_cast<int32_t>(__popcnt(static_cast<uint32_t>(v >> 32)));
#endif
}
} // namespace

int32_t CpuSet::count() const {
  int32_t n = 0;
  for (int32_t i = 0; i < kNumWords; ++i) {
    n += popcount64(words_[i]);
  }
  return n;
}

bool CpuSet::bindCurrentThread() const {
  // Find which processor group(s) this set covers.
  // SetThreadGroupAffinity only supports one group at a time, so we bind
  // to the group with the most bits set (the "primary" group).
  // For thread groups built by buildThreadGroups(), all CPUs will typically
  // be in the same group since groups are ≤16 CPUs and L3-coherent.

  int32_t bestGroup = -1;
  int32_t bestCount = 0;
  KAFFINITY bestMask = 0;

  int32_t numGroups = static_cast<int32_t>(GetActiveProcessorGroupCount());
  for (int32_t g = 0; g < numGroups; ++g) {
    KAFFINITY mask = 0;
    int32_t groupBase = g * 64;
    int32_t groupSize = static_cast<int32_t>(GetActiveProcessorCount(static_cast<WORD>(g)));
    for (int32_t i = 0; i < groupSize && i < 64; ++i) {
      if (contains(groupBase + i)) {
        mask |= KAFFINITY{1} << i;
      }
    }
    int32_t c = popcount64(mask);
    if (c > bestCount) {
      bestCount = c;
      bestGroup = g;
      bestMask = mask;
    }
  }

  if (bestGroup < 0 || bestMask == 0) {
    return false;
  }

  GROUP_AFFINITY affinity = {};
  affinity.Group = static_cast<WORD>(bestGroup);
  affinity.Mask = bestMask;
  return SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) != 0;
}

int32_t CpuSet::currentHardwareThread() {
  PROCESSOR_NUMBER pn;
  GetCurrentProcessorNumberEx(&pn);
  return static_cast<int32_t>(pn.Group) * 64 + static_cast<int32_t>(pn.Number);
}

// =============================================================================
// Topology detection (Windows)
// =============================================================================

namespace {

// Retrieve SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX for a given relationship type.
std::vector<char> getLogicalProcessorInfo(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(relationship, nullptr, &length);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
    return {};
  }
  std::vector<char> buf(length);
  if (!GetLogicalProcessorInformationEx(
          relationship,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
          &length)) {
    return {};
  }
  buf.resize(length);
  return buf;
}

// Convert a GROUP_AFFINITY array to a flat list of CPU IDs.
std::vector<int32_t> groupAffinityToCpuList(const GROUP_AFFINITY* masks, WORD groupCount) {
  std::vector<int32_t> cpus;
  for (WORD g = 0; g < groupCount; ++g) {
    int32_t groupBase = static_cast<int32_t>(masks[g].Group) * 64;
    KAFFINITY mask = masks[g].Mask;
    while (mask) {
      unsigned long bit;
      _BitScanForward64(&bit, mask);
      cpus.push_back(groupBase + static_cast<int32_t>(bit));
      mask &= mask - 1; // Clear lowest set bit
    }
  }
  std::sort(cpus.begin(), cpus.end());
  return cpus;
}

const std::vector<CpuSet>& getNumaSets() {
  static const std::vector<CpuSet> numaSets = []() {
    std::vector<CpuSet> sets;
    auto buf = getLogicalProcessorInfo(RelationNumaNode);
    if (buf.empty()) {
      // API failure or unsupported — fall back to a single node with all CPUs.
      // GetActiveProcessorCount gives only a count, not a mask, so we assume
      // dense [0, count) numbering per group. This is correct on virtually all
      // systems where GetLogicalProcessorInformationEx is unavailable (legacy
      // Windows versions with single processor groups). The primary path above
      // handles sparse CPU IDs correctly via GROUP_AFFINITY masks.
      CpuSet fallback;
      WORD numGroups = GetActiveProcessorGroupCount();
      for (WORD g = 0; g < numGroups; ++g) {
        int32_t groupBase = static_cast<int32_t>(g) * 64;
        DWORD count = GetActiveProcessorCount(g);
        for (DWORD i = 0; i < count; ++i) {
          fallback.add(groupBase + static_cast<int32_t>(i));
        }
      }
      sets.push_back(std::move(fallback));
      return sets;
    }

    // Collect NUMA nodes, indexed by node number (may be sparse).
    std::map<int32_t, CpuSet> byNode;

    DWORD offset = 0;
    while (offset < buf.size()) {
      auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
      if (info->Relationship == RelationNumaNode) {
        int32_t nodeNum = static_cast<int32_t>(info->NumaNode.NodeNumber);
        CpuSet& nodeSet = byNode[nodeNum];
        // Windows 10 20H2+ (NTDDI_WIN10_FE) added GroupCount/GroupMasks for
        // multi-group NUMA nodes. Older SDKs only have the single-group GroupMask.
#if defined(NTDDI_WIN10_FE) && NTDDI_VERSION >= NTDDI_WIN10_FE
        WORD groupCount = info->NumaNode.GroupCount;
        if (groupCount > 0) {
          auto nodeCpus = groupAffinityToCpuList(info->NumaNode.GroupMasks, groupCount);
          for (int32_t cpu : nodeCpus) {
            nodeSet.add(cpu);
          }
        } else
#endif
        {
          auto cpus = groupAffinityToCpuList(&info->NumaNode.GroupMask, 1);
          for (int32_t cpu : cpus) {
            nodeSet.add(cpu);
          }
        }
      }
      if (info->Size == 0) {
        break;
      }
      offset += info->Size;
    }

    // Flatten to a contiguous vector indexed by node order.
    sets.reserve(byNode.size());
    for (auto& kv : byNode) {
      sets.push_back(std::move(kv.second));
    }
    return sets;
  }();
  return numaSets;
}

std::vector<CacheGroup> parseCacheGroupsWin(int cacheLevel) {
  std::vector<CacheGroup> groups;
  auto buf = getLogicalProcessorInfo(RelationCache);
  if (buf.empty()) {
    return groups;
  }

  int32_t nextCacheId = 0;
  DWORD offset = 0;
  while (offset < buf.size()) {
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
    if (info->Relationship == RelationCache && static_cast<int>(info->Cache.Level) == cacheLevel &&
        (info->Cache.Type == CacheData || info->Cache.Type == CacheUnified)) {
      CacheGroup group;
      group.cacheId = nextCacheId++;
      // Windows 10 20H2+ (NTDDI_WIN10_FE) added GroupCount/GroupMasks for
      // multi-group caches. Older SDKs only have the single-group GroupMask.
#if defined(NTDDI_WIN10_FE) && NTDDI_VERSION >= NTDDI_WIN10_FE
      WORD groupCount = info->Cache.GroupCount;
      if (groupCount > 0) {
        group.cpus = groupAffinityToCpuList(info->Cache.GroupMasks, groupCount);
      } else
#endif
      {
        group.cpus = groupAffinityToCpuList(&info->Cache.GroupMask, 1);
      }
      if (!group.cpus.empty()) {
        groups.push_back(std::move(group));
      }
    }
    if (info->Size == 0) {
      break;
    }
    offset += info->Size;
  }

  std::sort(groups.begin(), groups.end(), [](const CacheGroup& a, const CacheGroup& b) {
    return a.cpus.front() < b.cpus.front();
  });
  return groups;
}

const std::vector<CacheGroup>& getL2Groups() {
  static const std::vector<CacheGroup> groups = parseCacheGroupsWin(2);
  return groups;
}

const std::vector<CacheGroup>& getL3Groups() {
  static const std::vector<CacheGroup> groups = parseCacheGroupsWin(3);
  return groups;
}

int32_t getNumNumaNodes() {
  static const int32_t n = []() {
    auto sz = static_cast<int32_t>(getNumaSets().size());
    return sz > 0 ? sz : 1;
  }();
  return n;
}

} // namespace

#else // !DISPENSO_CPUSET_LINUXY && !DISPENSO_CPUSET_WINDOWS

// Platforms without native affinity support (macOS, etc.)
// Set manipulation uses the portable bitset so that buildThreadGroups()
// produces correct groups via the contiguous fallback path. Only
// binding and topology detection are no-ops.

void CpuSet::clear() {
  memset(words_, 0, sizeof(words_));
}

void CpuSet::add(int32_t hardwareThread) {
  // Real runtime bounds check (not just a debug assert): out-of-range IDs are
  // silently ignored rather than writing past words_, which would be undefined
  // behavior in release builds. CPU IDs can exceed kMaxCpus on very large
  // machines (e.g. Windows hosts with >16 processor groups); such CPUs are
  // simply not representable here (graceful degradation, no corruption).
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return;
  }
  words_[hardwareThread / kBitsPerWord] |= uint64_t{1} << (hardwareThread % kBitsPerWord);
}

void CpuSet::addRange(int32_t start, int32_t end) {
  for (int32_t i = start; i < end; ++i) {
    add(i);
  }
}

void CpuSet::remove(int32_t hardwareThread) {
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return;
  }
  words_[hardwareThread / kBitsPerWord] &= ~(uint64_t{1} << (hardwareThread % kBitsPerWord));
}

void CpuSet::removeRange(int32_t start, int32_t end) {
  for (int32_t i = start; i < end; ++i) {
    remove(i);
  }
}

bool CpuSet::contains(int32_t hardwareThread) const {
  if (hardwareThread < 0 || hardwareThread >= kMaxCpus) {
    return false;
  }
  return (words_[hardwareThread / kBitsPerWord] >> (hardwareThread % kBitsPerWord)) & 1;
}

int32_t CpuSet::count() const {
  int32_t n = 0;
  for (int32_t i = 0; i < kNumWords; ++i) {
    // Portable popcount: __builtin_popcountll on GCC/Clang, loop fallback otherwise
#if defined(__GNUC__) || defined(__clang__)
    n += __builtin_popcountll(words_[i]);
#else
    uint64_t w = words_[i];
    while (w) {
      ++n;
      w &= w - 1;
    }
#endif
  }
  return n;
}

bool CpuSet::bindCurrentThread() const {
  return false; // Not supported on this platform
}

int32_t CpuSet::currentHardwareThread() {
#if defined(__APPLE__)
  // _os_cpu_number is a private Apple API in libsystem_platform.dylib.
  // Load via dlsym to avoid a hard link dependency on the private symbol.
  using OsCpuNumberFn = unsigned int (*)();
  static const auto fn = reinterpret_cast<OsCpuNumberFn>(dlsym(RTLD_DEFAULT, "_os_cpu_number"));
  if (fn) {
    return static_cast<int32_t>(fn());
  }
  return -1;
#else
  return -1;
#endif
}

namespace {
const std::vector<CpuSet>& getNumaSets() {
  // Return a single-node fallback containing all CPUs so that node(0)
  // and all() work correctly on platforms without NUMA detection.
  static const std::vector<CpuSet> fallback = []() {
    std::vector<CpuSet> sets;
    CpuSet s;
    s.addRange(0, static_cast<int32_t>(std::thread::hardware_concurrency()));
    sets.push_back(std::move(s));
    return sets;
  }();
  return fallback;
}
const std::vector<CacheGroup>& getL2Groups() {
  static const std::vector<CacheGroup> empty;
  return empty;
}
const std::vector<CacheGroup>& getL3Groups() {
  static const std::vector<CacheGroup> empty;
  return empty;
}
int32_t getNumNumaNodes() {
  return 1;
}
} // namespace

#endif // platform selection

// =============================================================================
// Static query methods (platform-independent)
// =============================================================================

int32_t CpuSet::totalNumaNodes() {
  return getNumNumaNodes();
}

const CpuSet& CpuSet::node(int32_t numaNode) {
  assert(numaNode >= 0 && numaNode < getNumNumaNodes());
  return getNumaSets()[static_cast<size_t>(numaNode)];
}

const CpuSet& CpuSet::all() {
  static const CpuSet set = []() {
    // Build from union of all NUMA nodes (handles non-contiguous CPU IDs)
    const auto& nodes = getNumaSets();
    if (!nodes.empty()) {
      CpuSet s;
      // Scan the full range supported by the underlying set representation
      // (1024 on both Linux cpu_set_t and the portable bitset). Using
      // hardware_concurrency() would miss sparse CPU IDs above that count.
      constexpr int32_t kScanLimit = 1024;
      for (const auto& nodeSet : nodes) {
        for (int32_t i = 0; i < kScanLimit; ++i) {
          if (nodeSet.contains(i)) {
            s.add(i);
          }
        }
      }
      return s;
    }
    // Fallback: assume contiguous [0, hardware_concurrency)
    CpuSet s;
    s.addRange(0, static_cast<int32_t>(std::thread::hardware_concurrency()));
    return s;
  }();
  return set;
}

int32_t CpuSet::availableCount() {
#if defined(__linux__)
  cpu_set_t mask;
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    return static_cast<int32_t>(CPU_COUNT(&mask));
  }
#elif defined(_WIN32)
  {
    WORD numGroups = GetActiveProcessorGroupCount();
    if (numGroups <= 1) {
      // Single processor group: GetProcessAffinityMask correctly reflects
      // process-level restrictions (job objects, start /affinity, etc.).
      DWORD_PTR processAffinity, systemAffinity;
      if (GetProcessAffinityMask(GetCurrentProcess(), &processAffinity, &systemAffinity)) {
        int32_t count = 0;
        for (auto m = processAffinity; m != 0; m &= m - 1) {
          ++count;
        }
        return count;
      }
    }
    // Multi-group: no single Win32 API gives per-group process affinity.
    // Sum active processors per group as the best available approximation.
    int32_t count = 0;
    for (WORD g = 0; g < numGroups; ++g) {
      count += static_cast<int32_t>(GetActiveProcessorCount(g));
    }
    if (count > 0) {
      return count;
    }
  }
#endif
  auto hc = std::thread::hardware_concurrency();
  return hc > 0 ? static_cast<int32_t>(hc) : 1;
}

const std::vector<CacheGroup>& CpuSet::l2CacheGroups() {
  return getL2Groups();
}

const std::vector<CacheGroup>& CpuSet::l3CacheGroups() {
  return getL3Groups();
}

// =============================================================================
// Thread group building
// =============================================================================

namespace {

// Build a CpuSet from a vector of CPU IDs.
CpuSet cpuSetFromIds(const std::vector<int32_t>& cpus) {
  CpuSet s;
  for (int32_t cpu : cpus) {
    s.add(cpu);
  }
  return s;
}

// Flush accumulated CPUs into a ThreadGroup and push it to the output.
void flushGroup(std::vector<int32_t>& pending, std::vector<ThreadGroup>& out) {
  if (pending.empty()) {
    return;
  }
  std::sort(pending.begin(), pending.end());
  ThreadGroup group;
  group.cpus = std::move(pending);
  group.affinityMask = cpuSetFromIds(group.cpus);
  out.push_back(std::move(group));
  pending.clear();
}

// Largest CPU count across the given cache groups.
int32_t largestGroupSize(const std::vector<CacheGroup>& groups) {
  int32_t maxSize = 0;
  for (const auto& g : groups) {
    maxSize = std::max(maxSize, static_cast<int32_t>(g.cpus.size()));
  }
  return maxSize;
}

// Build a lookup mapping each CPU ID to its L3 group index (-1 if none). Sized by
// the max CPU ID seen in the L3 groups (may exceed hardware_concurrency() when the
// process is pinned to a subset of cores).
std::vector<int32_t> buildCpuToL3Map(const std::vector<CacheGroup>& l3Groups) {
  int32_t maxCpu = static_cast<int32_t>(std::thread::hardware_concurrency());
  for (const auto& g : l3Groups) {
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
  return cpuToL3;
}

// Look up a CPU's L3 group index, returning -1 when out of range / unmapped.
int32_t l3IndexForCpu(const std::vector<int32_t>& cpuToL3, int32_t cpu) {
  if (cpu >= 0 && static_cast<size_t>(cpu) < cpuToL3.size()) {
    return cpuToL3[static_cast<size_t>(cpu)];
  }
  return -1;
}

// Pack L2 atoms into contiguous groups, flushing whenever the next atom would
// cross an L3 boundary or exceed maxGroupSize. L2 groups are already sorted by
// first CPU ID, so iterating in order yields contiguous, cache-coherent groups.
std::vector<ThreadGroup> buildGroupsFromCacheTopology(
    const std::vector<CacheGroup>& l2Groups,
    const std::vector<CacheGroup>& l3Groups,
    int32_t maxGroupSize) {
  // Clamp maxGroupSize to at least the largest L2 atom, so we never split siblings.
  maxGroupSize = std::max(maxGroupSize, largestGroupSize(l2Groups));

  const std::vector<int32_t> cpuToL3 = buildCpuToL3Map(l3Groups);

  std::vector<ThreadGroup> result;
  std::vector<int32_t> pending;
  int32_t currentL3 = -1;
  for (const auto& l2 : l2Groups) {
    if (l2.cpus.empty()) {
      continue;
    }
    const int32_t l2L3 = l3IndexForCpu(cpuToL3, l2.cpus[0]);
    const int32_t l2Size = static_cast<int32_t>(l2.cpus.size());

    const bool crossesL3 = (l2L3 != currentL3 && currentL3 >= 0);
    const bool exceedsMax = (static_cast<int32_t>(pending.size()) + l2Size > maxGroupSize);
    if (crossesL3 || exceedsMax) {
      flushGroup(pending, result);
    }

    currentL3 = l2L3;
    pending.insert(pending.end(), l2.cpus.begin(), l2.cpus.end());
  }
  flushGroup(pending, result);
  return result;
}

// Fallback when no cache topology is available: chunk all online CPUs into
// contiguous groups of up to maxGroupSize.
std::vector<ThreadGroup> buildGroupsContiguous(int32_t maxGroupSize) {
  std::vector<ThreadGroup> result;
  const CpuSet& allSet = CpuSet::all();
  constexpr int32_t kScanLimit = 1024;

  std::vector<int32_t> pending;
  for (int32_t i = 0; i < kScanLimit; ++i) {
    if (!allSet.contains(i)) {
      continue;
    }
    pending.push_back(i);
    if (static_cast<int32_t>(pending.size()) >= maxGroupSize) {
      flushGroup(pending, result);
    }
  }
  flushGroup(pending, result);
  return result;
}

} // namespace

std::vector<ThreadGroup> CpuSet::buildThreadGroups(int32_t maxGroupSize) {
  const auto& l2Groups = getL2Groups();
  const auto& l3Groups = getL3Groups();

  // With cache topology, build groups bottom-up from L2 atoms respecting L3
  // boundaries; otherwise fall back to contiguous chunking of all online CPUs.
  if (!l2Groups.empty()) {
    return buildGroupsFromCacheTopology(l2Groups, l3Groups, maxGroupSize);
  }
  return buildGroupsContiguous(maxGroupSize);
}

} // namespace dispenso
