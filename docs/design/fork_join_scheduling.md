# Fork-Join Scheduling & Thread Groups

Design document for adding fork-join-style scheduling to dispenso's thread
pool, enabling cache-affinity-aware `parallel_for` while preserving all
existing semantics.

**Status:** Design phase
**Target:** post-1.5.0

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Component 1: MpmcRingBuffer](#component-1-mpmcringbuffer)
- [Component 2: Thread Groups](#component-2-thread-groups)
- [Component 3: Topology Detection (CpuSet)](#component-3-topology-detection-cpuset)
- [Component 4: Scheduling Paths](#component-4-scheduling-paths)
- [Component 5: Wake Strategy](#component-5-wake-strategy)
- [Steal Order](#steal-order)
- [Platform Support](#platform-support)
- [Commit Plan](#commit-plan)
- [Open Questions](#open-questions)
- [Appendix: Architecture-Specific Topology](#appendix-architecture-specific-topology)

## Motivation

Dispenso's shared `moodycamel::ConcurrentQueue` gives excellent throughput for
general task scheduling but cannot express thread-to-data affinity. In
benchmarks on multi-NUMA machines, this shows up as a significant performance
gap on locality-sensitive `parallel_for` patterns.

### Measured Impact

**cascading_parallel_for_benchmark, 10M elements, 96 threads, dual-socket AMD
EPYC:**

| Metric | OMP (fork-join) | dispenso (shared queue) | TBB (work-stealing) |
|--------|-----------------|------------------------|---------------------|
| Time | 2.99 ms | 15.2 ms | 15.4 ms |
| IPC | **1.53** | **0.30** | **0.24** |
| Insns/iter | 1.43 B | 2.62 B | 2.43 B |

The IPC data is the smoking gun: task-based threads run at 0.24-0.30 IPC
(stalled ~80% on memory), while OMP's statically-bound threads run at 1.53
IPC. The instruction count difference is only ~1.8x (framework overhead); the
remaining ~5x comes from memory stalls caused by non-deterministic
task-to-thread mapping destroying L2 cache affinity across iterations. Each
thread's chunk (~104K elements x 4B = 416 KB input + output = 832 KB) fits in
L2 (1 MB). With OMP, thread T always gets chunk T (warm L2). With queue-based
dispatch, each iteration is an L2-cold start.

The 100K element case (chunk ~4 KB, fits in L1) shows only 1.35x gap,
confirming the effect scales with chunk size relative to cache capacity.

### What This Won't Help

- **Graphs** (`graph_benchmark`): already ~1880x faster than TaskFlow;
  heterogeneous nodes have no spatial locality.
- **Pipelines** (`pipeline_benchmark`): already ~2.6x faster than TBB;
  parallelism is across pipeline stages, not spatial.
- **Non-pool callers**: no per-thread ring, work goes to central queue as
  today.

## Architecture Overview

```
                     +---------------------+
                     |   Central MPMC Queue |  <-- existing moodycamel queue
                     |   (unchanged)        |
                     +----------+----------+
                                | overflow / non-pool callers
          +---------+-----------+-----------+---------+
          v         v           v           v         v
     +---------+ +---------+         +---------+ +---------+
     | Ring  0 | | Ring  1 |  ...    | Ring N-2| | Ring N-1|
     | (16 sl.)| | (16 sl.)|         | (16 sl.)| | (16 sl.)|
     +---------+ +---------+         +---------+ +---------+
       Thread 0    Thread 1            Thread N-2   Thread N-1
     |<--- Group 0 --->|  ...  |<--- Group K --->|
       EpochWaiter 0              EpochWaiter K
```

Three new components layered on the existing design:

1. **MpmcRingBuffer** - Standalone bounded MPMC queue (Vyukov algorithm)
2. **Thread Groups** - Scheduling and wake units of <= maxGroupSize threads
3. **Topology Detection** - L2/L3 cache hierarchy parsing for group assignment

## Component 1: MpmcRingBuffer

### Algorithm

Vyukov bounded MPMC queue with per-slot sequence numbers and power-of-two ring
buffer. O(1) push/pop, fail-fast when full/empty, no CAS retry loops on the
uncontended fast path.

Well-established algorithm: used in Rust crossbeam `ArrayQueue`, Go runtime,
folly, and many production systems.

### Why Vyukov MPMC, Not Chase-Lev

Chase-Lev is fundamentally SPMC -- the owner's wait-free push/pop relies on
exclusive access to the `bottom` pointer. Making push MPMC requires CAS on
every push, destroying the key performance property. There is no known cheap
lock-free MPMC deque with asymmetric LIFO/FIFO pop. For 16-deep queues the
LIFO-vs-FIFO ordering difference is negligible: the real locality win is
thread-to-chunk affinity, not pop ordering.

### API

Standalone header-only template in `dispenso/mpmc_ring_buffer.h`, following
the pattern of `dispenso/spsc_ring_buffer.h`.

```cpp
template <typename T, size_t Capacity = 16, bool RoundUpToPowerOfTwo = true>
class MpmcRingBuffer {
 public:
  // Single-item operations
  bool try_push(T&& val);
  bool try_push(const T& val);
  template <typename... Args>
  bool try_emplace(Args&&... args);
  bool try_pop(T& val);
  OpResult<T> try_pop();

  // Bulk operations
  // Returns M <= count: number of items actually pushed. Items are moved from,
  // so the pointer is non-const. Caller handles overflow (e.g. push remainder
  // to central queue).
  size_t try_push_batch(T* items, size_t count);

  // Queries
  bool empty() const;
  bool full() const;
  size_t size() const;
  static constexpr size_t capacity();
};
```

### Bulk Push Strategy

Bulk push returns M <= N (number actually enqueued), enabling natural overflow:

```cpp
size_t remaining = count;
size_t pushed = ring.try_push_batch(items, remaining);
remaining -= pushed;
if (remaining > 0) {
    // Overflow to central queue
    pool.scheduleBulk(remaining, ...);
}
```

The implementation uses a single CAS:

1. Scan forward from `tail_`, counting consecutive available slots (each slot's
   sequence number equals its expected position). Stop at the first unavailable
   slot, yielding an available count M (capped at `count`).
2. CAS `tail_` forward by M to reserve that contiguous run in one shot.
3. **On CAS failure**: return 0 (a peer claimed an overlapping range -- caller
   uses the central queue). No retry loop.

After a successful CAS, each reserved slot is move-constructed from `items[i]`
and its sequence number published independently, so consumers can start popping
slot 0 before slot M-1 is written.

### Memory Layout

Each slot is padded up to a cache-line multiple to avoid false sharing between
adjacent slots under contention from multiple stealers. The element buffer is
placed *before* the sequence number so that any padding `T`'s alignment requires
does not land between `seq` and `data` (interior padding that could otherwise
tip a slot over a cache-line boundary):

```cpp
struct alignas(kCacheLineSize) Slot {
    alignas(T) char data[sizeof(T)];
    std::atomic<size_t> seq;
};
```

Per-slot size depends on `sizeof(T)`, not just the cache line size: a slot fits
in one cache line only when `sizeof(T) + sizeof(std::atomic<size_t>)` does
(roughly `sizeof(T) <= 56` for a 64-byte line, `<= 120` for 128-byte lines).
For a small `T`, 16 slots x 64 bytes = **1 KB per ring**; with 256 threads,
total ring memory is 256 KB -- negligible. Larger elements scale the per-ring
footprint by the actual slot size (e.g. a 64-byte element yields 128-byte slots,
so 2 KB per ring).

## Component 2: Thread Groups

The **thread group** is the fundamental scheduling and wake unit.

### Properties

- Each group contains <= `maxGroupSize` threads (default: 16)
- All threads in a group are on the same NUMA node (NUMA-coherent)
- All threads in a group share an L3 cache when possible
- SMT siblings (threads sharing L2) are never split across groups
- Each group has its own `EpochWaiter` for targeted waking

### Why Groups, Not Per-Thread or Per-NUMA-Node Waking

**Per-thread EpochWaiter is too expensive for bulk wake.** Waking 256 threads
individually costs ~38 us in futex syscalls alone, plus cache line bouncing.
The scheduling thread can't start its own chunk until all wakes are issued.

**Per-NUMA-node EpochWaiter is too coarse.** A NUMA node can have 32-128
threads. Waking all of them for a small parallel_for is wasteful (thundering
herd, cross-CCD cache pollution, wasted power).

**Per-group EpochWaiter (groups of <= 16)** gives:
- O(num_groups) futex calls for full wake (~16-24 calls, ~3-4 us)
- Targeted wake: small parallel_for wakes only the needed groups
- Natural match to L3/CCD cache boundaries
- Uniform behavior across platforms (non-NUMA systems just have more groups)

### Data Structure

```cpp
struct ThreadGroup {
    alignas(kCacheLineSize) EpochWaiter waiter;
    alignas(kCacheLineSize) std::atomic<int32_t> numSleeping{0};
    uint16_t startThreadIdx;   // First thread index in pool
    uint16_t count;            // Number of threads in this group
};

// In ThreadPool:
std::vector<ThreadGroup> groups_;
std::vector<uint16_t> threadToGroup_;  // thread idx -> group idx
```

### Default maxGroupSize: 16

- Matches AMD CCD boundary (8 cores x 2 SMT = 16 threads)
- Gives Power10/11 (SMT8) at least 2 cores per group
- Works well for AmpereOne (16 cores per group, arbitrary but reasonable)
- Scanning 15 neighbor rings during steal is cheap
- 1 KB/ring x 16 rings = 16 KB ring data per group (fits in L1)
- Configurable at ThreadPool construction for tuning on specific hardware

### Degenerate Cases

| System | Groups | Behavior |
|--------|--------|----------|
| Single-socket, <= 16 threads | 1 group | Equivalent to single EpochWaiter (status quo) |
| Non-NUMA, 64 threads | 4 groups | Arbitrary but contiguous grouping |
| NUMA detection fails (Mac) | ceil(N/16) groups | Falls back to contiguous chunking |

## Component 3: Topology Detection (CpuSet)

### Group Building Algorithm

Groups are built bottom-up from cache topology:

```
1. Parse unique L2 groups (sysfs cache/index2/shared_cpu_list)
   -> These are SMT sibling sets: {0,1}, {2,3}, ... on x86
   -> Atoms that are never split across groups

2. Parse unique L3 groups (sysfs cache/index3/shared_cpu_list)
   -> CCD boundaries on AMD, tile boundaries on Intel
   -> Containment boundary: groups never cross L3 boundaries

3. Within each L3 group, pack L2 atoms into scheduling groups
   of <= maxGroupSize threads

4. Never split an L2 atom across scheduling groups
```

### Why L2 as the Atom

L2 sharing identifies SMT siblings -- threads that share L1 and L2 caches.
These must stay together because:
- Splitting them wastes a group slot (SMT sibling will contend for the same
  cache regardless of group assignment)
- They share cache constructively when processing adjacent chunks

L2 detection via sysfs works reliably even under virtualization (VMs correctly
report L2 sharing = SMT pairs), unlike L3/cluster detection which hypervisors
often flatten.

### Architecture Examples

| System | L2 atom | L3 group | maxGroupSize=16 | Scheduling groups |
|--------|---------|----------|-----------------|-------------------|
| AMD EPYC Genoa 96c x 2 SMT | 2 threads | 16 threads (CCD) | 12 groups/socket | = CCD boundaries |
| AMD Threadripper 7995WX 96c x 2 SMT | 2 threads | 16 threads (CCD) | 12 groups | = CCD boundaries |
| Intel Sapphire Rapids 56c x 2 SMT | 2 threads | ~28 threads (tile) | 4 groups of 14 | Sub-tile |
| AmpereOne 192c (no SMT) | 1 thread | 192 threads (all) | 12 groups of 16 | Contiguous |
| Power10/11 16c x SMT8 | 8 threads | ~128 threads (chip) | 8 groups of 16 | 2 cores/group |
| Laptop 8c x 2 SMT | 2 threads | 16 threads (all) | 1 group of 16 | Single group |
| VM (flat topology) | 2 threads | all threads | N/16 groups | Contiguous |

### CpuSet API

```cpp
class CpuSet {
 public:
  CpuSet();

  void clear();
  void add(int hardwareThread);
  void addRange(int hardwareThreadStart, int hardwareThreadEnd);
  void remove(int hardwareThread);
  void removeRange(int hardwareThreadStart, int hardwareThreadEnd);

  bool contains(int hardwareThread) const;
  int count() const;

  bool bindCurrentThread();  // Returns false on failure or unsupported

  static int totalNumaNodes();       // >= 1 (falls back to 1)
  static int currentHardwareThread();
  static const CpuSet& node(int numaNode);
  static const CpuSet& all();
};
```

### sysfs Paths Used

All detection uses files that exist on standard Linux kernels (no library
dependencies):

```
/sys/devices/system/node/nodeN/cpulist          -> NUMA node membership
/sys/devices/system/cpu/cpuN/cache/index2/shared_cpu_list  -> L2 (SMT siblings)
/sys/devices/system/cpu/cpuN/cache/index2/id    -> Unique L2 instance
/sys/devices/system/cpu/cpuN/cache/index3/shared_cpu_list  -> L3 (CCD/tile)
/sys/devices/system/cpu/cpuN/cache/index3/id    -> Unique L3 instance
```

### Platform Support

| Platform | Binding | Topology detection | Group building |
|----------|---------|-------------------|----------------|
| Linux | `pthread_setaffinity_np` | sysfs cache + NUMA | Full L2/L3 hierarchy |
| Windows | `SetThreadGroupAffinity` | `GetLogicalProcessorInformationEx` | L2/L3 from CACHE_RELATIONSHIP |
| macOS | No-op (unsupported) | `sysctlbyname("hw.packages")` | Fallback: chunk by maxGroupSize |
| FreeBSD | `cpuset_setaffinity` | sysfs (Linux compat) or manual | Deferred (no test hardware) |

Windows and Linux implementations land as separate commits. macOS provides
best-effort topology query but no binding. FreeBSD is deferred.

## Component 4: Scheduling Paths

### Path 1: Targeted (kStatic parallel_for)

Caller divides range into N chunks with linear layout (contiguous chunks go
to the same group), pushes chunk K to ring K.

```
256 chunks, 16 groups of 16 threads:
  Group 0 threads get chunks 0-15   (contiguous in array)
  Group 1 threads get chunks 16-31  (contiguous in array)
  ...
  Group 15 threads get chunks 240-255
```

Each thread's ring receives exactly 1 chunk. Calling thread takes one chunk
inline and participates.

**Linear layout rationale**: keeps spatially adjacent chunks on the same NUMA
node and L3 cache domain. Round-robin would scatter each thread's chunks
across the entire array, defeating NUMA locality and first-touch allocation
benefits. Work stealing handles any load imbalance.

### Path 2: Best-Effort Targeted (kAuto parallel_for)

kAuto creates up to `numThreads * dynamicFactor` chunks (e.g. 256 chunks for
16 threads). These are distributed to per-thread rings with linear layout,
overflow going to central queue.

```
256 chunks, 16 threads, 16-slot rings:
  Ring 0 gets chunks 0-15  via try_push_batch (1 CAS)
  Ring 1 gets chunks 16-31 via try_push_batch (1 CAS)
  ...
  If ring K is partially full, try_push_batch returns M < 16,
  remaining 16-M chunks go to central queue.
```

Bulk push makes this efficient: one CAS per ring for up to 16 items. Total
scheduling cost for 256 chunks = 16 CAS operations + overflow to central
queue.

### Path 3: Central Queue (existing, unchanged)

Used by Futures, Pipelines, non-pool callers, and as overflow from paths 1-2.
Behavior is identical to current dispenso.

### TaskSet Integration

TaskSet must remain in the loop -- bypassing it would break the unified
wait/cancel/exception model. Tasks pushed to per-thread rings still increment
`outstandingTaskCount_` and go through `packageTask` wrapping for exception
handling and cancellation support.

The key change to the wait loop:

```cpp
while (outstandingTaskCount_ > 0) {
    // 1. Check own ring first (highest locality)
    if (myRing.tryPop(func)) { exec(func); continue; }

    // 2. Steal from same-group rings (random sample, not full scan)
    if (stealFromGroup(myGroupIdx, func)) { exec(func); continue; }

    // 3. Central queue (existing path)
    if (pool_.tryExecuteNext()) continue;

    yield();
}
```

## Component 5: Wake Strategy

### Per-Group EpochWaiter

Each `ThreadGroup` has its own `EpochWaiter`. Thread pool threads sleep on
their group's waiter.

**Targeted wake (fork-join path):**

```cpp
void wakeGroup(int groupIdx, int count) {
    groups_[groupIdx].waiter.bumpAndWakeN(
        count, groups_[groupIdx].numSleeping.load());
}
```

For kStatic parallel_for: wake only the groups whose rings received work.
For a 16-thread parallel_for on a 256-thread pool, wake 1 group (1 futex)
instead of all threads.

**Central queue wake (existing work path):**

```cpp
void wakeForCentralQueue(int n) {
    // Round-robin across groups to distribute wakes
    for (int i = 0; i < numGroups && n > 0; ++i) {
        int g = (lastWokenGroup_ + i) % numGroups;
        int toWake = std::min(n, groups_[g].numSleeping.load());
        if (toWake > 0) {
            groups_[g].waiter.bumpAndWakeN(toWake, ...);
            n -= toWake;
        }
    }
}
```

## Steal Order

```
own ring  ->  random ring(s) in same group  ->  central queue
```

**Own ring first**: highest locality (the fork-join scheduler put work here
specifically for this thread).

**Same-group random sample**: steal from a random subset of group peers
(e.g. 2-4 victims), not a full scan. All group peers share L3, so stolen
work still has cache locality. Random selection avoids herding.

**Central queue last**: picks up overflow work from any source. This is the
existing proven path and handles all non-fork-join work.

**Not included (initially)**: cross-group ring stealing. This adds complexity
for marginal benefit -- cross-group work is better served by the central queue
which is already designed for contended multi-producer/multi-consumer access.
Can be added later if profiling shows the central queue is a bottleneck for
cross-group steal scenarios.

## Platform Support

### Commit Plan

Implementation is layered so each commit is independently useful and testable:

| # | Commit | Dependencies | Independently useful? |
|---|--------|-------------|----------------------|
| 1 | `MpmcRingBuffer` header + tests | None | Yes (general-purpose MPMC queue) |
| 2 | `CpuSet` (Linux) + topology detection | None | Yes (users can bind threads manually) |
| 3 | Per-thread rings in ThreadPool | #1 | Yes (cache affinity for kStatic) |
| 4 | Thread groups + per-group EpochWaiter | #1, #2 | Yes (targeted waking) |
| 5 | kStatic targeted scheduling | #3, #4 | Yes (primary perf win) |
| 6 | kAuto best-effort ring scheduling | #3, #4 | Yes (incremental locality win) |
| 7 | `CpuSet` (Windows) | #2 | Yes (Windows topology support) |
| 8 | `CpuSet` (macOS best-effort) | #2 | Yes (topology query, no binding) |

Commits 1-2 are independent foundations that can land and be reviewed in
parallel. Commit 3 delivers the primary performance improvement (closing the
IPC gap with OMP). Commits 4-6 are incremental optimizations. Commits 7-8
extend platform coverage.

### Dependency Requirements

**Zero external library dependencies.** All topology detection uses OS-provided
interfaces:
- Linux: sysfs pseudo-filesystem, pthreads, futex
- Windows: Win32 API (`GetLogicalProcessorInformationEx`,
  `SetThreadGroupAffinity`, `WaitOnAddress`)
- macOS: `sysctlbyname`

## Open Questions

### Scheduling

- **kAuto + ring interaction with `ConcurrentTaskSet`**: `ConcurrentTaskSet`
  lacks a producer token. Should it use ring-based scheduling at all, or
  always use central queue? Leaning toward central queue only, since
  `ConcurrentTaskSet` is designed for multi-threaded submission where
  ring targeting is less meaningful.

- **Ring scanning cost**: at high thread counts, even same-group ring scanning
  (15 `tryPop` calls) adds up if done on every wait iteration. Should we limit
  steal attempts per iteration (e.g., try 2-4 random victims per iteration)?

- **`scheduleBulk` integration**: current `scheduleBulk` interleaves inline
  execution and queuing based on load. The fork-join path should integrate
  with this -- how does inline execution interact with ring assignment?
  Likely: scheduling thread takes chunk 0 inline, pushes chunks 1-N to rings.

### Topology

- **Heterogeneous cores (big.LITTLE, P/E cores)**: should groups respect
  core type boundaries? On Intel 14th gen, P-cores and E-cores have different
  L2/L3 topology. Grouping P and E cores together is probably wrong.
  Detection: `sysfs cpu/cpuN/cpufreq/cpuinfo_max_freq` differs between types.
  Deferred until there's demand.

- **`cpu_set_t` 1024 CPU limit**: `cpu_set_t` on Linux is limited to 1024
  CPUs. Machines with >1024 logical CPUs need `CPU_ALLOC`/`CPU_*_S` dynamic
  variants. Not urgent but worth a comment.

- **Hot-plug / cgroup changes**: topology is detected once at initialization.
  CPU hot-plug or cgroup changes after init are not reflected. This is
  acceptable for the initial implementation.

### Performance

- **Optimal ring capacity**: 16 slots is chosen to match kAuto's 16x
  oversubscription factor. Should this be configurable? Template parameter
  on MpmcRingBuffer makes it zero-cost to change, but ThreadPool would need
  to pick a fixed value.

- **Benchmarking plan**: once implemented, re-run `cascading_parallel_for_benchmark`,
  `locality_benchmark`, `simple_for`, and `summing_for` on bare-metal
  multi-NUMA hardware. Target: close the IPC gap with OMP for kStatic,
  measurable improvement for kAuto.

## Appendix: Architecture-Specific Topology

### AMD EPYC Genoa (Zen 4)

```
Socket
  +-- CCD 0 (8 cores, 16 threads, 32 MB shared L3)
  |     +-- Core 0 (2 SMT threads, 1 MB private L2)
  |     +-- Core 1 ...
  |     +-- Core 7
  +-- CCD 1 ...
  +-- CCD 11
```

- L2 atom: 2 threads (SMT pair)
- L3 group: 16 threads (CCD)
- With maxGroupSize=16: scheduling group = CCD (ideal)
- NPS settings (NPS1/2/4) affect NUMA node count but not L3/CCD boundaries

### Intel Sapphire Rapids / Granite Rapids

```
Socket
  +-- Tile 0 (~14 cores, 28 threads, shared L3 slice)
  |     +-- Core 0 (2 SMT threads, 2 MB private L2)
  |     +-- ...
  +-- Tile 1 ...
  +-- Tile 3
```

- L2 atom: 2 threads (SMT pair)
- L3 group: ~28 threads (tile)
- With maxGroupSize=16: 2 groups per tile (sub-tile, still shares L3)
- Sub-NUMA Clustering (SNC) exposes tiles as NUMA nodes

### AmpereOne (Neoverse)

```
Socket
  +-- Core 0 (1 thread, 2 MB private L2, no SMT)
  +-- Core 1 ...
  +-- Core 191
  +-- Shared SLC (System Level Cache, mesh-distributed)
```

- L2 atom: 1 thread (no SMT)
- L3 group: all cores (monolithic)
- With maxGroupSize=16: 12 groups of 16 (arbitrary but contiguous)
- No hardware CCD/tile boundaries; groups provide scheduling structure

### IBM Power10 / Power11

```
Chip
  +-- Core 0 (8 SMT threads, 2 MB shared L2)
  +-- Core 1 ...
  +-- Core 15
  +-- Distributed L3 (~8 MB/core, shared across chip)
```

- L2 atom: 8 threads (SMT8, all sharing L2)
- L3 group: all cores on chip (~128 threads)
- With maxGroupSize=16: 2 L2 atoms (2 cores) per group
- With maxGroupSize=8: 1 L2 atom (1 core) per group
- SMT8 means L2 contention is inherent; linear chunk layout helps by
  keeping adjacent chunks within the shared L2

### Virtualized Environments

VMs typically flatten topology:
- L2 sharing correctly reflects SMT pairs (hypervisor preserves this)
- L3 shows all vCPUs sharing a single instance
- NUMA often shows 1 node (even if host has multiple)
- Cluster/die topology set to sentinel values (e.g., cluster_id=65535)

The algorithm degrades gracefully: with a flat L3 group, it chunks by
maxGroupSize, producing contiguous scheduling groups. This is reasonable
since the VM's vCPUs are typically backed by a contiguous range of host
cores.
