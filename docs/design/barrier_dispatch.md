# Barrier-based Static Dispatch

## Problem

For memory-bound workloads with small per-chunk compute (e.g. repeated stencil
over large arrays), dispenso's task-based dispatch is ~2x slower than OpenMP's
`schedule(static)`. TBB shows the same gap -- this is inherent to task-queue
dispatch vs fork-join, not a dispenso-specific issue.

The overhead comes from:
- Closure packaging (OnceFunction allocation + type erasure)
- Concurrent queue enqueue/dequeue (CAS operations per task)
- Per-task atomic counter updates (outstandingTaskCount_)
- Polling-based wait vs optimized barrier synchronization

### Benchmark evidence (locality_benchmark, 166 HW threads)

Repeated 3-point stencil, 10 passes, 32M doubles (~256MB per array):

| Threads | OMP (static) | TBB | dispenso (kStatic) | dispenso (kAuto) |
|---------|-------------|------|-------------------|-----------------|
| 1       | 294ms       | 292ms | 287ms             | 281ms           |
| 4       | 75ms        | 104ms | 123ms             | 112ms           |
| 16      | 48ms        | 49ms  | 77ms              | 62ms            |
| 64      | 32ms        | 56ms  | 57ms              | 56ms            |
| 128     | 29ms        | 55ms  | 62ms              | 53ms            |

At 1 thread all four are equivalent, confirming the gap is purely dispatch
overhead at scale.

The effect is even more dramatic at smaller sizes where per-chunk work is
tiny (100K elements, 128 threads: OMP 491us vs dispenso_static 466us vs
dispenso_auto 1030us).

## Current dispatch path

```
parallel_for(kStatic) ->
  for each chunk:
    1. packageTask: wrap lambda in OnceFunction, atomic++ outstandingTaskCount_
    2. enqueue: CAS into moodycamel::ConcurrentQueue
    3. conditionallyWake: atomic read numSleeping_, epoch signal
  worker thread:
    4. try_dequeue: CAS from ConcurrentQueue
    5. execute closure via virtual dispatch (OnceCallable::run)
    6. atomic-- workRemaining_ (batched per 8)
  wait:
    7. poll outstandingTaskCount_ + steal from queue
```

Each chunk goes through steps 1-7. With 128 chunks x 10 passes = 1280
round-trips, each involving multiple cache-line-bouncing atomics across
128 cores.

## Proposed design: broadcast dispatch

For `parallel_for` with `kStatic` chunking, bypass the general task queue
and use a lightweight barrier-based dispatch.

### Data structures

```cpp
// Shared broadcast descriptor -- one per ThreadPool
struct BroadcastWork {
    void (*func)(size_t begin, size_t end, void* ctx);  // static trampoline
    void* ctx;                                           // captured state
    size_t total;                                        // total range size
    ssize_t numParticipants;                             // threads participating
    std::atomic<int> epoch{0};                           // signals new work
    std::atomic<int> completionCount{0};                 // barrier counter
    std::atomic<bool> claimed{false};                    // single-producer lock
};
```

### Dispatch path (parallel_for_staticImpl fast path)

```
parallel_for(kStatic) ->
  if nested (isPoolRecursive): fall back to normal task-queue path
  if !broadcast.claimed.exchange(true): fall back to normal path
  1. Write func, ctx, total, numParticipants to broadcast struct
  2. release fence
  3. epoch.fetch_add(1)             // single atomic -- signals all threads
  4. wakeN(sleeping)                // reuse existing mechanism
  5. Execute own chunk (tid=0) inline
  6. Spin on completionCount == numParticipants  (+ steal from general queue)
  7. Reset completionCount, release claimed
```

### Thread loop integration

```cpp
void ThreadPool::threadLoop(PerThreadData& data) {
    int lastEpoch = broadcast_.epoch.load(acquire);
    while (data.running()) {
        // Priority check: broadcast work available?
        int curEpoch = broadcast_.epoch.load(acquire);
        if (curEpoch != lastEpoch) {
            lastEpoch = curEpoch;
            // Compute own chunk arithmetically -- zero overhead
            size_t begin = (broadcast_.total * myTid) / broadcast_.numParticipants;
            size_t end = (broadcast_.total * (myTid + 1)) / broadcast_.numParticipants;
            broadcast_.func(begin, end, broadcast_.ctx);  // direct call
            broadcast_.completionCount.fetch_add(1, release);
            continue;  // re-check for more broadcast work
        }
        // Normal path: dequeue from general queue
        // ... existing try_dequeue logic ...
    }
}
```

### Type erasure without virtual dispatch

The broadcast function pointer uses a C-style `void(*)(size_t, size_t, void*)`
trampoline. `parallel_for` generates this at compile time:

```cpp
template <typename F>
void parallel_for_broadcast(TaskSet& tasks, size_t begin, size_t end, F&& f) {
    struct Ctx { F* func; };
    Ctx ctx{&f};
    auto trampoline = [](size_t b, size_t e, void* raw) {
        auto* c = static_cast<Ctx*>(raw);
        (*c->func)(b, e);
    };
    pool.broadcastDispatch(trampoline, &ctx, end - begin);
}
```

No OnceFunction, no allocation, no virtual dispatch. The trampoline is a
static function instantiated per call site.

### Nested parallelism

Do NOT use broadcast path when `PerPoolPerThreadInfo::isPoolRecursive(this)`
is true. Inner parallel_for would try to broadcast, but outer parallel_for's
threads are busy executing their outer chunks and won't service the broadcast.
This would deadlock.

Fall back to normal task-queue dispatch for nested cases. The existing
work-stealing in `wait()` handles this correctly.

### Concurrency with general tasks

The broadcast path is single-producer: only one parallel_for can use it at a
time. `claimed` is an atomic flag used as a try-lock:

```cpp
if (!broadcast_.claimed.exchange(true, acquire)) {
    // Someone else is using broadcast -- fall back to task queue
    return normal_dispatch(tasks, ...);
}
// ... do broadcast dispatch ...
broadcast_.claimed.store(false, release);
```

This keeps the optimization opportunistic. In the common case (single
parallel_for at a time, no nesting), it fires. When contended, it falls
back gracefully.

### Interaction with TaskSet wait/steal

The calling thread's `wait()` must still steal from the general queue while
waiting for broadcast completion. This prevents deadlock when broadcast
parallel_for is mixed with other queued work:

```cpp
// Caller after dispatching broadcast and executing own chunk:
while (broadcast_.completionCount.load(acquire) < numParticipants - 1) {
    pool.tryExecuteNext();  // steal general queue work
    // ... backoff ...
}
```

### Thread ID assignment

Each pool thread needs a stable `tid` for computing its chunk bounds. Options:
- Use the existing `PerPoolPerThreadInfo` thread-local registration to assign
  sequential IDs at thread creation
- Store tid in `PerThreadData` (already exists, just add an index field)

The caller (master thread) always takes tid=0.

## Expected impact

Should bring kStatic parallel_for close to OMP for regular, memory-bound
workloads. Per-dispatch overhead drops from O(numChunks) atomic operations
to O(1) (single epoch bump + barrier).

The remaining gap vs OMP would be:
- Barrier quality (OMP uses architecture-specific tree/butterfly barriers;
  we'd start with a simple atomic counter)
- Thread affinity (OMP runtimes pin threads to cores; dispenso doesn't)
- The epoch check in threadLoop adds one atomic load to every dequeue
  iteration (should be negligible -- same cache line, read-only when idle)

## Risks and open questions

- **Complexity**: Dual dispatch in threadLoop must not regress the common
  task-queue path. The epoch check is a single relaxed atomic load, but
  code complexity increases.
- **Fairness**: If broadcast work keeps arriving, general queue tasks could
  starve. May need a fairness counter (check general queue every N broadcast
  rounds, or after each broadcast completion).
- **Memory ordering**: The broadcast struct write must be visible before the
  epoch bump. A release fence between writing func/ctx/total and incrementing
  epoch suffices.
- **Thread count mismatch**: If some pool threads are sleeping and don't wake
  in time, the barrier would hang. Need a timeout or adaptive participant
  count based on actually-awake threads.
- **Partial pool usage**: If the user's TaskSet uses a subset of the pool's
  threads (via stealingLoadMultiplier), broadcast must respect this.

## Validation

locality_benchmark already exists and covers the target workload. Success
criteria: dispenso_static within 1.5x of OMP at 64-128 threads for the
32M stencil case (currently 2x).
