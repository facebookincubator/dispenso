<!-- Part of the dispenso roadmap; index at ../roadmap.md -->

# Roadmap: language and API surface

C++ standard level, concepts, coroutines, and API-breaking changes.

## Planned

| Feature | Description | Priority | Doc |
|---------|-------------|----------|-----|
| C++20 concepts | Better error messages with concept constraints | High | [cpp20_concepts](../proposals/cpp20_concepts.md) |
| Barrier/Semaphore | C++20-style synchronization for C++14/17 | Medium | - |
| Coroutine integration | Coroutine-based task scheduling | Lower | [coroutines](../proposals/coroutines.md) |
| Single-header amalgamation | Full library in one header | Lower | - |

## 2.0 (API-breaking changes)

The following changes require a major version bump due to API breakage:

| Feature | Description |
|---------|-------------|
| Remove poll mode | Remove `threadLoopPoll`, the `setSignalingWake(false)` path, and the `DISPENSO_WAKEUP_ENABLE` / `DISPENSO_POLL_PERIOD_US` defines. Signaling wake (futex/WaitOnAddress/ulock) is always-on and well-tested. Poll mode is a legacy fallback that adds code complexity without benefit on any supported platform. |
| C++17 minimum | Bump minimum standard from C++14 to C++17. Enables `std::optional` (replacing `OpResult`), `if constexpr`, structured bindings, `[[nodiscard]]`, and `std::string_view`. Simplifies template metaprogramming throughout, and retires dispenso's hand-written over-aligned allocation (see Code TODOs below). |
| C++20 consideration | Evaluate C++20 as minimum for a later 2.x release. Enables `std::atomic<T>::wait/notify` (potential EpochWaiter simplification), concepts (better error messages), coroutine integration, and `std::jthread`. |

## Backlog

### Code TODOs

Open `TODO` comments in the source, each named by file and function. See
[the README](../README.md#conventions) for how these are kept in step.

- **Launch policies for `dispenso::async`** (`dispenso/future.h`). Decide whether to extend the launch policies, building on `std::launch`, and whether `std::launch::async` should mean a new thread. Today it means `ForceQueuingTag`.
- **Over-aligned allocation after C++17.** Before C++17, `new T` and `std::allocator<T>` ignore over-alignment, so dispenso allocates over-aligned types by hand. With C++17 as the minimum (the 2.0 row above), each of these becomes plain `new`, `std::make_unique`, `std::make_shared` or `std::allocator`:
  - `detail::AlignedAllocator` (`dispenso/platform.h`; code TODO), used for `ThreadPool::threads_` and in two tests.
  - `SmallVector::allocate` and `deallocate` (`dispenso/small_vector.h`; code TODO).
  - `ThreadPool`'s class `operator new` and `operator delete` (`dispenso/thread_pool.h`), already compiled only below C++17.
  - `detail::makeAligned`, `detail::makeAlignedArray` and `detail::make_shared` (`dispenso/platform.h`), and the `alignof(T)` placement-new allocations in `detail/pipeline_impl.h`, `graph.cpp` and `detail/concurrent_vector_impl.h`.
  - The C++14 note on `CacheAligned` (`dispenso/platform.h`).

  Allocations aligned beyond their type's own alignment stay: the small buffer allocator's chunks and blocks, arena buffers, `parallel_for`'s scratch buffers and `ResourcePool`'s padded slots. So does the public `dispenso::alignedMalloc`.
