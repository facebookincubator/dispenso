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
| C++17 minimum | Bump minimum standard from C++14 to C++17. Enables `std::optional` (replacing `OpResult`), `if constexpr`, structured bindings, `[[nodiscard]]`, and `std::string_view`. Simplifies template metaprogramming throughout. |
| C++20 consideration | Evaluate C++20 as minimum for a later 2.x release. Enables `std::atomic<T>::wait/notify` (potential EpochWaiter simplification), concepts (better error messages), coroutine integration, and `std::jthread`. |
