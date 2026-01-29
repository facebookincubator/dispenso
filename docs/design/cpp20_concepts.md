# C++20 Concepts for Dispenso

## Overview

This document outlines a plan to add C++20 concept constraints to dispenso's template APIs.
The goal is to provide better error messages when template requirements are not met, while
maintaining full backward compatibility with C++14/17.

## Motivation

Current template errors in dispenso can be cryptic. For example, passing a non-callable
to `parallel_for` produces deep template instantiation errors. With concepts, users get
clear messages like "constraint not satisfied: F must be invocable with (size_t, size_t)".

## Design Principles

1. **Backward compatible** - All code that compiles with C++14/17 continues to work
2. **Graceful degradation** - Concepts only active when `__cpp_concepts >= 201907L`
3. **Incremental adoption** - Can add concepts to APIs one at a time
4. **Documentation value** - Concepts serve as executable documentation of requirements

## Implementation Strategy

### Feature Detection Macro

```cpp
// In dispenso/platform.h or a new dispenso/concepts.h

#if __cplusplus >= 202002L && defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define DISPENSO_HAS_CONCEPTS 1
#else
#define DISPENSO_HAS_CONCEPTS 0
#endif
```

### Concept Definitions

```cpp
// dispenso/concepts.h (new header)

#pragma once

#include <dispenso/platform.h>

#if DISPENSO_HAS_CONCEPTS

#include <concepts>
#include <type_traits>

namespace dispenso {

/// A type that can be invoked with the given arguments
template<typename F, typename... Args>
concept Invocable = std::invocable<F, Args...>;

/// A type that can be invoked and returns a specific type
template<typename F, typename R, typename... Args>
concept InvocableR = std::invocable<F, Args...> &&
    std::convertible_to<std::invoke_result_t<F, Args...>, R>;

/// A callable suitable for parallel_for with index range
template<typename F>
concept ParallelForCallable = Invocable<F, size_t, size_t>;

/// A callable suitable for parallel_for with single index
template<typename F>
concept ParallelForIndexCallable = Invocable<F, size_t>;

/// A callable suitable for parallel_for with state
template<typename F, typename State>
concept ParallelForStateCallable = Invocable<F, State&, size_t, size_t>;

/// A callable suitable for TaskSet::schedule
template<typename F>
concept TaskCallable = Invocable<F> && std::is_void_v<std::invoke_result_t<F>>;

/// A state factory for parallel_for
template<typename F, typename State>
concept StateFactory = Invocable<F> &&
    std::convertible_to<std::invoke_result_t<F>, State>;

/// A binary operation for reduce
template<typename Op, typename T>
concept BinaryOp = Invocable<Op, T, T> &&
    std::convertible_to<std::invoke_result_t<Op, T, T>, T>;

/// Iterator that supports random access (needed for efficient parallel iteration)
template<typename It>
concept RandomAccessIterator = std::random_access_iterator<It>;

/// Iterator that is at least forward (minimum for parallel algorithms)
template<typename It>
concept ForwardIterator = std::forward_iterator<It>;

}  // namespace dispenso

#endif  // DISPENSO_HAS_CONCEPTS
```

### Applying Concepts to APIs

Use `requires` clauses that are conditionally compiled:

```cpp
// Example: parallel_for

#if DISPENSO_HAS_CONCEPTS
template <typename F>
  requires ParallelForCallable<F>
#else
template <typename F>
#endif
void parallel_for(size_t begin, size_t end, F&& f);
```

Or use a macro for cleaner syntax:

```cpp
// In platform.h or concepts.h
#if DISPENSO_HAS_CONCEPTS
#define DISPENSO_REQUIRES(...) requires (__VA_ARGS__)
#else
#define DISPENSO_REQUIRES(...)
#endif

// Usage
template <typename F>
DISPENSO_REQUIRES(ParallelForCallable<F>)
void parallel_for(size_t begin, size_t end, F&& f);
```

## Priority APIs for Concept Constraints

### High Priority (Most User-Facing)

| API | Concept Constraint |
|-----|-------------------|
| `parallel_for` (all overloads) | `ParallelForCallable`, `ParallelForIndexCallable`, `ParallelForStateCallable` |
| `TaskSet::schedule` | `TaskCallable` |
| `Future::then` | `Invocable<F, T>` where T is the Future's value type |
| `for_each` | `Invocable<F, T&>` where T is the element type |
| `Pipeline` stage functions | Various invocable constraints |

### Medium Priority

| API | Concept Constraint |
|-----|-------------------|
| `Graph` node functions | `TaskCallable` |
| `ConcurrentVector` operations | Element type constraints |
| `reduce` (when implemented) | `BinaryOp` |

### Lower Priority

| API | Concept Constraint |
|-----|-------------------|
| Allocator APIs | Standard allocator concepts |
| Internal utilities | Not user-facing, lower value |

## Example Error Messages

### Before (C++17)

```
error: no matching function for call to 'parallel_for'
note: candidate template ignored: substitution failure [with F = int]:
      no type named 'type' in 'std::invoke_result<int, unsigned long, unsigned long>'
```

### After (C++20 with concepts)

```
error: constraints not satisfied for 'parallel_for<int>'
note: because 'int' does not satisfy 'ParallelForCallable'
note: because 'std::invocable<int, size_t, size_t>' evaluated to false
```

## Testing Strategy

1. **Positive tests** - Verify valid code compiles with concepts enabled
2. **Negative tests** - Verify invalid code produces expected concept errors (manual verification)
3. **Cross-version tests** - Same tests pass on C++14, C++17, and C++20
4. **CI matrix** - Test with multiple compilers (GCC, Clang, MSVC) and standards

## Implementation Phases

### Phase 1: Infrastructure
- [ ] Add `DISPENSO_HAS_CONCEPTS` macro to platform.h
- [ ] Create `dispenso/concepts.h` with core concept definitions
- [ ] Add `DISPENSO_REQUIRES` macro

### Phase 2: Core APIs
- [ ] `parallel_for` (all overloads)
- [ ] `TaskSet::schedule`
- [ ] `for_each`

### Phase 3: Future and Async
- [ ] `Future::then`
- [ ] `Future::whenAll`
- [ ] `async`

### Phase 4: Advanced APIs
- [ ] `Pipeline` stage functions
- [ ] `Graph` node functions
- [ ] Parallel algorithms (as they are implemented)

## Compatibility Notes

- GCC 10+ supports concepts
- Clang 10+ supports concepts
- MSVC 19.28+ (VS 2019 16.8+) supports concepts
- Older compilers will simply not get the concept constraints

## Open Questions

1. Should we provide our own concept definitions or rely on `<concepts>` header?
   - Recommendation: Define our own in terms of std concepts for customization

2. Should concepts be in a sub-namespace?
   - Recommendation: No, keep in `dispenso::` for simplicity

3. Should we add concepts to internal/detail APIs?
   - Recommendation: No, focus on public APIs for maximum user benefit
