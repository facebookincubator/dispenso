# Dispenso Parallel Algorithms Design Document

## Overview

This document outlines the design for a set of parallel algorithms in dispenso that mirror
the C++ standard library algorithms but use dispenso's execution model. This provides users
with familiar interfaces while leveraging dispenso's thread pool control, work stealing,
and nested parallelism support.

## Motivation

C++17 introduced execution policies for standard algorithms (`std::execution::par`), but:

1. **No thread pool control** - Users cannot specify which thread pool to use or limit parallelism
2. **Implementation-defined behavior** - Backends vary (TBB, OpenMP, proprietary)
3. **Limited customization** - No options for chunking strategy, stealing multiplier, etc.
4. **Nested parallelism issues** - Many implementations deadlock or perform poorly
5. **C++17 requirement** - Not available to C++14 codebases

Dispenso parallel algorithms address these limitations while providing a familiar API.

## Design Goals

1. **API compatibility** - Mirror std algorithm signatures where practical
2. **Performance** - Match or exceed std::execution::par performance
3. **Composability** - Algorithms should work well together and support nesting
4. **Flexibility** - Support custom thread pools and execution options
5. **Incremental adoption** - Users can migrate algorithm-by-algorithm

## Execution Policy

### Core Types

```cpp
namespace dispenso {

/// Options controlling parallel execution
struct ParallelOptions {
  /// Maximum threads to use (0 = use pool size)
  uint32_t maxThreads = 0;

  /// Minimum items per chunk to avoid excessive overhead
  uint32_t minItemsPerChunk = 1;

  /// Whether to wait for completion (always true for algorithms with return values)
  bool wait = true;
};

/// Execution policy referencing a specific thread pool
class ParallelPolicy {
 public:
  explicit ParallelPolicy(ThreadPool& pool, ParallelOptions options = {});
  explicit ParallelPolicy(ParallelOptions options = {});  // Uses global pool

  ThreadPool& pool() const;
  const ParallelOptions& options() const;
};

/// Create a parallel execution policy using the global thread pool
inline ParallelPolicy par(ParallelOptions options = {});

/// Create a parallel execution policy using a specific thread pool
inline ParallelPolicy par(ThreadPool& pool, ParallelOptions options = {});

}  // namespace dispenso
```

### Usage Examples

```cpp
// Use global thread pool with defaults
auto sum = dispenso::reduce(dispenso::par(), v.begin(), v.end(), 0);

// Use custom thread pool
dispenso::ThreadPool myPool(4);
dispenso::for_each(dispenso::par(myPool), v.begin(), v.end(), process);

// With options
dispenso::sort(dispenso::par({.maxThreads = 2}), v.begin(), v.end());
```

## Algorithm Categories

Algorithms are grouped by implementation complexity and interdependencies.

### Tier 1: Foundation Algorithms

These provide building blocks for other algorithms and should be implemented first.

| Algorithm | Description | Building Block |
|-----------|-------------|----------------|
| `for_each` | Apply function to range | Uses `dispenso::for_each` directly |
| `for_each_n` | Apply function to n elements | Uses `dispenso::for_each_n` directly |
| `transform` | Transform range to output | Parallel iteration, no dependencies |
| `fill` | Fill range with value | Parallel iteration, no dependencies |
| `fill_n` | Fill n elements | Parallel iteration, no dependencies |

### Tier 2: Reduction Algorithms

These require combining results from parallel work.

| Algorithm | Description | Implementation Notes |
|-----------|-------------|---------------------|
| `reduce` | Reduce range with binary op | Per-thread accumulators + final reduction |
| `transform_reduce` | Transform then reduce | Fused for efficiency |
| `count` | Count elements matching value | Reduction with increment |
| `count_if` | Count elements matching predicate | Reduction with conditional increment |

### Tier 3: Search Algorithms

These may terminate early and require coordination.

| Algorithm | Description | Implementation Notes |
|-----------|-------------|---------------------|
| `find` | Find first matching element | Parallel search with early termination |
| `find_if` | Find first element matching predicate | Parallel search with early termination |
| `find_if_not` | Find first element not matching | Parallel search with early termination |
| `any_of` | Check if any element matches | Parallel search, return on first match |
| `all_of` | Check if all elements match | Parallel search, return on first mismatch |
| `none_of` | Check if no elements match | Parallel search, return on first match |

### Tier 4: Mutating Algorithms

These modify the input range and may have ordering constraints.

| Algorithm | Description | Implementation Notes |
|-----------|-------------|---------------------|
| `copy` | Copy range | Parallel iteration |
| `copy_if` | Copy elements matching predicate | Parallel filter, requires output coordination |
| `move` | Move range | Parallel iteration |
| `replace` | Replace matching values | Parallel iteration |
| `replace_if` | Replace values matching predicate | Parallel iteration |

### Tier 5: Ordering Algorithms

These have complex parallelization patterns.

| Algorithm | Description | Implementation Notes |
|-----------|-------------|---------------------|
| `sort` | Sort range | Parallel quicksort or mergesort |
| `stable_sort` | Stable sort | Parallel mergesort |
| `partial_sort` | Partial sort | Parallel selection + sort |
| `nth_element` | Nth element partitioning | Parallel partitioning |
| `unique` | Remove consecutive duplicates | Parallel compaction (complex) |

### Tier 6: Numeric Algorithms

From `<numeric>`, often used in scientific computing.

| Algorithm | Description | Implementation Notes |
|-----------|-------------|---------------------|
| `inclusive_scan` | Inclusive prefix sum | Parallel scan algorithm |
| `exclusive_scan` | Exclusive prefix sum | Parallel scan algorithm |
| `transform_inclusive_scan` | Transform + inclusive scan | Fused operation |
| `transform_exclusive_scan` | Transform + exclusive scan | Fused operation |

## Implementation Strategy

### When to Use Existing Dispenso Primitives

**Use `dispenso::parallel_for` when:**
- Work is embarrassingly parallel (no cross-iteration dependencies)
- Output location is determined by input index
- Examples: `transform`, `fill`, `replace`

**Use `dispenso::for_each` when:**
- Iterating over non-random-access containers
- Work per element is the goal (side effects)
- Examples: `for_each` (direct mapping)

### When to Use Custom Implementation

**Custom implementation needed when:**
- Early termination is beneficial (`find`, `any_of`)
- Results must be combined across threads (`reduce`, `count`)
- Complex coordination required (`sort`, `unique`)
- Memory access patterns benefit from custom chunking

### Reduction Pattern

For algorithms that combine results (reduce, count, etc.):

```cpp
template<typename Policy, typename It, typename T, typename BinaryOp>
T reduce(Policy&& policy, It first, It last, T init, BinaryOp op) {
  const auto n = std::distance(first, last);
  if (n == 0) return init;

  ThreadPool& pool = policy.pool();
  const auto numThreads = std::min<size_t>(
      pool.numThreads() + 1,  // +1 for calling thread
      policy.options().maxThreads ? policy.options().maxThreads : pool.numThreads() + 1);

  // Per-thread partial results
  std::vector<T> partials;

  dispenso::parallel_for(
      partials,
      [&init]() { return init; },  // Initialize each partial with identity
      size_t{0}, static_cast<size_t>(n),
      [&](T& partial, size_t start, size_t end) {
        auto it = first;
        std::advance(it, start);
        for (size_t i = start; i < end; ++i, ++it) {
          partial = op(partial, *it);
        }
      });

  // Combine partials
  T result = init;
  for (const T& p : partials) {
    result = op(result, p);
  }
  return result;
}
```

### Early Termination Pattern

For search algorithms that can stop early:

```cpp
template<typename Policy, typename It, typename Pred>
It find_if(Policy&& policy, It first, It last, Pred pred) {
  const auto n = std::distance(first, last);
  if (n == 0) return last;

  std::atomic<ssize_t> foundIndex{-1};

  dispenso::parallel_for(
      policy.pool(),
      size_t{0}, static_cast<size_t>(n),
      [&](size_t i) {
        // Skip if already found earlier
        ssize_t found = foundIndex.load(std::memory_order_relaxed);
        if (found >= 0 && static_cast<ssize_t>(i) > found) {
          return;
        }

        auto it = first;
        std::advance(it, i);
        if (pred(*it)) {
          // Atomically update to earliest found
          ssize_t expected = foundIndex.load(std::memory_order_relaxed);
          while ((expected < 0 || static_cast<ssize_t>(i) < expected) &&
                 !foundIndex.compare_exchange_weak(expected, i)) {
          }
        }
      });

  ssize_t found = foundIndex.load();
  if (found < 0) return last;

  auto result = first;
  std::advance(result, found);
  return result;
}
```

Note: The above is a simple implementation. A production version would use chunking
for better cache behavior and reduced atomic contention.

## API Signatures

### Compatibility with std

Where possible, signatures should match std exactly (minus the execution policy):

```cpp
// std signature:
template<class ExecutionPolicy, class ForwardIt, class T>
T reduce(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, T init);

// dispenso signature:
template<class ForwardIt, class T>
T reduce(ParallelPolicy policy, ForwardIt first, ForwardIt last, T init);
```

### Signature Variations to Support

For each algorithm, support all standard overloads where practical:

```cpp
// reduce overloads
template<class It>
typename std::iterator_traits<It>::value_type
reduce(ParallelPolicy policy, It first, It last);

template<class It, class T>
T reduce(ParallelPolicy policy, It first, It last, T init);

template<class It, class T, class BinaryOp>
T reduce(ParallelPolicy policy, It first, It last, T init, BinaryOp op);
```

### Deferred Signatures (TODO)

Some overloads may be deferred if they add significant complexity:

- Iterator sentinel overloads (C++20 ranges)
- Projections (C++20)
- Overloads requiring complex type deduction

## Header Organization

```
dispenso/
├── algorithm.h              // Main header, includes all algorithms
├── algorithm/
│   ├── for_each.h          // for_each, for_each_n
│   ├── transform.h         // transform (unary and binary)
│   ├── fill.h              // fill, fill_n
│   ├── reduce.h            // reduce, transform_reduce
│   ├── count.h             // count, count_if
│   ├── find.h              // find, find_if, find_if_not
│   ├── predicates.h        // any_of, all_of, none_of
│   ├── copy.h              // copy, copy_if, copy_n
│   ├── replace.h           // replace, replace_if
│   ├── sort.h              // sort, stable_sort
│   └── scan.h              // inclusive_scan, exclusive_scan
└── execution_policy.h       // ParallelPolicy, ParallelOptions, par()
```

## Performance Considerations

### Chunking Strategy

- Default chunk size should amortize scheduling overhead
- For small workloads, fall back to sequential execution
- Allow user override via `ParallelOptions::minItemsPerChunk`

### Memory Access Patterns

- Prefer sequential memory access within chunks for cache efficiency
- For random-access iterators, use index-based chunking
- For forward iterators, pre-compute chunk boundaries

### Avoiding False Sharing

- Per-thread accumulators should be cache-line aligned
- Use `dispenso::ConcurrentVector` for parallel output when needed

### Nested Parallelism

- Algorithms should work correctly when called from within parallel regions
- Rely on dispenso's work-stealing to prevent deadlock

### SIMD Optimization Opportunities

For certain algorithms with contiguous memory and simple element types, SIMD vectorization
can provide significant additional performance gains beyond thread-level parallelism:

**High SIMD benefit:**
- `fill`, `fill_n` - Vectorized stores
- `copy`, `copy_n` - Vectorized loads/stores (or memcpy for trivially copyable types)
- `transform` - When operation maps to SIMD intrinsics (e.g., arithmetic, min/max)
- `reduce`, `transform_reduce` - Vectorized horizontal operations
- `min_element`, `max_element`, `minmax_element` - SIMD comparisons
- `count`, `count_if` - SIMD comparison + popcount
- `equal`, `mismatch` - SIMD comparison
- `find`, `find_if` - SIMD search with early termination

**Implementation considerations:**
- Require contiguous iterators (pointers or types satisfying contiguous_iterator)
- Restrict to arithmetic types, pointers, or types with known SIMD-friendly operations
- Consider platform-specific implementations (SSE, AVX, NEON) or rely on compiler auto-vectorization
- May provide specializations for common cases (e.g., `float`, `double`, `int32_t`)
- Chunk boundaries should be SIMD-aligned where beneficial

## Testing Strategy

Each algorithm should have tests for:

1. **Correctness** - Compare results to sequential std:: version
2. **Empty range** - Handle begin == end
3. **Single element** - Degenerate case
4. **Large range** - Verify parallelization occurs
5. **Custom thread pool** - Non-global pool works correctly
6. **Options** - maxThreads, minItemsPerChunk respected
7. **Iterator types** - Random access, bidirectional, forward (where applicable)
8. **Nested calls** - Algorithm called from within parallel_for

## Implementation Phases

### Phase 1: Foundation (Recommended First)

1. `ParallelPolicy` and `par()` execution policy
2. `for_each` / `for_each_n` (wraps existing dispenso::for_each)
3. `transform` (unary version)
4. `fill` / `fill_n`
5. `reduce` (all overloads)
6. `transform_reduce`

### Phase 2: Search and Count

1. `count` / `count_if`
2. `find` / `find_if` / `find_if_not`
3. `any_of` / `all_of` / `none_of`

### Phase 3: Mutating

1. `copy` / `copy_n`
2. `copy_if`
3. `replace` / `replace_if`
4. `transform` (binary version)

### Phase 4: Sorting

1. `sort`
2. `stable_sort`
3. `partial_sort`
4. `nth_element`

### Phase 5: Advanced

1. `inclusive_scan` / `exclusive_scan`
2. `transform_inclusive_scan` / `transform_exclusive_scan`
3. `unique`

## Complete Algorithm Inventory

This section tracks all C++ standard library algorithms that support parallel execution policies.
Algorithms are categorized by priority based on their value in a parallel context.

### Priority Definitions

- **High**: Algorithms that benefit significantly from parallelization and are commonly used
- **Medium**: Algorithms with moderate parallel benefit or less common use cases
- **Low**: Algorithms where parallelization provides minimal benefit or adds unnecessary overhead

### High Priority Algorithms

| Algorithm | Status | Notes |
|-----------|--------|-------|
| `for_each` | Planned (Phase 1) | Core building block |
| `for_each_n` | Planned (Phase 1) | Core building block |
| `transform` | Planned (Phase 1) | Embarrassingly parallel |
| `reduce` | Planned (Phase 1) | Common reduction pattern |
| `transform_reduce` | Planned (Phase 1) | Fused transform + reduce |
| `sort` | Planned (Phase 4) | High-value, well-understood parallel algorithms |
| `stable_sort` | Planned (Phase 4) | Parallel mergesort |
| `fill` | Planned (Phase 1) | Simple parallel write |
| `fill_n` | Planned (Phase 1) | Simple parallel write |
| `copy` | Planned (Phase 3) | Parallel memory copy |
| `copy_n` | Planned (Phase 3) | Parallel memory copy |
| `inclusive_scan` | Planned (Phase 5) | Prefix sum, scientific computing |
| `exclusive_scan` | Planned (Phase 5) | Prefix sum, scientific computing |
| `transform_exclusive_scan` | Planned (Phase 5) | Fused operation |
| `transform_inclusive_scan` | Planned (Phase 5) | Fused operation |
| `count` | Planned (Phase 2) | Parallel reduction |
| `count_if` | Planned (Phase 2) | Parallel reduction with predicate |
| `find` | Planned (Phase 2) | Parallel search with early termination |
| `find_if` | Planned (Phase 2) | Parallel search with early termination |
| `find_if_not` | Planned (Phase 2) | Parallel search with early termination |
| `any_of` | Planned (Phase 2) | Parallel short-circuit evaluation |
| `all_of` | Planned (Phase 2) | Parallel short-circuit evaluation |
| `none_of` | Planned (Phase 2) | Parallel short-circuit evaluation |
| `inner_product` | Not yet planned | Equivalent to transform_reduce |
| `adjacent_difference` | Not yet planned | Parallel stencil operation |
| `max_element` | Not yet planned | Parallel reduction to find max |
| `min_element` | Not yet planned | Parallel reduction to find min |
| `minmax_element` | Not yet planned | Parallel reduction to find both |

### Medium Priority Algorithms

| Algorithm | Status | Notes |
|-----------|--------|-------|
| `copy_if` | Planned (Phase 3) | Requires output coordination |
| `replace` | Planned (Phase 3) | Simple parallel mutation |
| `replace_if` | Planned (Phase 3) | Simple parallel mutation |
| `replace_copy` | Not yet planned | Parallel copy with replacement |
| `replace_copy_if` | Not yet planned | Parallel copy with conditional replacement |
| `move` | Planned (Phase 3) | Parallel move semantics |
| `partial_sort` | Planned (Phase 4) | Parallel selection + sort |
| `partial_sort_copy` | Not yet planned | Parallel partial sort to output |
| `nth_element` | Planned (Phase 4) | Parallel partitioning |
| `partition` | Not yet planned | Parallel partitioning |
| `stable_partition` | Not yet planned | Stable parallel partitioning |
| `partition_copy` | Not yet planned | Parallel partition to two outputs |
| `unique` | Planned (Phase 5) | Complex parallel compaction |
| `unique_copy` | Not yet planned | Parallel unique to output |
| `merge` | Not yet planned | Parallel merge of sorted ranges |
| `inplace_merge` | Not yet planned | In-place parallel merge |
| `set_union` | Not yet planned | Parallel set operation |
| `set_intersection` | Not yet planned | Parallel set operation |
| `set_difference` | Not yet planned | Parallel set operation |
| `set_symmetric_difference` | Not yet planned | Parallel set operation |
| `generate` | Not yet planned | Parallel generation (if generator is thread-safe) |
| `generate_n` | Not yet planned | Parallel generation (if generator is thread-safe) |
| `remove` | Not yet planned | Parallel compaction |
| `remove_if` | Not yet planned | Parallel compaction with predicate |
| `remove_copy` | Not yet planned | Parallel filtered copy |
| `remove_copy_if` | Not yet planned | Parallel filtered copy with predicate |
| `reverse` | Not yet planned | Parallel in-place reversal |
| `reverse_copy` | Not yet planned | Parallel reverse to output |
| `rotate` | Not yet planned | Parallel rotation |
| `rotate_copy` | Not yet planned | Parallel rotate to output |
| `swap_ranges` | Not yet planned | Parallel swap |
| `uninitialized_copy` | Not yet planned | Parallel uninitialized memory copy |
| `uninitialized_copy_n` | Not yet planned | Parallel uninitialized memory copy |
| `uninitialized_fill` | Not yet planned | Parallel uninitialized fill |
| `uninitialized_fill_n` | Not yet planned | Parallel uninitialized fill |
| `equal` | Not yet planned | Parallel comparison with early termination |
| `mismatch` | Not yet planned | Parallel search for first difference |
| `is_sorted` | Not yet planned | Parallel scan; less common with large sequences |
| `is_sorted_until` | Not yet planned | Parallel scan; less common with large sequences |
| `is_partitioned` | Not yet planned | Parallel scan; less common with large sequences |
| `lexicographical_compare` | Not yet planned | Parallel comparison; less common with large sequences |
| `includes` | Not yet planned | Parallel subset check |
| `search` | Not yet planned | Parallel subsequence search |
| `search_n` | Not yet planned | Parallel consecutive value search |

### Low Priority Algorithms

These algorithms have limited benefit from parallelization due to their nature
(e.g., heap operations have inherent sequential dependencies, or the operations
are too simple to justify parallel overhead in most cases).

| Algorithm | Status | Notes |
|-----------|--------|-------|
| `is_heap` | Not yet planned | Heap structure has sequential dependencies |
| `is_heap_until` | Not yet planned | Heap structure has sequential dependencies |
| `adjacent_find` | Not yet planned | Requires checking adjacent pairs, limited benefit |
| `find_end` | Not yet planned | Searching from end, complex parallelization |
| `find_first_of` | Not yet planned | Multi-target search, moderate complexity |

### Algorithm Status Legend

- **Planned (Phase N)**: Included in implementation roadmap
- **Not yet planned**: Identified but not scheduled; may be added based on demand
- **Implemented**: Available in dispenso (update when completed)
- **Deferred**: Explicitly deferred due to complexity or low value

## Open Questions

1. **Namespace**: Should algorithms live in `dispenso::` or a sub-namespace like `dispenso::alg::`?

2. **Sequential fallback**: Should there be automatic fallback to sequential for small inputs?
   Recommended: Yes, with a reasonable threshold (e.g., < 1000 elements or < numThreads chunks).

3. **Exception handling**: How should exceptions from user functors be handled?
   Recommended: Follow std behavior - propagate one exception, others may be lost.

4. **Allocator support**: Should algorithms that need temporary storage accept allocators?
   Recommended: Defer to later; use dispenso's internal allocation initially.

## References

- [C++ Standard Algorithms Library](https://en.cppreference.com/w/cpp/algorithm)
- [C++17 Execution Policies](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag)
- [P0024R2 - The Parallelism TS Should be Standardized](https://wg21.link/p0024r2)

## Implementation Checklist

Algorithms in implementation order. Check off as completed.

### Phase 1: Foundation
- [ ] `for_each`
- [ ] `for_each_n`
- [ ] `transform`
- [ ] `fill`
- [ ] `fill_n`
- [ ] `reduce`
- [ ] `transform_reduce`

### Phase 2: Search and Count
- [ ] `count`
- [ ] `count_if`
- [ ] `find`
- [ ] `find_if`
- [ ] `find_if_not`
- [ ] `any_of`
- [ ] `all_of`
- [ ] `none_of`

### Phase 3: Mutating
- [ ] `copy`
- [ ] `copy_n`
- [ ] `copy_if`
- [ ] `move`
- [ ] `replace`
- [ ] `replace_if`

### Phase 4: Sorting
- [ ] `sort`
- [ ] `stable_sort`
- [ ] `partial_sort`
- [ ] `nth_element`

### Phase 5: Advanced
- [ ] `inclusive_scan`
- [ ] `exclusive_scan`
- [ ] `transform_inclusive_scan`
- [ ] `transform_exclusive_scan`
- [ ] `unique`

### Future (High Priority)
- [ ] `min_element`
- [ ] `max_element`
- [ ] `minmax_element`
- [ ] `inner_product`
- [ ] `adjacent_difference`

### Future (Medium Priority)
- [ ] `replace_copy`
- [ ] `replace_copy_if`
- [ ] `partial_sort_copy`
- [ ] `partition`
- [ ] `stable_partition`
- [ ] `partition_copy`
- [ ] `unique_copy`
- [ ] `merge`
- [ ] `inplace_merge`
- [ ] `set_union`
- [ ] `set_intersection`
- [ ] `set_difference`
- [ ] `set_symmetric_difference`
- [ ] `generate`
- [ ] `generate_n`
- [ ] `remove`
- [ ] `remove_if`
- [ ] `remove_copy`
- [ ] `remove_copy_if`
- [ ] `reverse`
- [ ] `reverse_copy`
- [ ] `rotate`
- [ ] `rotate_copy`
- [ ] `swap_ranges`
- [ ] `uninitialized_copy`
- [ ] `uninitialized_copy_n`
- [ ] `uninitialized_fill`
- [ ] `uninitialized_fill_n`
- [ ] `equal`
- [ ] `mismatch`
- [ ] `is_sorted`
- [ ] `is_sorted_until`
- [ ] `is_partitioned`
- [ ] `lexicographical_compare`
- [ ] `includes`
- [ ] `search`
- [ ] `search_n`

### Future (Low Priority)
- [ ] `is_heap`
- [ ] `is_heap_until`
- [ ] `adjacent_find`
- [ ] `find_end`
- [ ] `find_first_of`
