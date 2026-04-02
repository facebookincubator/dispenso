# Dispenso Benchmark Results

## Machine Information

- **Date**: 2026-02-05T15:59:21.769268
- **Platform**: Linux 6.17.4-200.fc42.x86_64
- **CPU**: AMD Ryzen Threadripper PRO 7995WX 96-Cores
- **Hardware Threads**: 192
- **Memory**: 250.9 GB

## Results Summary

- **Benchmarks run**: 18
- **Successful**: 18
- **Failed**: 0

### concurrent_vector

**Serial/Access Operations:**

![concurrent_vector serial](concurrent_vector_serial_chart.png)

**Parallel Operations:**

![concurrent_vector parallel](concurrent_vector_parallel_chart.png)

[View detailed results table](concurrent_vector_details.md)

### for_latency

**default elements:**

![for_latency default](for_latency_default_chart.png)

[View detailed results table](for_latency_details.md)

### future

**Full comparison (including std::async):**

![future results](future_chart.png)

**Zoomed (excluding std::async):**

![future zoomed](future_zoomed_chart.png)

[View detailed results table](future_details.md)

### graph

![graph results](graph_chart.png)

[View detailed results table](graph_details.md)

### graph_scene

![graph_scene results](graph_scene_chart.png)

[View detailed results table](graph_scene_details.md)

### idle_pool

**default elements:**

![idle_pool default](idle_pool_default_chart.png)

**1K elements:**

![idle_pool 1K](idle_pool_1000_chart.png)

**10K elements:**

![idle_pool 10K](idle_pool_10000_chart.png)

**1M elements:**

![idle_pool 1M](idle_pool_1000000_chart.png)

[View detailed results table](idle_pool_details.md)

### nested_for

**10 elements:**

![nested_for 10](nested_for_10_chart.png)

**10 elements (Y-Axis Zoomed):**

![nested_for 10 zoomed](nested_for_10_zoomed_chart.png)

**500 elements:**

![nested_for 500](nested_for_500_chart.png)

**500 elements (Y-Axis Zoomed):**

![nested_for 500 zoomed](nested_for_500_zoomed_chart.png)

**3K elements:**

![nested_for 3K](nested_for_3000_chart.png)

**3K elements (Y-Axis Zoomed):**

![nested_for 3K zoomed](nested_for_3000_zoomed_chart.png)

[View detailed results table](nested_for_details.md)

### nested_pool

**1K elements:**

![nested_pool 1K](nested_pool_1000_chart.png)

**10K elements:**

![nested_pool 10K](nested_pool_10000_chart.png)

**1M elements:**

![nested_pool 1M](nested_pool_1000000_chart.png)

*Note: folly::CPUThreadPoolExecutor is excluded from the 1M chart as it fails to complete (likely due to memory exhaustion from creating too many futures).*

[View detailed results table](nested_pool_details.md)

### once_function

**Move Operations:**

![once_function move](once_function_move_chart.png)

**Queue Operations:**

![once_function queue](once_function_queue_chart.png)

[View detailed results table](once_function_details.md)

### pipeline

![pipeline results](pipeline_chart.png)

[View detailed results table](pipeline_details.md)

### pool_allocator

**Single-threaded:**

![pool_allocator 1t](pool_allocator_1t_chart.png)

**2 Threads:**

![pool_allocator 2t](pool_allocator_2t_chart.png)

**8 Threads:**

![pool_allocator 8t](pool_allocator_8t_chart.png)

**16 Threads:**

![pool_allocator 16t](pool_allocator_16t_chart.png)

[View detailed results table](pool_allocator_details.md)

### rw_lock

**Serial Operations:**

![rw_lock serial](rw_lock_serial_chart.png)

**Parallel Operations:**

**2 Iterations:**

![rw_lock parallel 2](rw_lock_parallel_2_chart.png)

**8 Iterations:**

![rw_lock parallel 8](rw_lock_parallel_8_chart.png)

**32 Iterations:**

![rw_lock parallel 32](rw_lock_parallel_32_chart.png)

**128 Iterations:**

![rw_lock parallel 128](rw_lock_parallel_128_chart.png)

**512 Iterations:**

![rw_lock parallel 512](rw_lock_parallel_512_chart.png)

[View detailed results table](rw_lock_details.md)

### simple_for

**1K elements:**

![simple_for 1K](simple_for_1000_chart.png)

**1K elements (Y-Axis Zoomed):**

![simple_for 1K zoomed](simple_for_1000_zoomed_chart.png)

**1M elements:**

![simple_for 1M](simple_for_1000000_chart.png)

**1M elements (Y-Axis Zoomed):**

![simple_for 1M zoomed](simple_for_1000000_zoomed_chart.png)

**100M elements:**

![simple_for 100M](simple_for_100000000_chart.png)

**100M elements (Y-Axis Zoomed):**

![simple_for 100M zoomed](simple_for_100000000_zoomed_chart.png)

[View detailed results table](simple_for_details.md)

### simple_pool

**1K elements:**

![simple_pool 1K](simple_pool_1000_chart.png)

**10K elements:**

![simple_pool 10K](simple_pool_10000_chart.png)

**1M elements:**

![simple_pool 1M](simple_pool_1000000_chart.png)

[View detailed results table](simple_pool_details.md)

### small_buffer

![small_buffer results](small_buffer_chart.png)

[View detailed results table](small_buffer_details.md)

### summing_for

**1K elements:**

![summing_for 1K](summing_for_1000_chart.png)

**1K elements (Y-Axis Zoomed):**

![summing_for 1K zoomed](summing_for_1000_zoomed_chart.png)

**1M elements:**

![summing_for 1M](summing_for_1000000_chart.png)

**1M elements (Y-Axis Zoomed):**

![summing_for 1M zoomed](summing_for_1000000_zoomed_chart.png)

**100M elements:**

![summing_for 100M](summing_for_100000000_chart.png)

**100M elements (Y-Axis Zoomed):**

![summing_for 100M zoomed](summing_for_100000000_zoomed_chart.png)

[View detailed results table](summing_for_details.md)

### timed_task

*Visualization not available for this benchmark.*

[View detailed results table](timed_task_details.md)

### trivial_compute

**100 elements:**

![trivial_compute 100](trivial_compute_100_chart.png)

**100 elements (Y-Axis Zoomed):**

![trivial_compute 100 zoomed](trivial_compute_100_zoomed_chart.png)

**1M elements:**

![trivial_compute 1M](trivial_compute_1000000_chart.png)

**1M elements (Y-Axis Zoomed):**

![trivial_compute 1M zoomed](trivial_compute_1000000_zoomed_chart.png)

**100M elements:**

![trivial_compute 100M](trivial_compute_100000000_chart.png)

**100M elements (Y-Axis Zoomed):**

![trivial_compute 100M zoomed](trivial_compute_100000000_zoomed_chart.png)

[View detailed results table](trivial_compute_details.md)

