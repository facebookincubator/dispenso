/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file dispenso.h
 * @brief Convenience header that includes all public dispenso headers.
 *
 * For finer-grained control over compile times, include individual headers instead.
 **/

#pragma once

// Core threading primitives
#include <dispenso/future.h>
#include <dispenso/schedulable.h>
#include <dispenso/task_set.h>
#include <dispenso/thread_pool.h>

// Parallel algorithms
#include <dispenso/for_each.h>
#include <dispenso/parallel_for.h>
#include <dispenso/pipeline.h>

// Graph-based task scheduling
#include <dispenso/graph.h>
#include <dispenso/graph_executor.h>

// Concurrent containers
#include <dispenso/concurrent_object_arena.h>
#include <dispenso/concurrent_vector.h>

// Synchronization primitives
#include <dispenso/completion_event.h>
#include <dispenso/latch.h>
#include <dispenso/rw_lock.h>

// Memory management
#include <dispenso/pool_allocator.h>
#include <dispenso/resource_pool.h>
#include <dispenso/small_buffer_allocator.h>

// Async utilities
#include <dispenso/async_request.h>
#include <dispenso/once_function.h>
#include <dispenso/timed_task.h>

// Utilities
#include <dispenso/platform.h>
#include <dispenso/priority.h>
#include <dispenso/thread_id.h>
#include <dispenso/timing.h>
#include <dispenso/util.h>

// Sanitizer support
#include <dispenso/tsan_annotations.h>
