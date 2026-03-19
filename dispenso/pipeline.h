/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file pipeline.h
 * @ingroup group_graph
 * A file providing utilities for parallel pipelining of work.
 **/

#pragma once

#include <limits>

#include <dispenso/detail/pipeline_impl.h>

namespace dispenso {

/**
 * OpResult is like a poor-man's std::optional for those who wish to use dispenso pipeline filtering
 * in C++14.  In C++17 and beyond, it is recommended to use std::optional instead.  OpResult has
 * implicit construct from T, just like std::optional, and move/copy constructors and operators,
 * bool conversion, and value() function, but otherwise provides less functionality than
 * std::optional.
 **/
template <typename T>
using OpResult = detail::OpResult<T>;

/**
 * A simple constant representing maximum parallelism for a stage.  This number has no particular
 * significance, and is simply here for convenience.
 **/
constexpr ssize_t kStageNoLimit = std::numeric_limits<ssize_t>::max();

/**
 * Create a stage for use in the pipeline function.
 *
 * @param f A function-like object that can accept the result of the previous stage (if any), and
 * which produces the output for the next stage (if any).
 * @param limit How many threads may concurrently run work for this stage.  Values larger than the
 * number of threads in the associated thread pool of the used ConcurrentTaskSet will be capped to
 * the size of the pool.
 * @return A stage object suitable for pipelining.
 **/
template <typename F>
auto stage(F&& f, ssize_t limit) {
  return detail::Stage<F>(std::forward<F>(f), limit);
}

/**
 * Pipeline work in stages.  Pipelines allow stages to specify parallelism limits by using the
 * <code>stage</code> function, or a function-like object can simply be passed directly, indicating
 * a serial stage.  Even if stages are serial, there can be parallelism between stages, so in a 3
 * stage serial pipeline, the expected runtime is the max of the 3 stages runtimes (note that this
 * is in the absence of pipeline overheads and with an infinitely long workstream.  In practice
 * speedup is somewhat less). This function will block until the entire pipeline has completed.
 *
 * @param pool The ThreadPool to run the work in.  This inherently determines the upper bound for
 * parallelism of the pipeline.
 * @param sIn The stages to run.  The first stage must be a Generator stage, the last must be a Sink
 * stage, and intermediate stages are Transform stages.
 * - If there is only one stage, it takes no
 * arguments, but returns a bool indicating completion (false means the pipeline is complete).
 * - Otherwise, the Generator stage takes no arguments and  must return an OpResult or std::optional
 * value, and an invalid/nullopt result indicates that the Generator is done (no more values
 * forthcoming).
 * - Transform stages should accept the output of the prior stage (or output.value() in the case of
 * OpResult or std::optional), and should return either a value or an OpResult or std::optional
 * value if the Transform is capable of filtering results. Invalid/nullopt OpResult or std::optional
 * values indicate that the value should be filtered, and not passed on to the next stage.
 * - The Sink stage should accept the output of the prior stage, just as a Transform stage does, but
 * does not return any value (or at least the pipeline will ignore it).
 *
 * @note <b>Exception behavior:</b> If a stage function throws an exception, the pipeline will stop
 * producing new work (the generator checks for exceptions between iterations) and allow any
 * already-in-flight work to complete. Queued work that has not yet started will be discarded
 * without execution. The first captured exception is rethrown from <code>pipeline()</code> when the
 * pipeline winds down. Subsequent exceptions from concurrently in-flight stages are silently
 * discarded. All internal state (concurrency counters, resource slots, completion events) is
 * cleaned up via RAII, so the thread pool remains in a valid state after an exception and may be
 * reused for subsequent pipelines. When exceptions are disabled at compile time
 * (<code>__cpp_exceptions</code> not defined), all exception-related checks are eliminated with
 * zero overhead.
 **/
template <typename... Stages>
void pipeline(ThreadPool& pool, Stages&&... sIn) {
  ConcurrentTaskSet tasks(pool);
  auto pipes = detail::makePipes(tasks, std::forward<Stages>(sIn)...);
  pipes.execute();
  pipes.wait();
}

/**
 * Pipeline work in stages.  Pipelines allow stages to specify parallelism limits by using the
 * <code>stage</code> function, or a function-like object can simply be passed directly, indicating
 * a serial stage.  Even if stages are serial, there can be parallelism between stages, so in a 3
 * stage serial pipeline, the expected runtime is the max of the 3 stages runtimes (note that this
 * is in the absence of pipeline overheads and with an infinitely long workstream.  In practice
 * speedup is somewhat less). Work will be run on dispenso's global thread pool.  This function will
 * block until the entire pipeline has completed.
 *
 * @param sIn The stages to run.  The first stage must be a Generator stage, the last must be a Sink
 * stage, and intermediate stages are Transform stages.
 * - If there is only one stage, it takes no
 * arguments, but returns a bool indicating completion (false means the pipeline is complete).
 * - Otherwise, the Generator stage takes no arguments and  must return an OpResult or std::optional
 * value, and an invalid/nullopt result indicates that the Generator is done (no more values
 * forthcoming).
 * - Transform stages should accept the output of the prior stage (or output.value() in the case of
 * OpResult or std::optional), and should return either a value or an OpResult or std::optional
 * value if the Transform is capable of filtering results. Invalid/nullopt OpResult or std::optional
 * values indicate that the value should be filtered, and not passed on to the next stage.
 * - The Sink stage should accept the output of the prior stage, just as a Transform stage does, but
 * does not return any value (or at least the pipeline will ignore it).
 *
 * @note See the ThreadPool overload of <code>pipeline()</code> for exception behavior details.
 **/
template <typename... Stages>
void pipeline(Stages&&... sIn) {
  pipeline(globalThreadPool(), std::forward<Stages>(sIn)...);
}

} // namespace dispenso
