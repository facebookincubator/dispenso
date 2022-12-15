/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/timed_task.h>

#include <deque>

#include <dispenso/completion_event.h>
#include <dispenso/schedulable.h>
#include "thread_benchmark_common.h"

#if !defined(BENCHMARK_WITHOUT_FOLLY)
#include <folly/experimental/FunctionScheduler.h>
#include <folly/synchronization/Baton.h>
#endif // !BENCHMARK_WITHOUT_FOLLY

size_t getIterations() {
  // If we want to get a sense of system overhead via getrusage, we need to boost things into the
  // milliseconds range to get any kind of reasonable values.  Typically on Linux I'm seeing that
  // Folly uses slightly more resources (system + user) on average, but that dispenso uses more in
  // the worst cases.  The very worst case is still less than 1% of 1 cpu (~8 seconds active,
  // 50ms of system+user time).
  static const bool itersEnv = getenv("TEST_WITH_MANY_ITERS") != nullptr;
  if (itersEnv) {
    return 2000;
  } else {
    return 200;
  }
}

void absTimesToErrors(std::vector<double>& times, double prevTime, double expectedDelta) {
  for (size_t i = 0; i < times.size(); ++i) {
    auto temp = times[i];
    times[i] -= prevTime;
    times[i] -= expectedDelta;
    times[i] = std::abs(times[i]);
    prevTime = temp;
  }
}

void absTimesToSteadyErrors(std::vector<double>& times, double prevTime, double expectedDelta) {
  for (size_t i = 0; i < times.size(); ++i) {
    times[i] -= prevTime;
    times[i] -= expectedDelta;
    times[i] = std::abs(times[i]);
    prevTime += expectedDelta;
  }
}

#if !defined(BENCHMARK_WITHOUT_FOLLY)
template <size_t kInMs, bool kSteady>
void BM_folly(benchmark::State& state) {
  std::vector<double> times(getIterations());
  std::atomic<size_t> count(0);
  startRusage();
  folly::FunctionScheduler fs;
  if (kSteady) {
    fs.setSteady(true);
  }
  folly::Baton fin;
  fs.addFunction(
      [&] {
        auto cur = count.fetch_add(1, std::memory_order_acq_rel);
        if (cur < times.size()) {
          times[cur] = dispenso::getTime();
        } else if (cur == times.size()) {
          fin.post();
        }
      },
      std::chrono::milliseconds(kInMs),
      "add");
  double prevTime = dispenso::getTime();
  fs.start();
  fin.wait();
  fs.shutdown();

  for (auto UNUSED_VAR : state) {
    // dummy iteration
    state.SetIterationTime(0.1);
  }

  if (kSteady) {
    // We need to adjust prevTime because for steady scheduling, folly tries to schedule right away.
    absTimesToSteadyErrors(times, prevTime - kInMs * 1e-3, kInMs * 1e-3);
  } else {
    absTimesToErrors(times, prevTime, kInMs * 1e-3);
  }
  endRusage(state);

  doStats(times, state);
}

struct FollyItem {
  std::vector<double> times;
  std::atomic<size_t> count{0};
  size_t millis;
  folly::Baton<true, std::atomic> fin;

  FollyItem() : times(getIterations()) {}
};

template <bool kSteady>
void BM_folly_mixed(benchmark::State& state) {
  FollyItem items[3];

  items[0].millis = 1;
  items[1].millis = 4;
  items[2].millis = 3;

  startRusage();
  folly::FunctionScheduler fs;
  if (kSteady) {
    fs.setSteady(true);
  }

  auto doSchedule = [&fs](FollyItem& fitem, const char* name) {
    fs.addFunction(
        [&] {
          auto cur = fitem.count.fetch_add(1, std::memory_order_acq_rel);
          if (cur < fitem.times.size()) {
            fitem.times[cur] = dispenso::getTime();
          } else if (cur == fitem.times.size()) {
            fitem.fin.post();
          }
        },
        std::chrono::milliseconds(fitem.millis),
        name);
  };

  size_t i = 0;
  for (auto name : {"a", "b", "c"}) {
    doSchedule(items[i++], name);
  }

  double prevTime = dispenso::getTime();
  fs.start();
  for (auto& item : items) {
    item.fin.wait();
  }
  fs.shutdown();

  for (auto UNUSED_VAR : state) {
    // dummy iteration
    state.SetIterationTime(0.1);
  }

  if (kSteady) {
    // We need to adjust prevTime because for steady scheduling, folly tries to schedule right away.
    for (auto& item : items) {
      absTimesToSteadyErrors(item.times, prevTime - item.millis * 1e-3, item.millis * 1e-3);
    }
  } else {
    for (auto& item : items) {
      absTimesToErrors(item.times, prevTime, item.millis * 1e-3);
    }
  }
  endRusage(state);

  // append all times
  items[0].times.insert(items[0].times.end(), items[1].times.begin(), items[1].times.end());
  items[0].times.insert(items[0].times.end(), items[2].times.begin(), items[2].times.end());

  doStats(items[0].times, state);
}

#endif

dispenso::TimedTaskScheduler& getScheduler() {
  static const bool rtEnv = getenv("TEST_WITH_REALTIME") != nullptr;
  if (rtEnv) {
    static dispenso::TimedTaskScheduler sched(dispenso::ThreadPriority::kRealtime);
    return sched;
  } else {
    return dispenso::globalTimedTaskScheduler();
  }
}

template <size_t kInMs, bool kSteady>
void BM_dispenso(benchmark::State& state) {
  std::vector<double> times(getIterations());
  std::atomic<size_t> count(0);

  double period = 1e-3 * kInMs;

  startRusage();
  dispenso::CompletionEvent fin;
  double prevTime = dispenso::getTime();

  auto type = kSteady ? dispenso::TimedTaskType::kSteady : dispenso::TimedTaskType::kNormal;

  auto task = getScheduler().schedule(
      dispenso::kImmediateInvoker,
      [&] {
        auto cur = count.fetch_add(1, std::memory_order_acq_rel);
        if (cur < times.size()) {
          times[cur] = dispenso::getTime();
        }
        if (cur + 1 == times.size()) {
          fin.notify();
          return false;
        }
        return true;
      },
      prevTime + period,
      period,
      times.size(),
      type);
  fin.wait();

  for (auto UNUSED_VAR : state) {
    // dummy iteration
    state.SetIterationTime(0.1);
  }

  if (kSteady) {
    absTimesToSteadyErrors(times, prevTime, kInMs * 1e-3);

  } else {
    absTimesToErrors(times, prevTime, kInMs * 1e-3);
  }
  endRusage(state);

  doStats(times, state);
}

struct DispensoItem {
  std::vector<double> times;
  std::atomic<size_t> count{0};
  size_t millis;
  dispenso::CompletionEvent fin;
  DispensoItem() : times(getIterations()) {}
};

template <bool kSteady>
void BM_dispenso_mixed(benchmark::State& state) {
  DispensoItem items[3];
  std::deque<dispenso::TimedTask> tasks;

  items[0].millis = 1;
  items[1].millis = 4;
  items[2].millis = 3;

  startRusage();
  double prevTime = dispenso::getTime();
  auto type = kSteady ? dispenso::TimedTaskType::kSteady : dispenso::TimedTaskType::kNormal;
  auto doSchedule = [&](DispensoItem& ditem) {
    double period = 1e-3 * ditem.millis;

    tasks.push_back(getScheduler().schedule(
        dispenso::kImmediateInvoker,
        [&] {
          auto cur = ditem.count.fetch_add(1, std::memory_order_acq_rel);
          if (cur < ditem.times.size()) {
            ditem.times[cur] = dispenso::getTime();
          }
          if (cur + 1 == ditem.times.size()) {
            ditem.fin.notify();
            return false;
          }
          return true;
        },
        prevTime + period,
        period,
        ditem.times.size(),
        type));
  };

  for (size_t i = 0; i < 3; ++i) {
    doSchedule(items[i]);
  }

  for (auto& item : items) {
    item.fin.wait();
  }

  for (auto UNUSED_VAR : state) {
    // dummy iteration
    state.SetIterationTime(0.1);
  }

  if (kSteady) {
    for (auto& item : items) {
      absTimesToSteadyErrors(item.times, prevTime, item.millis * 1e-3);
    }
  } else {
    for (auto& item : items) {
      absTimesToErrors(item.times, prevTime, item.millis * 1e-3);
    }
  }
  endRusage(state);

  // append all times
  items[0].times.insert(items[0].times.end(), items[1].times.begin(), items[1].times.end());
  items[0].times.insert(items[0].times.end(), items[2].times.begin(), items[2].times.end());

  doStats(items[0].times, state);
}

BENCHMARK_TEMPLATE2(BM_dispenso, 2, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_dispenso, 4, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_dispenso, 6, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_dispenso, 2, true)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_dispenso, 4, true)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_dispenso, 6, true)->UseManualTime();

BENCHMARK_TEMPLATE(BM_dispenso_mixed, false)->UseManualTime();
BENCHMARK_TEMPLATE(BM_dispenso_mixed, true)->UseManualTime();

#if !defined(BENCHMARK_WITHOUT_FOLLY)
BENCHMARK_TEMPLATE2(BM_folly, 2, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_folly, 4, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_folly, 6, false)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_folly, 2, true)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_folly, 4, true)->UseManualTime();
BENCHMARK_TEMPLATE2(BM_folly, 6, true)->UseManualTime();

BENCHMARK_TEMPLATE(BM_folly_mixed, false)->UseManualTime();
BENCHMARK_TEMPLATE(BM_folly_mixed, true)->UseManualTime();
#endif // FOLLY
