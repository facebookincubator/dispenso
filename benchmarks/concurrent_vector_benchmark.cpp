/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <deque>
#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#if !defined(BENCHMARK_WITHOUT_TBB)
#include "tbb/concurrent_vector.h"
#endif // !BENCHMARK_WITHOUT_TBB

#include <dispenso/concurrent_vector.h>
#include <dispenso/parallel_for.h>

#include "thread_benchmark_common.h"

constexpr size_t kLength = (1 << 20);

void checkIotaSum(int64_t sum) {
  if (sum != (static_cast<int64_t>(kLength - 1) * kLength) / 2) {
    std::cout << sum << " vs " << ((kLength - 1) * kLength) / 2 << std::endl;

    std::abort();
  }
}

template <typename Cont>
void checkIotaSum(const Cont& c, int64_t sum) {
  if (sum != (static_cast<int64_t>(kLength - 1) * kLength) / 2) {
    std::cout << sum << " vs " << ((kLength - 1) * kLength) / 2 << std::endl;

    std::vector<uint8_t> accountedFor(kLength);
    for (auto v : c) {
      accountedFor[v] = 1;
    }

    for (size_t i = 0; i < kLength; ++i) {
      if (!accountedFor[i]) {
        std::cout << "missing " << i << std::endl;
      }
    }

    std::abort();
  }
}

template <typename ContainerInit>
void pushBackImpl(benchmark::State& state, ContainerInit containerInit) {
  for (auto UNUSED_VAR : state) {
    auto values = containerInit();
    for (size_t i = 0; i < kLength; ++i) {
      values.push_back(i);
    }
  }
}

#if !defined(BENCHMARK_WITHOUT_TBB)
template <typename ContainerInit>
void pushBackGrowByAlternativeTbb(benchmark::State& state, ContainerInit containerInit) {
  for (auto UNUSED_VAR : state) {
    auto values = containerInit();
    auto it = values.grow_by(kLength);
    auto end = values.end();
    size_t i = 0;
    for (; it != end; ++it) {
      *it = i++;
    }
  }
}
#endif // !BENCHMARK_WITHOUT_TBB

template <typename ContainerInit>
void pushBackGrowByAlternativeDispenso(benchmark::State& state, ContainerInit containerInit) {
  for (auto UNUSED_VAR : state) {
    auto values = containerInit();
    values.grow_by_generator(kLength, [i = size_t{0}]() mutable { return i++; });
  }
}

void BM_std_push_back_serial(benchmark::State& state) {
  pushBackImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_push_back_serial(benchmark::State& state) {
  pushBackImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_push_back_serial(benchmark::State& state) {
  pushBackImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_push_back_serial(benchmark::State& state) {
  pushBackImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_push_back_serial_grow_by_alternative(benchmark::State& state) {
  pushBackGrowByAlternativeTbb(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_push_back_serial_grow_by_alternative(benchmark::State& state) {
  pushBackGrowByAlternativeDispenso(state, []() { return dispenso::ConcurrentVector<int>(); });
}

void BM_std_push_back_serial_reserve(benchmark::State& state) {
  pushBackImpl(state, []() {
    std::vector<int> v;
    v.reserve(kLength);
    return v;
  });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_push_back_serial_reserve(benchmark::State& state) {
  pushBackImpl(state, []() {
    tbb::concurrent_vector<int> v;
    v.reserve(kLength);
    return v;
  });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_push_back_serial_reserve(benchmark::State& state) {
  pushBackImpl(
      state, []() { return dispenso::ConcurrentVector<int>(kLength, dispenso::ReserveTag); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_push_back_serial_grow_by_alternative_reserve(benchmark::State& state) {
  pushBackGrowByAlternativeTbb(state, []() {
    tbb::concurrent_vector<int> v;
    v.reserve(kLength);
    return v;
  });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_push_back_serial_grow_by_alternative_reserve(benchmark::State& state) {
  pushBackGrowByAlternativeDispenso(
      state, []() { return dispenso::ConcurrentVector<int>(kLength, dispenso::ReserveTag); });
}

template <typename ContainerInit>
void iterateImpl(benchmark::State& state, ContainerInit containerInit) {
  auto values = containerInit();
  for (size_t i = 0; i < kLength; ++i) {
    values.push_back(i);
  }
  int64_t sum;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    for (auto i : values) {
      sum += i;
    }
  }

  checkIotaSum(sum);
}

void BM_std_iterate(benchmark::State& state) {
  iterateImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_iterate(benchmark::State& state) {
  iterateImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_iterate(benchmark::State& state) {
  iterateImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_iterate(benchmark::State& state) {
  iterateImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

template <typename T>
struct ReverseWrapper {
  T& iterable;
};

template <typename T>
auto begin(ReverseWrapper<T> w) {
  return std::rbegin(w.iterable);
}

template <typename T>
auto end(ReverseWrapper<T> w) {
  return std::rend(w.iterable);
}

template <typename T>
ReverseWrapper<T> reverse(T&& iterable) {
  return {iterable};
}

template <typename ContainerInit>
void iterateReverseImpl(benchmark::State& state, ContainerInit containerInit) {
  auto values = containerInit();
  for (size_t i = 0; i < kLength; ++i) {
    values.push_back(i);
  }
  int64_t sum;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    for (auto i : reverse(values)) {
      sum += i;
    }
  }

  checkIotaSum(sum);
}

void BM_std_iterate_reverse(benchmark::State& state) {
  iterateReverseImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_iterate_reverse(benchmark::State& state) {
  iterateReverseImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_iterate_reverse(benchmark::State& state) {
  iterateReverseImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_iterate_reverse(benchmark::State& state) {
  iterateReverseImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

template <typename ContainerInit>
void lowerBoundImpl(benchmark::State& state, ContainerInit containerInit) {
  auto values = containerInit();
  for (size_t i = 0; i < kLength; ++i) {
    values.push_back(i);
  }
  int64_t sum;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    for (size_t i = 0; i < kLength; ++i) {
      sum += std::lower_bound(std::begin(values), std::end(values), i) - std::begin(values);
    }
  }

  checkIotaSum(sum);
}

void BM_std_lower_bound(benchmark::State& state) {
  lowerBoundImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_lower_bound(benchmark::State& state) {
  lowerBoundImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_lower_bound(benchmark::State& state) {
  lowerBoundImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_lower_bound(benchmark::State& state) {
  lowerBoundImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

template <typename ContainerInit>
void indexImpl(benchmark::State& state, ContainerInit containerInit) {
  auto values = containerInit();
  for (size_t i = 0; i < kLength; ++i) {
    values.push_back(i);
  }
  int64_t sum;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    size_t len = values.size();
    for (size_t i = 0; i < len; ++i) {
      sum += values[i];
    }
  }

  checkIotaSum(sum);
}

void BM_std_index(benchmark::State& state) {
  indexImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_index(benchmark::State& state) {
  indexImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_index(benchmark::State& state) {
  indexImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_index(benchmark::State& state) {
  indexImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

template <typename ContainerInit>
void randomImpl(benchmark::State& state, ContainerInit containerInit) {
  auto values = containerInit();
  std::vector<size_t> indices;
  for (size_t i = 0; i < kLength; ++i) {
    values.push_back(i);
    indices.push_back(i);
  }

  // Make this repeatable.
  std::mt19937 rng(27);
  std::shuffle(indices.begin(), indices.end(), rng);

  int64_t sum;
  for (auto UNUSED_VAR : state) {
    sum = 0;
    for (auto i : indices) {
      sum += values[i];
    }
  }

  checkIotaSum(sum);
}

void BM_std_random(benchmark::State& state) {
  randomImpl(state, []() { return std::vector<int>(); });
}

void BM_deque_random(benchmark::State& state) {
  randomImpl(state, []() { return std::deque<int>(); });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_random(benchmark::State& state) {
  randomImpl(state, []() { return tbb::concurrent_vector<int>(); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_random(benchmark::State& state) {
  randomImpl(state, []() { return dispenso::ConcurrentVector<int>(); });
}

template <typename ContainerInit, typename ContainerPush>
void parallelImpl(
    benchmark::State& state,
    ContainerInit containerInit,
    ContainerPush containerPush) {
  for (auto UNUSED_VAR : state) {
    auto values = containerInit();
    dispenso::parallel_for(
        0, kLength, [&values, containerPush](size_t i) { containerPush(values, i); });
  }
}

void BM_std_parallel(benchmark::State& state) {
  std::mutex mtx;
  parallelImpl(
      state,
      []() { return std::vector<int>(); },
      [&mtx](std::vector<int>& c, int i) {
        std::lock_guard<std::mutex> lk(mtx);
        c.push_back(i);
      });
}

void BM_deque_parallel(benchmark::State& state) {
  std::mutex mtx;
  parallelImpl(
      state,
      []() { return std::deque<int>(); },
      [&mtx](std::deque<int>& c, int i) {
        std::lock_guard<std::mutex> lk(mtx);
        c.push_back(i);
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel(benchmark::State& state) {
  parallelImpl(
      state,
      []() { return tbb::concurrent_vector<int>(); },
      [](tbb::concurrent_vector<int>& c, int i) { c.push_back(i); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel(benchmark::State& state) {
  parallelImpl(
      state,
      []() { return dispenso::ConcurrentVector<int>(); },
      [](dispenso::ConcurrentVector<int>& c, int i) { c.push_back(i); });
}

void BM_std_parallel_reserve(benchmark::State& state) {
  std::mutex mtx;
  parallelImpl(
      state,
      []() {
        std::vector<int> v;
        v.reserve(kLength);
        return v;
      },
      [&mtx](std::vector<int>& c, int i) {
        std::lock_guard<std::mutex> lk(mtx);
        c.push_back(i);
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel_reserve(benchmark::State& state) {
  parallelImpl(
      state,
      []() {
        tbb::concurrent_vector<int> v;
        v.reserve(kLength);
        return v;
      },
      [](tbb::concurrent_vector<int>& c, int i) { c.push_back(i); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel_reserve(benchmark::State& state) {
  parallelImpl(
      state,
      []() { return dispenso::ConcurrentVector<int>(kLength, dispenso::ReserveTag); },
      [](dispenso::ConcurrentVector<int>& c, int i) { c.push_back(i); });
}

template <typename ContainerInit, typename ContainerPush>
void parallelImplClear(
    benchmark::State& state,
    ContainerInit containerInit,
    ContainerPush containerPush) {
  auto values = containerInit();

  for (auto UNUSED_VAR : state) {
    values.clear();
    dispenso::parallel_for(
        0, kLength, [&values, containerPush](size_t i) { containerPush(values, i); });
  }

  int64_t sum = 0;

  for (auto i : values) {
    sum += i;
  }

  checkIotaSum(sum);
}

void BM_std_parallel_clear(benchmark::State& state) {
  std::mutex mtx;
  parallelImplClear(
      state,
      []() { return std::vector<int>(); },
      [&mtx](std::vector<int>& c, int i) {
        std::lock_guard<std::mutex> lk(mtx);
        c.push_back(i);
      });
}

void BM_deque_parallel_clear(benchmark::State& state) {
  std::mutex mtx;
  parallelImplClear(
      state,
      []() { return std::deque<int>(); },
      [&mtx](std::deque<int>& c, int i) {
        std::lock_guard<std::mutex> lk(mtx);
        c.push_back(i);
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel_clear(benchmark::State& state) {
  parallelImplClear(
      state,
      []() { return tbb::concurrent_vector<int>(); },
      [](tbb::concurrent_vector<int>& c, int i) { c.push_back(i); });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel_clear(benchmark::State& state) {
  parallelImplClear(
      state,
      []() { return dispenso::ConcurrentVector<int>(); },
      [](dispenso::ConcurrentVector<int>& c, int i) { c.push_back(i); });
}

template <typename ContainerInit, typename ContainerPush>
void parallelImplGrowBy(
    size_t growBy,
    benchmark::State& state,
    ContainerInit containerInit,
    ContainerPush containerPush) {
  auto values = containerInit();

  for (auto UNUSED_VAR : state) {
    values.clear();
    dispenso::parallel_for(
        dispenso::makeChunkedRange(0, kLength, dispenso::ParForChunking::kStatic),
        [&values, containerPush, growBy](size_t i, size_t end) {
          while (i + growBy <= end) {
            containerPush(values, i, i + growBy);
            i += growBy;
          }
          containerPush(values, i, end);
        });
  }

  int64_t sum = 0;

  for (auto i : values) {
    sum += i;
  }

  checkIotaSum(values, sum);
}

void BM_std_parallel_grow_by_10(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowBy(
      10,
      state,
      []() { return std::vector<int>(); },
      [&mtx](std::vector<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

void BM_deque_parallel_grow_by_10(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowBy(
      10,
      state,
      []() { return std::deque<int>(); },
      [&mtx](std::deque<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel_grow_by_10(benchmark::State& state) {
  parallelImplGrowBy(
      10,
      state,
      []() { return tbb::concurrent_vector<int>(); },
      [](tbb::concurrent_vector<int>& c, int i, int end) {
        auto it = c.grow_by(end - i);
        for (; i != end; ++i, ++it) {
          *it = i;
        }
      });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel_grow_by_10(benchmark::State& state) {
  parallelImplGrowBy(
      10,
      state,
      []() { return dispenso::ConcurrentVector<int>(); },
      [](dispenso::ConcurrentVector<int>& c, int i, int end) {
        c.grow_by_generator(end - i, [i]() mutable { return i++; });
      });
}

void BM_std_parallel_grow_by_100(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowBy(
      100,
      state,
      []() { return std::vector<int>(); },
      [&mtx](std::vector<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

void BM_deque_parallel_grow_by_100(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowBy(
      100,
      state,
      []() { return std::deque<int>(); },
      [&mtx](std::deque<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel_grow_by_100(benchmark::State& state) {
  parallelImplGrowBy(
      100,
      state,
      []() { return tbb::concurrent_vector<int>(); },
      [](tbb::concurrent_vector<int>& c, int i, int end) {
        auto it = c.grow_by(end - i);
        for (; i != end; ++i, ++it) {
          *it = i;
        }
      });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel_grow_by_100(benchmark::State& state) {
  parallelImplGrowBy(
      100,
      state,
      []() { return dispenso::ConcurrentVector<int>(); },
      [](dispenso::ConcurrentVector<int>& c, int i, int end) {
        c.grow_by_generator(end - i, [i]() mutable { return i++; });
      });
}

template <typename ContainerInit, typename ContainerPush>
void parallelImplGrowByMax(
    benchmark::State& state,
    ContainerInit containerInit,
    ContainerPush containerPush) {
  auto values = containerInit();

  for (auto UNUSED_VAR : state) {
    values.clear();
    dispenso::parallel_for(
        dispenso::makeChunkedRange(0, kLength, dispenso::ParForChunking::kStatic),
        [&values, containerPush](size_t i, size_t end) { containerPush(values, i, end); });
  }

  int64_t sum = 0;

  for (auto i : values) {
    sum += i;
  }

  checkIotaSum(sum);
}

void BM_std_parallel_grow_by_max(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowByMax(
      state,
      []() { return std::vector<int>(); },
      [&mtx](std::vector<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

void BM_deque_parallel_grow_by_max(benchmark::State& state) {
  std::mutex mtx;
  parallelImplGrowByMax(
      state,
      []() { return std::deque<int>(); },
      [&mtx](std::deque<int>& c, int i, int end) {
        std::lock_guard<std::mutex> lk(mtx);
        for (; i != end; ++i) {
          c.push_back(i);
        }
      });
}

#if !defined(BENCHMARK_WITHOUT_TBB)
void BM_tbb_parallel_grow_by_max(benchmark::State& state) {
  parallelImplGrowByMax(
      state,
      []() { return tbb::concurrent_vector<int>(); },
      [](tbb::concurrent_vector<int>& c, int i, int end) {
        auto it = c.grow_by(end - i);
        for (; i != end; ++i, ++it) {
          *it = i;
        }
      });
}
#endif // !BENCHMARK_WITHOUT_TBB

void BM_dispenso_parallel_grow_by_max(benchmark::State& state) {
  parallelImplGrowByMax(
      state,
      []() { return dispenso::ConcurrentVector<int>(); },
      [](dispenso::ConcurrentVector<int>& c, int i, int end) {
        c.grow_by_generator(end - i, [i]() mutable { return i++; });
      });
}

BENCHMARK(BM_std_push_back_serial);
BENCHMARK(BM_deque_push_back_serial);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_push_back_serial);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_push_back_serial);

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_push_back_serial_grow_by_alternative);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_push_back_serial_grow_by_alternative);

BENCHMARK(BM_std_push_back_serial_reserve);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_push_back_serial_reserve);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_push_back_serial_reserve);

#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_push_back_serial_grow_by_alternative_reserve);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_push_back_serial_grow_by_alternative_reserve);

BENCHMARK(BM_std_iterate);
BENCHMARK(BM_deque_iterate);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_iterate);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_iterate);

BENCHMARK(BM_std_iterate_reverse);
BENCHMARK(BM_deque_iterate_reverse);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_iterate_reverse);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_iterate_reverse);

BENCHMARK(BM_std_lower_bound);
BENCHMARK(BM_deque_lower_bound);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_lower_bound);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_lower_bound);

BENCHMARK(BM_std_index);
BENCHMARK(BM_deque_index);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_index);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_index);

BENCHMARK(BM_std_random);
BENCHMARK(BM_deque_random);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_random);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_random);

BENCHMARK(BM_std_parallel);
BENCHMARK(BM_deque_parallel);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel);

BENCHMARK(BM_std_parallel_reserve);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel_reserve);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel_reserve);

BENCHMARK(BM_std_parallel_clear);
BENCHMARK(BM_deque_parallel_clear);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel_clear);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel_clear);

BENCHMARK(BM_std_parallel_grow_by_10);
BENCHMARK(BM_deque_parallel_grow_by_10);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel_grow_by_10);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel_grow_by_10);

BENCHMARK(BM_std_parallel_grow_by_100);
BENCHMARK(BM_deque_parallel_grow_by_100);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel_grow_by_100);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel_grow_by_100);

BENCHMARK(BM_std_parallel_grow_by_max);
BENCHMARK(BM_deque_parallel_grow_by_max);
#if !defined(BENCHMARK_WITHOUT_TBB)
BENCHMARK(BM_tbb_parallel_grow_by_max);
#endif // !BENCHMARK_WITHOUT_TBB
BENCHMARK(BM_dispenso_parallel_grow_by_max);

BENCHMARK_MAIN();
