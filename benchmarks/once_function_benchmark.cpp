// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <cstring>
#include <deque>
#include <functional>

#include <benchmark/benchmark.h>
#include <dispenso/once_function.h>

constexpr size_t kSmallSize = 24;
constexpr size_t kMediumSize = 120;
constexpr size_t kLargeSize = 248;
// 1000 is larger than our largest optimized chunk size, so we may expect to see performance falloff
// here.
constexpr size_t kExtraLargeSize = 1000;

template <typename ExeType, typename Func>
void runMoveLoop(benchmark::State& state, Func f) {
  for (auto _ : state) {
    ExeType t(f);
    ExeType o;
    for (int i = 0; i < 10; ++i) {
      o = std::move(t);
      t = std::move(o);
    }
    t();
  }
}

template <typename Func>
class FuncConsumer {
 public:
  void add(Func&& f) {
    funcs_.emplace_back(std::move(f));
  }

  void consumeAll() {
    while (!funcs_.empty()) {
      Func f = std::move(funcs_.front());
      funcs_.pop_front();
      f();
    }
  }

 private:
  std::deque<Func> funcs_;
};

template <size_t kSize>
struct Foo {
  Foo() {
    buf[0] = 0;
    benchmark::ClobberMemory();
  }

  Foo(Foo<kSize>&& f) {
    std::memcpy(buf, f.buf, kSize);
  }

  Foo(const Foo<kSize>& f) {
    std::memcpy(buf, f.buf, kSize);
  }

  void operator()() {
    benchmark::DoNotOptimize(++buf[0]);
  }

  uint32_t buf[kSize / 4];
};

template <typename F>
void onceCall(F&& f) {
  F lf = std::move(f);
  lf();
}

template <size_t kSize>
void BM_move_std_function(benchmark::State& state) {
  runMoveLoop<std::function<void()>>(state, Foo<kSize>());
}

template <size_t kSize>
void BM_move_once_function(benchmark::State& state) {
  runMoveLoop<dispenso::OnceFunction>(state, Foo<kSize>());
}

constexpr int kMediumLoopLen = 200;

template <size_t kSize>
void BM_queue_inline_function(benchmark::State& state) {
  FuncConsumer<Foo<kSize>> consumer;
  for (auto _ : state) {
    for (int i = 0; i < kMediumLoopLen; ++i) {
      consumer.add(Foo<kSize>());
    }
    consumer.consumeAll();
  }
}

template <size_t kSize>
void BM_queue_std_function(benchmark::State& state) {
  FuncConsumer<std::function<void()>> consumer;
  for (auto _ : state) {
    for (int i = 0; i < kMediumLoopLen; ++i) {
      consumer.add(Foo<kSize>());
    }
    consumer.consumeAll();
  }
}

template <size_t kSize>
void BM_queue_once_function(benchmark::State& state) {
  FuncConsumer<dispenso::OnceFunction> consumer;
  for (auto _ : state) {
    for (int i = 0; i < kMediumLoopLen; ++i) {
      consumer.add(Foo<kSize>());
    }
    consumer.consumeAll();
  }
}

BENCHMARK_TEMPLATE(BM_move_std_function, kSmallSize);
BENCHMARK_TEMPLATE(BM_move_once_function, kSmallSize);

BENCHMARK_TEMPLATE(BM_move_std_function, kMediumSize);
BENCHMARK_TEMPLATE(BM_move_once_function, kMediumSize);

BENCHMARK_TEMPLATE(BM_move_std_function, kLargeSize);
BENCHMARK_TEMPLATE(BM_move_once_function, kLargeSize);

BENCHMARK_TEMPLATE(BM_move_std_function, kExtraLargeSize);
BENCHMARK_TEMPLATE(BM_move_once_function, kExtraLargeSize);

BENCHMARK_TEMPLATE(BM_queue_inline_function, kSmallSize);
BENCHMARK_TEMPLATE(BM_queue_std_function, kSmallSize);
BENCHMARK_TEMPLATE(BM_queue_once_function, kSmallSize);

BENCHMARK_TEMPLATE(BM_queue_inline_function, kMediumSize);
BENCHMARK_TEMPLATE(BM_queue_std_function, kMediumSize);
BENCHMARK_TEMPLATE(BM_queue_once_function, kMediumSize);

BENCHMARK_TEMPLATE(BM_queue_inline_function, kLargeSize);
BENCHMARK_TEMPLATE(BM_queue_std_function, kLargeSize);
BENCHMARK_TEMPLATE(BM_queue_once_function, kLargeSize);

BENCHMARK_TEMPLATE(BM_queue_inline_function, kExtraLargeSize);
BENCHMARK_TEMPLATE(BM_queue_std_function, kExtraLargeSize);
BENCHMARK_TEMPLATE(BM_queue_once_function, kExtraLargeSize);

BENCHMARK_MAIN();
