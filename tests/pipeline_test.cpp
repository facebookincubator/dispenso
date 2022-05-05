/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/pipeline.h>

#include <numeric>

#include <gtest/gtest.h>

#if __cplusplus >= 201703L
template <typename T>
using TestOptional = std::optional<T>;
#else
template <typename T>
using TestOptional = dispenso::OpResult<T>;
#endif // C++17

TEST(Pipeline, SingleStageSerial) {
  int counter = 0;

  dispenso::pipeline([&counter]() {
    if (counter++ < 10) {
      return true;
    }
    return false;
  });

  EXPECT_EQ(counter, 11);
}

TEST(Pipeline, SingleStageSerialPassRefObject) {
  int counter = 0;

  auto single = [&counter]() {
    if (counter++ < 10) {
      return true;
    }
    return false;
  };

  dispenso::pipeline(single);

  EXPECT_EQ(counter, 11);
}

TEST(Pipeline, MultiStageSerial) {
  std::vector<int> inputs(10);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<int> sum(0);
  dispenso::pipeline(
      [&inputs, it = inputs.begin()]() mutable -> TestOptional<int> {
        if (it != inputs.end()) {
          return *it++;
        }
        return {};
      },
      [](auto num) -> TestOptional<int> {
        if (num > 2 && num < 5) {
          return {};
        }
        return num * num;
      },
      [](int num) { return num + 5; },
      [&sum](auto num) { sum.fetch_add(num, std::memory_order_relaxed); });

  int actualSum = 0;
  for (int i = 0; i < 10; ++i) {
    if (i <= 2 || i >= 5) {
      actualSum += i * i + 5;
    }
  }

  EXPECT_EQ(sum.load(std::memory_order_acquire), actualSum);
}

#if __cplusplus >= 201703L
// This test is nearly identical to the one above, but ensures that we always are testing
// dispenso::OpResult, because the rest of the tests use std::optional if we are in C++17 mode.
TEST(Pipeline, MultiStageSerialOpResult) {
  std::vector<int> inputs(10);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<int> sum(0);
  dispenso::pipeline(
      [&inputs, it = inputs.begin()]() mutable -> dispenso::OpResult<int> {
        if (it != inputs.end()) {
          return *it++;
        }
        return {};
      },
      [](auto num) -> dispenso::OpResult<int> {
        if (num > 2 && num < 5) {
          return {};
        }
        return num * num;
      },
      [](int num) { return num + 5; },
      [&sum](auto num) { sum.fetch_add(num, std::memory_order_relaxed); });

  int actualSum = 0;
  for (int i = 0; i < 10; ++i) {
    if (i <= 2 || i >= 5) {
      actualSum += i * i + 5;
    }
  }

  EXPECT_EQ(sum.load(std::memory_order_acquire), actualSum);
}
#endif // C++17

TEST(Pipeline, SingleStageParallel) {
  std::atomic<int> counter(0);

  constexpr int kMaxConcurrency = 4;

  dispenso::pipeline(dispenso::stage(
      [&counter]() {
        if (counter.fetch_add(1, std::memory_order_acq_rel) < 10) {
          return true;
        }
        return false;
      },
      kMaxConcurrency));

  EXPECT_LE(counter.load(std::memory_order_acquire), 10 + kMaxConcurrency);
}

TEST(Pipeline, SingleStageParallelPassPrebuiltStage) {
  std::atomic<int> counter(0);

  constexpr int kMaxConcurrency = 4;

  auto stage = dispenso::stage(
      [&counter]() {
        if (counter.fetch_add(1, std::memory_order_acq_rel) < 10) {
          return true;
        }
        return false;
      },
      kMaxConcurrency);

  dispenso::pipeline(stage);

  EXPECT_LE(counter.load(std::memory_order_acquire), 10 + kMaxConcurrency);
}

TEST(Pipeline, MultiStageGenIsParallel) {
  constexpr int kNumInputs = 1000;
  std::vector<int> inputs(kNumInputs);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<int> sum(0);
  std::atomic<size_t> counter(0);

  dispenso::pipeline(
      dispenso::stage(
          [&counter, &inputs]() -> TestOptional<int> {
            size_t cur = counter.fetch_add(1, std::memory_order_acq_rel);
            if (cur < inputs.size()) {
              return inputs[cur];
            }
            return {};
          },
          3),
      [](auto num) -> TestOptional<int> {
        if (num > 2 && num < 5) {
          return {};
        }
        return num * num;
      },
      [](int num) { return num + 5; },
      [&sum](auto num) { sum.fetch_add(num, std::memory_order_relaxed); });

  int actualSum = 0;
  for (int i = 0; i < kNumInputs; ++i) {
    if (i <= 2 || i >= 5) {
      actualSum += i * i + 5;
    }
  }

  EXPECT_EQ(sum.load(std::memory_order_acquire), actualSum);
}

struct Gen {
  Gen(std::atomic<size_t>& c, std::vector<int>& i) : counter(c), inputs(i) {}
  TestOptional<int*> operator()() {
    size_t cur = counter.fetch_add(1, std::memory_order_acq_rel);
    if (cur < inputs.size()) {
      return &inputs[cur];
    }
    return {};
  };

  std::atomic<size_t>& counter;
  std::vector<int>& inputs;
};

struct Xform0 {
  TestOptional<int*> operator()(int* n) {
    int& num = *n;
    if (num > 2 && num < 5) {
      return {};
    }
    num *= num;
    return n;
  };
};

struct Xform1 {
  int* operator()(int* num) {
    *num += 5;
    return num;
  }
};

struct Sink {
  void operator()(int* num) {
    *num /= 2;
  }
};

TEST(Pipeline, MultiStageCarryPointers) {
  constexpr size_t kNumInputs = 1000;
  std::vector<int> inputs(kNumInputs);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<size_t> counter(0);

  Gen gen(counter, inputs);
  Xform0 xform0;
  Xform1 xform1;
  Sink sink;
  dispenso::pipeline(gen, xform0, xform1, sink);

  for (size_t i = 0; i < kNumInputs; ++i) {
    if (i <= 2 || i >= 5) {
      EXPECT_EQ(inputs[i], ((i * i) + 5) / 2);
    }
  }
}

TEST(Pipeline, MultiStageCarryPointers2) {
  constexpr size_t kNumInputs = 1000;
  std::vector<int> inputs(kNumInputs);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<size_t> counter(0);

  dispenso::pipeline(Gen(counter, inputs), Xform0(), Xform1(), Sink());

  for (size_t i = 0; i < kNumInputs; ++i) {
    if (i <= 2 || i >= 5) {
      EXPECT_EQ(inputs[i], ((i * i) + 5) / 2);
    }
  }
}

TEST(Pipeline, MultiStageCarryPointersMultiFilterParallel) {
  constexpr size_t kPar = 8;
  constexpr size_t kNumInputs = 1000;
  std::vector<int> inputs(kNumInputs);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<size_t> counter(0);

  dispenso::pipeline(
      dispenso::stage(Gen(counter, inputs), kPar),
      dispenso::stage(Xform0(), kPar),
      dispenso::stage(
          [&inputs](int* in) -> TestOptional<int*> {
            if (in - inputs.data() > 75 && in - inputs.data() < 80) {
              return {};
            }
            return in;
          },
          kPar),
      dispenso::stage(Xform1(), kPar),
      dispenso::stage(Sink(), kPar));

  for (size_t i = 0; i < kNumInputs; ++i) {
    if (i > 2 && i < 5) {
      EXPECT_EQ(inputs[i], i) << " at index " << i;
    } else if (i > 75 && i < 80) {
      EXPECT_EQ(inputs[i], i * i) << " at index " << i;
    } else {
      EXPECT_EQ(inputs[i], ((i * i) + 5) / 2) << " at index " << i;
    }
  }
}

TEST(Pipeline, MultiStageCarryPointersMultiFilterUnlimitedParallel) {
  constexpr size_t kPar = dispenso::kStageNoLimit;
  constexpr size_t kNumInputs = 1000;
  std::vector<int> inputs(kNumInputs);
  std::iota(inputs.begin(), inputs.end(), 0);
  std::atomic<size_t> counter(0);

  dispenso::pipeline(
      dispenso::stage(Gen(counter, inputs), kPar),
      dispenso::stage(Xform0(), kPar),
      dispenso::stage(
          [&inputs](int* in) -> TestOptional<int*> {
            if (in - inputs.data() > 75 && in - inputs.data() < 80) {
              return {};
            }
            return in;
          },
          kPar),
      dispenso::stage(Xform1(), kPar),
      dispenso::stage(Sink(), kPar));

  for (size_t i = 0; i < kNumInputs; ++i) {
    if (i > 2 && i < 5) {
      EXPECT_EQ(inputs[i], i) << " at index " << i;
    } else if (i > 75 && i < 80) {
      EXPECT_EQ(inputs[i], i * i) << " at index " << i;
    } else {
      EXPECT_EQ(inputs[i], ((i * i) + 5) / 2) << " at index " << i;
    }
  }
}

static size_t g_count = 0;

TestOptional<size_t> funkGen() {
  if (g_count < 10) {
    return g_count++;
  }
  return {};
}

static std::atomic<size_t> g_sum(0);
void funkSink(size_t in) {
  g_sum.fetch_add(in, std::memory_order_acq_rel);
}

TEST(Pipeline, PipelineFunctions) {
  g_count = 0;
  g_sum.store(0);

  dispenso::pipeline(funkGen, funkSink);

  EXPECT_EQ(45, g_sum.load(std::memory_order_acquire));
}

TEST(Pipeline, PipelineMoveOnly) {
  std::atomic<size_t> sum(0);

  dispenso::pipeline(
      [counter = 0]() mutable -> TestOptional<std::unique_ptr<size_t>> {
        if (counter < 10) {
          return std::make_unique<size_t>(counter++);
        }
        return {};
      },
      [](std::unique_ptr<size_t> val) {
        *val += 1;
        return val;
      },
      [&sum](std::unique_ptr<size_t> val) { sum.fetch_add(*val); });

  EXPECT_EQ(55, sum.load(std::memory_order_acquire));
}

TEST(Pipeline, PipelineMoveOnlyWithFiltering) {
  std::atomic<size_t> sum(0);

  dispenso::pipeline(
      [counter = 0]() mutable -> TestOptional<std::unique_ptr<size_t>> {
        if (counter < 10) {
          return std::make_unique<size_t>(counter++);
        }
        return {};
      },
      [](std::unique_ptr<size_t> val) -> TestOptional<std::unique_ptr<size_t>> {
        if (*val == 5) {
          return {};
        }
        *val += 1;
        return val;
      },
      [&sum](std::unique_ptr<size_t> val) { sum.fetch_add(*val); });

  EXPECT_EQ(49, sum.load(std::memory_order_acquire));
}
