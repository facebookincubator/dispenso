/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/task_set.h>

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// Add a shim to account for older gtest
#if !defined INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif // INSTANTIATE_TEST_CASE_P

enum ScheduleType { kDefault, kForceQueue, kMixed };

class TaskSetTest : public ::testing::TestWithParam<ScheduleType> {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  template <typename TaskSetT, typename F>
  void schedule(TaskSetT& taskSet, F&& f) {
    switch (GetParam()) {
      case kDefault:
        taskSet.schedule(std::forward<F>(f));
        break;
      case kForceQueue:
        taskSet.schedule(std::forward<F>(f), dispenso::ForceQueuingTag());
        break;
      case kMixed:
        if (count_++ & 1) {
          taskSet.schedule(std::forward<F>(f));
        } else {
          taskSet.schedule(std::forward<F>(f), dispenso::ForceQueuingTag());
        }
        break;
    }
  }

 private:
  size_t count_ = 0;
};

INSTANTIATE_TEST_SUITE_P(
    TaskSetTestParameters,
    TaskSetTest,
    testing::Values(kDefault, kForceQueue, kMixed));

TEST_P(TaskSetTest, MixedWork) {
  constexpr size_t kWorkItems = 10000;
  std::vector<size_t> outputsA(kWorkItems, 0);
  std::vector<size_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSet(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSet, [i, &a]() { a = i * i; });
    schedule(taskSet, [i, &b]() { b = i * i * i; });
  }

  taskSet.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }
}

TEST_P(TaskSetTest, MultiWait) {
  constexpr size_t kWorkItems = 10000;
  std::vector<size_t> outputsA(kWorkItems, 0);
  std::vector<size_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSet(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSet, [i, &a]() { a = i * i; });
    schedule(taskSet, [i, &b]() { b = i * i * i; });
  }

  taskSet.wait();

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSet, [i, &c]() { c = static_cast<int64_t>(i * i - 5); });
    schedule(taskSet, [i, &d]() { d = static_cast<int64_t>(i * i * i - 5); });
  }

  taskSet.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
}

TEST_P(TaskSetTest, MultiSet) {
  constexpr size_t kWorkItems = 10000;
  std::vector<size_t> outputsA(kWorkItems, 0);
  std::vector<size_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSetA(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSetA, [i, &a]() { a = i * i; });
    schedule(taskSetA, [i, &b]() { b = i * i * i; });
  }

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  dispenso::TaskSet taskSetB(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSetB, [i, &c]() { c = static_cast<int64_t>(i * i - 5); });
    schedule(taskSetB, [i, &d]() { d = static_cast<int64_t>(i * i * i - 5); });
  }

  taskSetA.wait();
  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }

  taskSetB.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
}

TEST_P(TaskSetTest, MultiSetTryWait) {
  constexpr size_t kWorkItems = 10000;
  std::vector<size_t> outputsA(kWorkItems, 0);
  std::vector<size_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSetA(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSetA, [i, &a]() { a = i * i; });
    schedule(taskSetA, [i, &b]() { b = i * i * i; });
    (void)taskSetA.tryWait(1);
  }

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  dispenso::TaskSet taskSetB(pool);
  for (size_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSetB, [i, &c]() { c = static_cast<int64_t>(i * i - 5); });
    schedule(taskSetB, [i, &d]() { d = static_cast<int64_t>(i * i * i - 5); });
    (void)taskSetB.tryWait(1);
  }

  while (!taskSetA.tryWait(1)) {
  }
  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }

  taskSetB.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
}

TEST(TaskSetTest, ParamConstruction) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSetA(pool);
  dispenso::TaskSet taskSetB(pool, 4);
  dispenso::TaskSet taskSetC(pool, dispenso::ParentCascadeCancel::kOn);
  dispenso::TaskSet taskSetD(pool, dispenso::ParentCascadeCancel::kOff, 4);
}

TEST(ConcurrentTaskSetTest, ParamConstruction) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet taskSetA(pool);
  dispenso::ConcurrentTaskSet taskSetB(pool, 4);
  dispenso::ConcurrentTaskSet taskSetC(pool, dispenso::ParentCascadeCancel::kOn);
  dispenso::ConcurrentTaskSet taskSetD(pool, dispenso::ParentCascadeCancel::kOff, 4);
}

static void recursiveFunc(dispenso::ThreadPool& pool, int num) {
  if (num <= 0)
    return;
  std::atomic<int> value(0);
  dispenso::TaskSet taskSet(pool);
  for (int i = 0; i < num; ++i) {
    taskSet.schedule([i, &value, &pool]() {
      recursiveFunc(pool, i - 1);
      ++value;
    });
  }
  taskSet.wait();
  EXPECT_EQ(value.load(), num);
}

TEST(TaskSet, Recursive) {
  dispenso::ThreadPool pool(10);
  recursiveFunc(pool, 20);
}

struct Node {
  int val;
  std::unique_ptr<Node> left, right;
};

static void buildTree(dispenso::ConcurrentTaskSet& tasks, std::unique_ptr<Node>& node, int depth) {
  if (depth) {
    node = std::make_unique<Node>();
    node->val = depth;
    tasks.schedule([&tasks, &left = node->left, depth]() { buildTree(tasks, left, depth - 1); });
    tasks.schedule([&tasks, &right = node->right, depth]() { buildTree(tasks, right, depth - 1); });
  }
}

static void verifyTree(const std::unique_ptr<Node>& node, int depthRemaining) {
  if (depthRemaining) {
    ASSERT_EQ(node->val, depthRemaining);
    verifyTree(node->left, depthRemaining - 1);
    verifyTree(node->right, depthRemaining - 1);
  }
}

TEST(ConcurrentTaskSet, DoTree) {
  std::unique_ptr<Node> root;
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);
  buildTree(tasks, root, 20);
  tasks.wait();
  verifyTree(root, 20);
}

TEST(TaskSet, OneChildCancels) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.schedule(
      []() {
        while (!dispenso::parentTaskSet()->canceled())
          ;
      },
      dispenso::ForceQueuingTag());

  tasks.schedule([]() { dispenso::parentTaskSet()->cancel(); }, dispenso::ForceQueuingTag());
  EXPECT_TRUE(tasks.wait());
}

TEST(TaskSet, ParentThreadCancels) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.schedule(
      []() {
        while (!dispenso::parentTaskSet()->canceled())
          ;
      },
      dispenso::ForceQueuingTag());

  tasks.cancel();
  EXPECT_TRUE(tasks.wait());
}

TEST(TaskSet, CascadingCancelOne) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.schedule(
      [&pool]() {
        dispenso::TaskSet tasks2(pool, dispenso::ParentCascadeCancel::kOn);
        tasks2.schedule([]() {
          while (!dispenso::parentTaskSet()->canceled())
            ;
        });
      },
      dispenso::ForceQueuingTag());

  tasks.cancel();
  EXPECT_TRUE(tasks.wait());
}

TEST(TaskSet, CascadingOne) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int a = 5;

  tasks.schedule(
      [&pool, &a]() {
        dispenso::TaskSet tasks2(pool);
        tasks2.schedule([&a]() { a = 7; });
      },
      dispenso::ForceQueuingTag());

  EXPECT_FALSE(tasks.wait());
  EXPECT_EQ(a, 7);
}

TEST(TaskSet, CascadingManyCancel) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  // Use nonconst instead of constexpr to avoid older MSVC error
  size_t kBranchFactor = 200;

  std::vector<size_t> values(kBranchFactor * kBranchFactor);

  for (size_t i = 0; i < kBranchFactor; ++i) {
    tasks.schedule(
        [&pool, kBranchFactor]() {
          dispenso::TaskSet tasks2(pool, dispenso::ParentCascadeCancel::kOn);
          for (size_t j = 0; j < kBranchFactor; ++j) {
            tasks2.schedule([]() {
              while (!dispenso::parentTaskSet()->canceled()) {
              }
            });
          }
        },
        dispenso::ForceQueuingTag());
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tasks.cancel();

  EXPECT_TRUE(tasks.wait());
}

TEST(TaskSet, CascadingMany) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  // Use nonconst instead of constexpr to avoid older MSVC error
  size_t kBranchFactor = 200;

  std::vector<size_t> values(kBranchFactor * kBranchFactor);

  for (size_t i = 0; i < kBranchFactor; ++i) {
    tasks.schedule(
        [&pool, &values, i, kBranchFactor]() {
          dispenso::TaskSet tasks2(pool);
          for (size_t j = 0; j < kBranchFactor; ++j) {
            tasks2.schedule(
                [&values, i, j, kBranchFactor]() { values[i * kBranchFactor + j] = i + j; });
          }
        },
        dispenso::ForceQueuingTag());
  }
  EXPECT_FALSE(tasks.wait());
  for (size_t i = 0; i < kBranchFactor; ++i) {
    for (size_t j = 0; j < kBranchFactor; ++j) {
      EXPECT_EQ(i + j, values[i * kBranchFactor + j]);
    }
  }
}

#if defined(__cpp_exceptions)
TEST(TaskSet, Exception) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); }, dispenso::ForceQueuingTag());

  bool caught = false;
  try {
    tasks.wait();
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(ConcurrentTaskSet, Exception) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); }, dispenso::ForceQueuingTag());

  bool caught = false;
  try {
    tasks.wait();
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(TaskSet, ExceptionNoForceQueuing) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); });

  bool caught = false;
  try {
    tasks.wait();
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(ConcurrentTaskSet, ExceptionNoForceQueuing) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); });

  bool caught = false;
  try {
    tasks.wait();
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(TaskSet, ExceptionTryWait) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data = 32767;

  tasks.schedule([]() { throw std::logic_error("oops"); }, dispenso::ForceQueuingTag());

  bool caught = false;
  try {
    while (!tasks.tryWait(1)) {
    }
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(ConcurrentTaskSet, ExceptionTryWait) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); }, dispenso::ForceQueuingTag());

  bool caught = false;
  try {
    while (!tasks.tryWait(1)) {
    }
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(TaskSet, ExceptionNoForceQueuingTryWait) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); });

  bool caught = false;
  try {
    while (!tasks.tryWait(1)) {
    }
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(ConcurrentTaskSet, ExceptionNoForceQueuingTryWait) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  int data = 3;

  tasks.schedule([]() { throw std::logic_error("oops"); });

  bool caught = false;
  try {
    while (!tasks.tryWait(1)) {
    }
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

TEST(TaskSet, ExceptionCancels) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data = 3;

  tasks.schedule(
      []() {
        while (!dispenso::parentTaskSet()->canceled())
          ;
      },
      dispenso::ForceQueuingTag());

  tasks.schedule([]() { throw std::logic_error("oops"); }, dispenso::ForceQueuingTag());

  bool caught = false;
  try {
    tasks.wait();
    EXPECT_EQ(data, 12);
  } catch (...) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}

#endif // __cpp_exceptions

TEST(TaskSet, EmptyTaskSet) {
  // Test that an empty TaskSet can be waited on without issues
  dispenso::ThreadPool pool(4);
  dispenso::TaskSet tasks(pool);

  // wait() on empty TaskSet should return immediately with no cancellation
  EXPECT_FALSE(tasks.wait());

  // Should be able to reuse after empty wait
  std::atomic<int> counter{0};
  tasks.schedule([&counter]() { counter.fetch_add(1); });
  EXPECT_FALSE(tasks.wait());
  EXPECT_EQ(counter.load(), 1);
}

TEST(TaskSet, TryWaitOnEmpty) {
  dispenso::ThreadPool pool(4);
  dispenso::TaskSet tasks(pool);

  // tryWait on empty TaskSet should return true immediately
  EXPECT_TRUE(tasks.tryWait(0));
  EXPECT_TRUE(tasks.tryWait(1));
  EXPECT_TRUE(tasks.tryWait(100));
}

TEST(TaskSet, CanceledStateBeforeCancellation) {
  dispenso::ThreadPool pool(4);
  dispenso::TaskSet tasks(pool);

  // Before any cancellation, canceled() should return false
  EXPECT_FALSE(tasks.canceled());

  std::atomic<bool> sawNotCanceled{false};
  tasks.schedule([&sawNotCanceled]() {
    if (dispenso::parentTaskSet() && !dispenso::parentTaskSet()->canceled()) {
      sawNotCanceled.store(true);
    }
  });

  tasks.wait();
  EXPECT_TRUE(sawNotCanceled.load());
  EXPECT_FALSE(tasks.canceled());
}

TEST(TaskSet, SingleTask) {
  dispenso::ThreadPool pool(4);
  dispenso::TaskSet tasks(pool);

  int result = 0;
  tasks.schedule([&result]() { result = 42; });
  tasks.wait();

  EXPECT_EQ(result, 42);
}

TEST(TaskSet, GlobalThreadPool) {
  // Explicitly test with globalThreadPool()
  dispenso::TaskSet tasks(dispenso::globalThreadPool());

  std::atomic<int> sum{0};
  for (int i = 0; i < 100; ++i) {
    tasks.schedule([&sum, i]() { sum.fetch_add(i); });
  }
  tasks.wait();

  // Sum of 0..99 = 4950
  EXPECT_EQ(sum.load(), 4950);
}

TEST(ConcurrentTaskSet, EmptyTaskSet) {
  dispenso::ThreadPool pool(4);
  dispenso::ConcurrentTaskSet tasks(pool);

  // wait() on empty ConcurrentTaskSet should return immediately
  EXPECT_FALSE(tasks.wait());
}

TEST(ConcurrentTaskSet, TryWaitOnEmpty) {
  dispenso::ThreadPool pool(4);
  dispenso::ConcurrentTaskSet tasks(pool);

  EXPECT_TRUE(tasks.tryWait(0));
  EXPECT_TRUE(tasks.tryWait(1));
}

TEST(ConcurrentTaskSet, ConcurrentScheduling) {
  // Test that multiple threads can schedule to a ConcurrentTaskSet simultaneously
  dispenso::ThreadPool pool(8);
  dispenso::ConcurrentTaskSet tasks(pool);

  std::atomic<int> counter{0};
  static constexpr int kTasksPerThread = 100;
  static constexpr int kNumSchedulers = 4;

  // Create threads that will concurrently schedule tasks
  std::vector<std::thread> schedulers;
  for (int t = 0; t < kNumSchedulers; ++t) {
    schedulers.emplace_back([&]() {
      for (int i = 0; i < kTasksPerThread; ++i) {
        tasks.schedule([&counter]() { counter.fetch_add(1); });
      }
    });
  }

  // Wait for all schedulers to finish
  for (auto& t : schedulers) {
    t.join();
  }

  // Wait for all tasks to complete
  tasks.wait();

  EXPECT_EQ(counter.load(), kTasksPerThread * kNumSchedulers);
}

TEST(TaskSet, LargeBatchOfTasks) {
  dispenso::ThreadPool pool(8);
  dispenso::TaskSet tasks(pool);

  constexpr int kNumTasks = 10000;
  std::vector<int> results(kNumTasks, 0);

  for (int i = 0; i < kNumTasks; ++i) {
    tasks.schedule([&results, i]() { results[static_cast<size_t>(i)] = i * i; });
  }

  tasks.wait();

  for (int i = 0; i < kNumTasks; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)], i * i);
  }
}

TEST(ConcurrentTaskSet, CanceledState) {
  dispenso::ThreadPool pool(4);
  dispenso::ConcurrentTaskSet tasks(pool);

  EXPECT_FALSE(tasks.canceled());

  tasks.schedule(
      []() {
        while (!dispenso::parentTaskSet()->canceled()) {
          std::this_thread::yield();
        }
      },
      dispenso::ForceQueuingTag());

  // Give the task time to start
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  tasks.cancel();
  EXPECT_TRUE(tasks.canceled());
  EXPECT_TRUE(tasks.wait());
}

TEST(TaskSet, ScheduleBulkBasic) {
  constexpr size_t kWorkItems = 1000;
  std::vector<size_t> outputs(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.scheduleBulk(
      kWorkItems, [&outputs](size_t i) { return [i, &outputs]() { outputs[i] = i * i; }; });

  tasks.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputs[i], i * i);
  }
}

TEST(TaskSet, ScheduleBulkEdgeCases) {
  dispenso::ThreadPool pool(10);

  // Count=0: no-op
  {
    dispenso::TaskSet tasks(pool);
    bool called = false;
    tasks.scheduleBulk(0, [&called](size_t) {
      called = true;
      return []() {};
    });
    EXPECT_FALSE(tasks.wait());
    EXPECT_FALSE(called);
  }

  // Count=1: single task
  {
    dispenso::TaskSet tasks(pool);
    int result = 0;
    tasks.scheduleBulk(
        1, [&result](size_t i) { return [i, &result]() { result = static_cast<int>(i) + 42; }; });
    EXPECT_FALSE(tasks.wait());
    EXPECT_EQ(result, 42);
  }

  // Count=2: two tasks
  {
    dispenso::TaskSet tasks(pool);
    std::vector<int> outputs(2, 0);
    tasks.scheduleBulk(2, [&outputs](size_t i) {
      return [i, &outputs]() { outputs[i] = static_cast<int>(i) + 1; };
    });
    EXPECT_FALSE(tasks.wait());
    EXPECT_EQ(outputs[0], 1);
    EXPECT_EQ(outputs[1], 2);
  }
}

TEST(TaskSet, ScheduleBulkLarge) {
  constexpr size_t kWorkItems = 10000;
  std::vector<size_t> outputs(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.scheduleBulk(
      kWorkItems, [&outputs](size_t i) { return [i, &outputs]() { outputs[i] = i * i; }; });

  tasks.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputs[i], i * i);
  }
}

TEST(TaskSet, ScheduleBulkMultipleWaits) {
  constexpr size_t kWorkItems = 1000;
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  std::vector<size_t> outputsA(kWorkItems, 0);
  tasks.scheduleBulk(
      kWorkItems, [&outputsA](size_t i) { return [i, &outputsA]() { outputsA[i] = i + 1; }; });
  tasks.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i + 1);
  }

  std::vector<size_t> outputsB(kWorkItems, 0);
  tasks.scheduleBulk(
      kWorkItems, [&outputsB](size_t i) { return [i, &outputsB]() { outputsB[i] = i * 2; }; });
  tasks.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsB[i], i * 2);
  }
}

TEST(TaskSet, ScheduleBulkMixedWithSchedule) {
  static constexpr size_t kBulkItems = 500;
  static constexpr size_t kScheduleItems = 500;
  static constexpr size_t kTotal = kBulkItems + kScheduleItems + kBulkItems;
  std::vector<size_t> outputs(kTotal, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  // First bulk batch
  tasks.scheduleBulk(
      kBulkItems, [&outputs](size_t i) { return [i, &outputs]() { outputs[i] = i + 1; }; });

  // Individual schedule
  for (size_t i = kBulkItems; i < kBulkItems + kScheduleItems; ++i) {
    tasks.schedule([i, &outputs]() { outputs[i] = i + 1; });
  }

  // Second bulk batch
  tasks.scheduleBulk(kBulkItems, [&outputs](size_t i) {
    size_t idx = i + kBulkItems + kScheduleItems;
    return [idx, &outputs]() { outputs[idx] = idx + 1; };
  });

  tasks.wait();

  for (size_t i = 0; i < kTotal; ++i) {
    EXPECT_EQ(outputs[i], i + 1);
  }
}

TEST(TaskSet, ScheduleBulkCancellation) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  // Schedule spinning tasks via individual schedule with ForceQueuingTag
  // to ensure they land on pool threads
  for (int i = 0; i < 4; ++i) {
    tasks.schedule(
        []() {
          while (!dispenso::parentTaskSet()->canceled()) {
            std::this_thread::yield();
          }
        },
        dispenso::ForceQueuingTag());
  }

  // Bulk-schedule more work
  std::vector<int> outputs(100, 0);
  tasks.scheduleBulk(100, [&outputs](size_t i) {
    return [i, &outputs]() { outputs[i] = static_cast<int>(i) + 1; };
  });

  tasks.cancel();
  EXPECT_TRUE(tasks.wait());
}

#if defined(__cpp_exceptions)
TEST(TaskSet, ScheduleBulkException) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  tasks.scheduleBulk(100, [](size_t i) {
    return [i]() {
      if (i == 50) {
        throw std::logic_error("bulk oops");
      }
    };
  });

  bool caught = false;
  try {
    tasks.wait();
  } catch (const std::logic_error&) {
    caught = true;
  }

  EXPECT_TRUE(caught);
}
#endif // __cpp_exceptions

TEST(ConcurrentTaskSet, ScheduleBulkBasic) {
  constexpr size_t kWorkItems = 1000;
  std::vector<size_t> outputs(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  tasks.scheduleBulk(
      kWorkItems, [&outputs](size_t i) { return [i, &outputs]() { outputs[i] = i * i; }; });

  tasks.wait();

  for (size_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputs[i], i * i);
  }
}

TEST(ConcurrentTaskSet, ScheduleBulkEdgeCases) {
  dispenso::ThreadPool pool(10);

  // Count=0: no-op
  {
    dispenso::ConcurrentTaskSet tasks(pool);
    bool called = false;
    tasks.scheduleBulk(0, [&called](size_t) {
      called = true;
      return []() {};
    });
    EXPECT_FALSE(tasks.wait());
    EXPECT_FALSE(called);
  }

  // Count=1: single task
  {
    dispenso::ConcurrentTaskSet tasks(pool);
    std::atomic<int> result{0};
    tasks.scheduleBulk(1, [&result](size_t i) {
      return [i, &result]() { result.store(static_cast<int>(i) + 42); };
    });
    EXPECT_FALSE(tasks.wait());
    EXPECT_EQ(result.load(), 42);
  }

  // Count=2: two tasks
  {
    dispenso::ConcurrentTaskSet tasks(pool);
    std::atomic<int> count{0};
    tasks.scheduleBulk(2, [&count](size_t) { return [&count]() { count.fetch_add(1); }; });
    EXPECT_FALSE(tasks.wait());
    EXPECT_EQ(count.load(), 2);
  }
}

TEST(ConcurrentTaskSet, ScheduleBulkConcurrent) {
  static constexpr int kTasksPerThread = 100;
  static constexpr int kNumSchedulers = 4;
  std::atomic<int> counter{0};
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  std::vector<std::thread> schedulers;
  for (int t = 0; t < kNumSchedulers; ++t) {
    schedulers.emplace_back([&tasks, &counter]() {
      tasks.scheduleBulk(static_cast<size_t>(kTasksPerThread), [&counter](size_t) {
        return [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); };
      });
    });
  }

  for (auto& t : schedulers) {
    t.join();
  }

  tasks.wait();
  EXPECT_EQ(counter.load(), kTasksPerThread * kNumSchedulers);
}

TEST(ConcurrentTaskSet, ScheduleBulkMixedWithSchedule) {
  static constexpr size_t kBulkItems = 500;
  static constexpr size_t kScheduleItems = 500;
  static constexpr size_t kTotal = kBulkItems + kScheduleItems + kBulkItems;
  std::vector<std::atomic<int>> outputs(kTotal);
  for (auto& o : outputs) {
    o.store(0);
  }
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  // First bulk batch
  tasks.scheduleBulk(kBulkItems, [&outputs](size_t i) {
    return [i, &outputs]() { outputs[i].store(static_cast<int>(i) + 1); };
  });

  // Individual schedule
  for (size_t i = kBulkItems; i < kBulkItems + kScheduleItems; ++i) {
    tasks.schedule([i, &outputs]() { outputs[i].store(static_cast<int>(i) + 1); });
  }

  // Second bulk batch
  tasks.scheduleBulk(kBulkItems, [&outputs](size_t i) {
    size_t idx = i + kBulkItems + kScheduleItems;
    return [idx, &outputs]() { outputs[idx].store(static_cast<int>(idx) + 1); };
  });

  tasks.wait();

  for (size_t i = 0; i < kTotal; ++i) {
    EXPECT_EQ(outputs[i].load(), static_cast<int>(i) + 1);
  }
}

TEST(ConcurrentTaskSet, ScheduleBulkCancellation) {
  dispenso::ThreadPool pool(10);
  dispenso::ConcurrentTaskSet tasks(pool);

  // Schedule spinning tasks via individual schedule with ForceQueuingTag
  for (int i = 0; i < 4; ++i) {
    tasks.schedule(
        []() {
          while (!dispenso::parentTaskSet()->canceled()) {
            std::this_thread::yield();
          }
        },
        dispenso::ForceQueuingTag());
  }

  // Bulk-schedule more work
  std::atomic<int> bulkCounter{0};
  tasks.scheduleBulk(
      100, [&bulkCounter](size_t) { return [&bulkCounter]() { bulkCounter.fetch_add(1); }; });

  tasks.cancel();
  EXPECT_TRUE(tasks.wait());
}
