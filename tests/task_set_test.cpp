/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/task_set.h>

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

void recursiveFunc(dispenso::ThreadPool& pool, int num) {
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

void buildTree(dispenso::ConcurrentTaskSet& tasks, std::unique_ptr<Node>& node, int depth) {
  if (depth) {
    node = std::make_unique<Node>();
    node->val = depth;
    tasks.schedule([&tasks, &left = node->left, depth]() { buildTree(tasks, left, depth - 1); });
    tasks.schedule([&tasks, &right = node->right, depth]() { buildTree(tasks, right, depth - 1); });
  }
}

void verifyTree(const std::unique_ptr<Node>& node, int depthRemaining) {
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
