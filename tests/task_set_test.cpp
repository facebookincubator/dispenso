// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/task_set.h>

#include <gtest/gtest.h>

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
  constexpr int64_t kWorkItems = 10000;
  std::vector<int64_t> outputsA(kWorkItems, 0);
  std::vector<int64_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSet(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSet, [i, &a]() { a = i * i; });
    schedule(taskSet, [i, &b]() { b = i * i * i; });
  }

  taskSet.wait();

  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }
}

TEST_P(TaskSetTest, MultiWait) {
  constexpr int64_t kWorkItems = 10000;
  std::vector<int64_t> outputsA(kWorkItems, 0);
  std::vector<int64_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSet(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSet, [i, &a]() { a = i * i; });
    schedule(taskSet, [i, &b]() { b = i * i * i; });
  }

  taskSet.wait();

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSet, [i, &c]() { c = i * i - 5; });
    schedule(taskSet, [i, &d]() { d = i * i * i - 5; });
  }

  taskSet.wait();

  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
}

TEST_P(TaskSetTest, MultiSet) {
  constexpr int64_t kWorkItems = 10000;
  std::vector<int64_t> outputsA(kWorkItems, 0);
  std::vector<int64_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSetA(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSetA, [i, &a]() { a = i * i; });
    schedule(taskSetA, [i, &b]() { b = i * i * i; });
  }

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  dispenso::TaskSet taskSetB(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSetB, [i, &c]() { c = i * i - 5; });
    schedule(taskSetB, [i, &d]() { d = i * i * i - 5; });
  }

  taskSetA.wait();
  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }

  taskSetB.wait();

  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
}

TEST_P(TaskSetTest, MultiSetTryWait) {
  constexpr int64_t kWorkItems = 10000;
  std::vector<int64_t> outputsA(kWorkItems, 0);
  std::vector<int64_t> outputsB(kWorkItems, 0);
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet taskSetA(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& a = outputsA[i];
    auto& b = outputsB[i];
    schedule(taskSetA, [i, &a]() { a = i * i; });
    schedule(taskSetA, [i, &b]() { b = i * i * i; });
    (void)taskSetA.tryWait(1);
  }

  std::vector<int64_t> outputsC(kWorkItems, 0);
  std::vector<int64_t> outputsD(kWorkItems, 0);
  dispenso::TaskSet taskSetB(pool);
  for (int64_t i = 0; i < kWorkItems; ++i) {
    auto& c = outputsC[i];
    auto& d = outputsD[i];
    schedule(taskSetB, [i, &c]() { c = i * i - 5; });
    schedule(taskSetB, [i, &d]() { d = i * i * i - 5; });
    (void)taskSetB.tryWait(1);
  }

  while (!taskSetA.tryWait(1)) {
  }
  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsA[i], i * i);
    EXPECT_EQ(outputsB[i], i * i * i);
  }

  taskSetB.wait();

  for (int64_t i = 0; i < kWorkItems; ++i) {
    EXPECT_EQ(outputsC[i], i * i - 5);
    EXPECT_EQ(outputsD[i], i * i * i - 5);
  }
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

#if defined(__cpp_exceptions)
TEST(TaskSet, Exception) {
  dispenso::ThreadPool pool(10);
  dispenso::TaskSet tasks(pool);

  int data;
  int* datap = &data;

  tasks.schedule(
      [datap]() {
        throw std::logic_error("oops");
        *datap = 5;
      },
      dispenso::ForceQueuingTag());

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

  int data;
  int* datap = &data;

  tasks.schedule(
      [datap]() {
        throw std::logic_error("oops");
        *datap = 5;
      },
      dispenso::ForceQueuingTag());

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

  int data;
  int* datap = &data;

  tasks.schedule([datap]() {
    throw std::logic_error("oops");
    *datap = 5;
  });

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

  int data;
  int* datap = &data;

  tasks.schedule([datap]() {
    throw std::logic_error("oops");
    *datap = 5;
  });

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
  int* datap = &data;

  tasks.schedule(
      [datap]() {
        throw std::logic_error("oops");
        *datap = 5;
      },
      dispenso::ForceQueuingTag());

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

  int data;
  int* datap = &data;

  tasks.schedule(
      [datap]() {
        throw std::logic_error("oops");
        *datap = 5;
      },
      dispenso::ForceQueuingTag());

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

  int data;
  int* datap = &data;

  tasks.schedule([datap]() {
    throw std::logic_error("oops");
    *datap = 5;
  });

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

  int data;
  int* datap = &data;

  tasks.schedule([datap]() {
    throw std::logic_error("oops");
    *datap = 5;
  });

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

#endif // __cpp_exceptions
