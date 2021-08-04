// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <deque>
#include <list>
#include <set>
#include <unordered_map>
#include <vector>

#include <dispenso/for_each.h>

#include <gtest/gtest.h>

template <typename Container>
void forEachTestHelper() {
  constexpr size_t kNumVals = 1 << 15;
  Container c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace_back(i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each(
      std::begin(c), std::end(c), [&validated](size_t val) { validated[val] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEach, Vector) {
  forEachTestHelper<std::vector<size_t>>();
}

TEST(ForEach, List) {
  forEachTestHelper<std::list<size_t>>();
}

TEST(ForEach, Deque) {
  forEachTestHelper<std::deque<size_t>>();
}

TEST(ForEach, Set) {
  constexpr size_t kNumVals = 1 << 12;
  std::set<size_t> c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace(i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each(
      std::begin(c), std::end(c), [&validated](size_t val) { validated[val] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEach, UnorderedMap) {
  constexpr size_t kNumVals = 1 << 12;
  std::unordered_map<size_t, size_t> c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace(i, i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each(
      std::begin(c), std::end(c), [&validated](auto p) { validated[p.second] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

template <typename Container>
void forEachNTestHelper() {
  constexpr size_t kNumVals = 1 << 15;
  Container c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace_back(i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each_n(
      std::begin(c), kNumVals, [&validated](size_t val) { validated[val] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEachN, Vector) {
  forEachNTestHelper<std::vector<size_t>>();
}

TEST(ForEachN, List) {
  forEachNTestHelper<std::list<size_t>>();
}

TEST(ForEachN, Deque) {
  forEachNTestHelper<std::deque<size_t>>();
}

TEST(ForEachN, Set) {
  constexpr size_t kNumVals = 1 << 12;
  std::set<size_t> c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace(i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each_n(
      std::begin(c), kNumVals, [&validated](size_t val) { validated[val] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEachN, UnorderedMap) {
  constexpr size_t kNumVals = 1 << 12;
  std::unordered_map<size_t, size_t> c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace(i, i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each_n(
      std::begin(c), kNumVals, [&validated](auto p) { validated[p.second] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEach, NoWaitFewerThreads) {
  constexpr size_t kNumVals = 1 << 14;
  std::atomic<size_t> count(0);
  std::atomic<bool> canStart(false);

  std::vector<size_t> c;
  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace_back(i);
  }

  dispenso::TaskSet tasks(dispenso::globalThreadPool());

  dispenso::for_each(
      tasks,
      std::begin(c),
      std::end(c),
      [&count, &canStart](auto val) {
        while (!canStart.load(std::memory_order_acquire))
          ;

        count.fetch_add(val, std::memory_order_release);
      },
      {3, false} /* 3 threads, no wait */);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Use sequential consistency to ensure that this store can't be put before the sleep, nor after
  // the wait.
  canStart.store(true, std::memory_order_seq_cst);

  tasks.wait();

  EXPECT_EQ(count.load(std::memory_order_acquire), 134209536);
}

TEST(ForEachN, NoWaitFewerThreads) {
  constexpr size_t kNumVals = 1 << 13;
  std::atomic<size_t> count(0);
  std::atomic<bool> canStart(false);

  std::vector<size_t> c;
  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace_back(i);
  }

  dispenso::TaskSet tasks(dispenso::globalThreadPool());

  dispenso::for_each_n(
      tasks,
      std::begin(c),
      kNumVals,
      [&count, &canStart](auto val) {
        while (!canStart.load(std::memory_order_acquire))
          ;

        count.fetch_add(val, std::memory_order_release);
      },
      {3, false} /* 3 threads, no wait */);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Use sequential consistency to ensure that this store can't be put before the sleep, nor after
  // the wait.
  canStart.store(true, std::memory_order_seq_cst);

  tasks.wait();

  EXPECT_EQ(count.load(std::memory_order_acquire), 33550336);
}

TEST(ForEach, SmallSet) {
  constexpr size_t kNumVals = 3;
  std::set<size_t> c;

  for (size_t i = 0; i < kNumVals; ++i) {
    c.emplace(i);
  }

  std::vector<uint8_t> validated(kNumVals, 0);

  dispenso::for_each(
      std::begin(c), std::end(c), [&validated](size_t val) { validated[val] = true; });

  for (auto v : validated) {
    EXPECT_TRUE(v);
  }
}

TEST(ForEach, EmptySet) {
  std::set<int> emptySet;
  dispenso::for_each(std::begin(emptySet), std::end(emptySet), [](int /*index*/) {
    EXPECT_FALSE(true) << "Should not get into this lambda";
  });
}

TEST(ForEach, References) {
  std::vector<int> values;
  for (int i = 0; i < 100; ++i) {
    values.push_back(i);
  }

  dispenso::for_each(std::begin(values), std::end(values), [](int& i) { i = -i; });

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(values[i], -i);
  }
}

TEST(ForEach, Cascade) {
  std::vector<int> values;
  for (int i = 0; i < 100; ++i) {
    values.push_back(i);
  }

  dispenso::TaskSet taskSet(dispenso::globalThreadPool());
  dispenso::ForEachOptions options;
  options.wait = false;

  dispenso::for_each(
      taskSet,
      std::begin(values),
      std::end(values),
      [](int& i) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        i = -i;
      },
      options);

  dispenso::ForEachOptions waitOptions;
  std::set<int> emptySet;
  dispenso::for_each(
      taskSet,
      std::begin(emptySet),
      std::end(emptySet),
      [](int /*index*/) { EXPECT_FALSE(true) << "Should not get into this lambda"; },
      waitOptions);

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(values[i], -i);
  }
}
