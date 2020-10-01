// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE.md file in the root directory of this source tree.

#include <dispenso/future.h>

#include <gtest/gtest.h>

TEST(Future, Invalid) {
  dispenso::Future<void> future;
  EXPECT_FALSE(future.valid());
  future = dispenso::make_ready_future();
  EXPECT_TRUE(future.valid());
}

TEST(Future, MakeReady) {
  auto voidFuture = dispenso::make_ready_future();
  EXPECT_TRUE(voidFuture.is_ready());
  voidFuture.get();

  int valueA = 66;
  auto intFuture = dispenso::make_ready_future(valueA);
  EXPECT_TRUE(intFuture.is_ready());
  EXPECT_EQ(valueA, intFuture.get());

  auto intRefFuture = dispenso::make_ready_future(std::ref(valueA));
  EXPECT_TRUE(intRefFuture.is_ready());
  EXPECT_EQ(valueA, intRefFuture.get());
  --valueA;
  EXPECT_EQ(valueA, intRefFuture.get());

  EXPECT_EQ(&valueA, &intRefFuture.get());
}

TEST(Future, ThreadPool) {
  int foo = 10;
  dispenso::Future<void> voidFuture([&foo]() { foo = 7; }, dispenso::globalThreadPool());
  EXPECT_TRUE(voidFuture.valid());
  voidFuture.get();
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture([]() { return 33; }, dispenso::globalThreadPool());
  EXPECT_TRUE(intFuture.valid());
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture(
      [&refTo]() -> int& { return refTo; }, dispenso::globalThreadPool());
  EXPECT_TRUE(refFuture.valid());
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, ThreadPoolForceQueuing) {
  int foo = 10;
  dispenso::Future<void> voidFuture(
      [&foo]() { foo = 7; }, dispenso::globalThreadPool(), std::launch::async);
  EXPECT_TRUE(voidFuture.valid());
  voidFuture.get();
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture(
      []() { return 33; }, dispenso::globalThreadPool(), std::launch::async);
  EXPECT_TRUE(intFuture.valid());
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture(
      [&refTo]() -> int& { return refTo; }, dispenso::globalThreadPool(), std::launch::async);
  EXPECT_TRUE(refFuture.valid());
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, TaskSet) {
  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  int foo = 10;
  dispenso::Future<void> voidFuture([&foo]() { foo = 7; }, tasks);
  EXPECT_TRUE(voidFuture.valid());
  voidFuture.get();
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture([]() { return 33; }, tasks);
  EXPECT_TRUE(intFuture.valid());
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture([&refTo]() -> int& { return refTo; }, tasks);
  EXPECT_TRUE(refFuture.valid());
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, TaskSetForceQueuing) {
  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  int foo = 10;
  dispenso::Future<void> voidFuture([&foo]() { foo = 7; }, tasks, std::launch::async);
  EXPECT_TRUE(voidFuture.valid());
  voidFuture.get();
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture([]() { return 33; }, tasks, std::launch::async);
  EXPECT_TRUE(intFuture.valid());
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture([&refTo]() -> int& { return refTo; }, tasks, std::launch::async);
  EXPECT_TRUE(refFuture.valid());
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, TaskSetWaitImpliesImmediatelyAvailable) {
  dispenso::TaskSet tasks(dispenso::globalThreadPool());
  int foo = 10;
  dispenso::Future<void> voidFuture([&foo]() { foo = 7; }, tasks);
  EXPECT_TRUE(voidFuture.valid());
  tasks.wait();
  EXPECT_TRUE(voidFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, voidFuture.wait_for(std::chrono::microseconds(1)));
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture([]() { return 33; }, tasks);
  EXPECT_TRUE(intFuture.valid());
  tasks.wait();
  EXPECT_TRUE(intFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, intFuture.wait_for(std::chrono::microseconds(1)));
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture([&refTo]() -> int& { return refTo; }, tasks);
  EXPECT_TRUE(refFuture.valid());
  tasks.wait();
  EXPECT_TRUE(refFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, refFuture.wait_for(std::chrono::microseconds(1)));
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, ConcurrentTaskSetWaitImpliesImmediatelyAvailable) {
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());
  int foo = 10;
  dispenso::Future<void> voidFuture([&foo]() { foo = 7; }, tasks);
  EXPECT_TRUE(voidFuture.valid());
  tasks.wait();
  EXPECT_TRUE(voidFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, voidFuture.wait_for(std::chrono::microseconds(1)));
  EXPECT_EQ(foo, 7);

  dispenso::Future<int> intFuture([]() { return 33; }, tasks, std::launch::async);
  EXPECT_TRUE(intFuture.valid());
  tasks.wait();
  EXPECT_TRUE(intFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, intFuture.wait_for(std::chrono::microseconds(1)));
  int result = intFuture.get();
  EXPECT_EQ(result, 33);

  int refTo = 77;
  dispenso::Future<int&> refFuture([&refTo]() -> int& { return refTo; }, tasks);
  EXPECT_TRUE(refFuture.valid());
  tasks.wait();
  EXPECT_TRUE(refFuture.is_ready());
  EXPECT_EQ(std::future_status::ready, refFuture.wait_for(std::chrono::microseconds(1)));
  int& ref = refFuture.get();
  EXPECT_EQ(&ref, &refTo);
  EXPECT_EQ(77, ref);
}

TEST(Future, LongRunMultipleWaitFor) {
  std::atomic<int> synchronizer(0);
  dispenso::Future<int> intFuture(
      [&synchronizer]() {
        while (synchronizer.load(std::memory_order_relaxed) < 10) {
        }
        return 77;
      },
      dispenso::globalThreadPool(),
      std::launch::async,
      // Note that kNotDeferred is required in this case, because otherwise the functor could run on
      // the current thread in wait_for below, which will hang because the synchronizer will never
      // be incemented.  This is not a common pattern, but it is the reason why kNotDeferred exists.
      dispenso::kNotDeferred);

  size_t loopCount = 0;
  while (intFuture.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
    synchronizer.fetch_add(1, std::memory_order_relaxed);
    ++loopCount;
  }
  int result = intFuture.get();

  EXPECT_GE(loopCount, 10);
  EXPECT_EQ(77, result);
}

TEST(Future, ShareInnerScopeWaits) {
  dispenso::Future<int> intFuture([]() { return 7; }, dispenso::globalThreadPool());

  const int* address;
  {
    auto sharedFuture = intFuture;

    EXPECT_EQ(7, sharedFuture.get());
    address = &sharedFuture.get();
  }
  EXPECT_TRUE(intFuture.is_ready());
  EXPECT_EQ(address, &intFuture.get());
}

TEST(Future, BasicLoop) {
  constexpr int kWorkItems = 10000;
  std::vector<dispenso::Future<int>> outputs;

  dispenso::ThreadPool pool(10);
  for (int i = 0; i < kWorkItems; ++i) {
    outputs.emplace_back([i]() { return i * i; }, pool);
  }

  int i = 0;
  for (auto& o : outputs) {
    EXPECT_EQ(o.get(), i * i);
    ++i;
  }
}

TEST(Future, CheckBackwards) {
  constexpr int kWorkItems = 10000;
  std::vector<dispenso::Future<int>> outputs;

  dispenso::ThreadPool pool(10);
  for (int i = 0; i < kWorkItems; ++i) {
    outputs.emplace_back([i]() { return i * i; }, pool);
  }

  for (int i = kWorkItems; i--;) {
    EXPECT_EQ(outputs[i].get(), i * i);
  }
}

TEST(Future, Async) {
  int value = 0;
  auto voidFuture = dispenso::async([&value]() { value = 66; });
  voidFuture.get();
  EXPECT_EQ(66, value);

  auto intFuture = dispenso::async([]() { return 77; });
  EXPECT_EQ(77, intFuture.get());

  auto refFuture = dispenso::async([&value]() -> int& { return value; });
  EXPECT_EQ(&value, &refFuture.get());
}

TEST(Future, AsyncNotDeferred) {
  int value = 0;
  auto voidFuture = dispenso::async(dispenso::kNotDeferred, [&value]() { value = 66; });
  voidFuture.get();
  EXPECT_EQ(66, value);

  auto intFuture = dispenso::async(dispenso::kNotDeferred, []() { return 77; });
  EXPECT_EQ(77, intFuture.get());

  auto refFuture = dispenso::async(dispenso::kNotDeferred, [&value]() -> int& { return value; });
  EXPECT_EQ(&value, &refFuture.get());
}

TEST(Future, AsyncNotAsync) {
  int value = 0;
  auto voidFuture = dispenso::async(dispenso::kNotAsync, [&value]() { value = 66; });
  voidFuture.get();
  EXPECT_EQ(66, value);

  auto intFuture = dispenso::async(dispenso::kNotAsync, []() { return 77; });
  EXPECT_EQ(77, intFuture.get());

  auto refFuture = dispenso::async(dispenso::kNotAsync, [&value]() -> int& { return value; });
  EXPECT_EQ(&value, &refFuture.get());
}

struct Node {
  int value;
  std::unique_ptr<Node> left, right;
};

void validateTree(Node* node, int val, int depth) {
  if (!node) {
    return;
  }

  EXPECT_EQ(node->value, val);

  validateTree(node->left.get(), val + depth, depth + 1);
  validateTree(node->right.get(), val + depth, depth + 1);
}

void buildTree(Node* node, int depth) {
  if (depth == 16) {
    return;
  }

  node->left = std::make_unique<Node>();
  node->right = std::make_unique<Node>();
  node->left->value = node->right->value = node->value + depth;
  auto lres = dispenso::async([left = node->left.get(), depth]() { buildTree(left, depth + 1); });
  auto rres =
      dispenso::async([right = node->right.get(), depth]() { buildTree(right, depth + 1); });

  lres.get();
  rres.get();
}

TEST(Future, RecursivelyBuildTree) {
  Node root;
  root.value = 77;
  buildTree(&root, 0);

  validateTree(&root, 77, 0);
}

TEST(Future, BasicThenUsage) {
  int value;
  auto voidFuture =
      dispenso::async([]() { return 77; }).then([&value](dispenso::Future<int>&& parent) {
        value = parent.get();
      });
  voidFuture.get();
  EXPECT_EQ(77, value);
  auto intFuture = dispenso::async([]() { return 55; }).then([](dispenso::Future<int>&& parent) {
    return parent.get();
  });
  EXPECT_EQ(55, intFuture.get());
  auto int2Future = dispenso::async([&value]() -> int& { return value; })
                        .then([](dispenso::Future<int&>&& parent) { return parent.get(); });
  EXPECT_EQ(77, int2Future.get());

  auto refFuture = dispenso::async([&value]() -> int& { return value; })
                       .then([](dispenso::Future<int&>&& parent) -> int& { return parent.get(); });
  EXPECT_EQ(77, refFuture.get());
  EXPECT_EQ(&value, &refFuture.get());

  intFuture =
      dispenso::async([&value]() { value = 33; }).then([&value](dispenso::Future<void>&& parent) {
        return value;
      });
  EXPECT_EQ(33, intFuture.get());

  int* valuePtr;
  voidFuture =
      dispenso::async([&value]() -> int& { return value; })
          .then([&valuePtr](dispenso::Future<int&>&& parent) { valuePtr = &parent.get(); });
  voidFuture.get();
  EXPECT_EQ(valuePtr, &value);
}

TEST(Future, LongerThenChain) {
  const int inval = 123;
  int outval;
  auto intFuture = dispenso::async([&inval]() -> const int& { return inval; })
                       .then([](dispenso::Future<const int&>&& parent) { return parent.get(); })
                       .then([&outval](dispenso::Future<int>&& parent) {
                         outval = parent.get();
                         return parent.get();
                       });
  EXPECT_EQ(inval, intFuture.get());
  EXPECT_EQ(inval, outval);
}

TEST(Future, MultiThenReadyAllInline) {
  dispenso::Future<int> intFuture = dispenso::make_ready_future(128);
  auto a = intFuture.then(
      [](dispenso::Future<int>&& parent) { return parent.get(); }, dispenso::kImmediateInvoker);
  auto b = intFuture.then(
      [](dispenso::Future<int>&& parent) { return parent.get() * parent.get(); },
      dispenso::kImmediateInvoker);
  EXPECT_EQ(intFuture.get(), 128);
  EXPECT_EQ(a.get(), 128);
  EXPECT_EQ(b.get(), 128 * 128);
}

TEST(Future, MultiThenReadyDelayedOrigin) {
  dispenso::Future<int> intFuture = dispenso::async([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 128;
  });
  auto a = intFuture.then(
      [](dispenso::Future<int>&& parent) { return parent.get(); }, dispenso::kImmediateInvoker);
  auto b = intFuture.then(
      [](dispenso::Future<int>&& parent) { return parent.get() * parent.get(); },
      dispenso::kImmediateInvoker);
  EXPECT_EQ(intFuture.get(), 128);
  EXPECT_EQ(a.get(), 128);
  EXPECT_EQ(b.get(), 128 * 128);
}

TEST(Future, MultiThenReadyDelayedOriginNotImmediateThen) {
  dispenso::Future<int> intFuture = dispenso::async([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 128;
  });
  auto a = intFuture.then([](dispenso::Future<int>&& parent) { return parent.get(); });
  auto b =
      intFuture.then([](dispenso::Future<int>&& parent) { return parent.get() * parent.get(); });
  EXPECT_EQ(intFuture.get(), 128);
  EXPECT_EQ(a.get(), 128);
  EXPECT_EQ(b.get(), 128 * 128);
}

TEST(Future, MultiThenReadyDelayedOriginTightLoop) {
  std::atomic<bool> originReady{false};
  dispenso::Future<int> intFuture = dispenso::async([&originReady]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    originReady.store(true);
    return 128;
  });

  std::vector<dispenso::Future<int>> futureVec;

  int extraLoops = 100;
  while (true) {
    futureVec.emplace_back(
        intFuture.then([](dispenso::Future<int>&& parent) { return parent.get() * parent.get(); }));
    if (originReady.load(std::memory_order_relaxed)) {
      if (extraLoops) {
        --extraLoops;
      } else {
        break;
      }
    }
  }

  EXPECT_GE(futureVec.size(), 101);

  for (auto& f : futureVec) {
    EXPECT_EQ(f.get(), 128 * 128);
  }
}

TEST(Future, ImmediateInvoker) {
  // Don't write code like this in real life.  It is (much) cheaper to use make_ready_future.
  dispenso::Future<int> future([]() { return 333; }, dispenso::kImmediateInvoker);
  EXPECT_TRUE(future.is_ready());
  EXPECT_EQ(333, future.get());
}

TEST(Future, NewThreadInvoker) {
  // Nearly always it should be better to use async/ThreadPool/TaskSet, but there may be occasions
  // where you actually want the work done on a new thread.
  dispenso::Future<int> future([]() { return 333; }, dispenso::kNewThreadInvoker);
  EXPECT_EQ(333, future.get());
}

#if defined(__cpp_exceptions)
TEST(Future, SimpleExceptions) {
  bool handledException = false;
  auto intFuture = dispenso::async([]() {
    throw(std::logic_error("oops"));
    return 333;
  });

  try {
    EXPECT_EQ(777, intFuture.get());
  } catch (std::logic_error) {
    handledException = true;
  }
  EXPECT_TRUE(handledException);
}

TEST(Future, SimpleExceptionsReference) {
  bool handledException = false;
  int val = 333;
  auto refFuture = dispenso::async([&val]() -> int& {
    throw(std::logic_error("oops"));
    return val;
  });

  try {
    EXPECT_EQ(777, refFuture.get());
  } catch (std::logic_error) {
    handledException = true;
  }
  EXPECT_TRUE(handledException);
}

TEST(Future, SimpleExceptionsVoid) {
  bool handledException = false;
  int val = 333;
  auto voidFuture = dispenso::async([&val]() -> int& {
    throw(std::logic_error("oops"));
    val = 222;
  });

  try {
    voidFuture.get();
    EXPECT_EQ(999, val);
  } catch (std::logic_error) {
    handledException = true;
  }
  EXPECT_TRUE(handledException);
}

TEST(Future, ThenExceptions) {
  bool handledException = false;
  auto intFuture = dispenso::async([]() {
    throw(std::logic_error("oops"));
    return 333;
  });

  try {
    auto resultingFuture = intFuture.then([](auto&& parent) { return parent.get(); });

    EXPECT_EQ(777, resultingFuture.get());
  } catch (std::logic_error) {
    handledException = true;
  }
  EXPECT_TRUE(handledException);
}

struct SomeType {
  SomeType(int init) : ptr(new int(init)) {}

  SomeType(const SomeType& oth) : ptr(new int(*oth.ptr)) {}

  SomeType& operator=(const SomeType& oth) {
    if (&oth != this) {
      ptr = new int(*oth.ptr);
    }
    return *this;
  }

  ~SomeType() {
    delete ptr;
  }
  int* ptr;
};

TEST(Future, ExceptionShouldntDestroyResultIfNotCreated) {
  auto noExcept = dispenso::async([]() { return SomeType(5); });

  auto res = noExcept.get();
  EXPECT_EQ(*res.ptr, 5);

  auto withExcept = dispenso::async([]() {
    throw(std::logic_error("oops"));
    return SomeType(5);
  });

  bool handledException = false;
  try {
    res = withExcept.get();
    EXPECT_EQ(777, *res.ptr);
  } catch (std::logic_error) {
    handledException = true;
  }
  EXPECT_TRUE(handledException);
}
#endif //__cpp_exceptions

TEST(Future, WhenAllEmptyVector) {
  std::deque<dispenso::Future<int>> items;
  auto dummy = dispenso::when_all(items.begin(), items.end()).then([](auto&& readyFutures) {
    EXPECT_TRUE(readyFutures.is_ready());
    auto& vec = readyFutures.get();
    EXPECT_EQ(0, vec.size());
  });

  dummy.wait();
}

TEST(Future, WhenAllVector) {
  constexpr int kWorkItems = 10000;
  std::deque<dispenso::Future<int>> items;

  dispenso::ThreadPool pool(10);
  int64_t expectedSum = 0;
  for (int i = 0; i < kWorkItems; ++i) {
    items.emplace_back([i]() { return i * i; }, pool);
    expectedSum += i * i;
  }

  auto finalSum = dispenso::when_all(items.begin(), items.end()).then([](auto&& readyFutures) {
    EXPECT_TRUE(readyFutures.is_ready());
    auto& vec = readyFutures.get();
    int64_t sum = 0;
    for (auto& f : vec) {
      EXPECT_TRUE(f.is_ready());
      sum += f.get();
    }
    return sum;
  });

  EXPECT_EQ(finalSum.get(), expectedSum);
}

TEST(Future, WhenAllEmptyTuple) {
  auto dummy =
      dispenso::when_all().then([](auto&& readyFutures) { EXPECT_TRUE(readyFutures.is_ready()); });
  dummy.wait();
}

TEST(Future, WhenAllTuple) {
  int value = 16;
  int value2 = 1024;

  dispenso::Future<int> intFuture([]() { return 77; }, dispenso::globalThreadPool());
  dispenso::Future<void> voidFuture([&value]() { value = 88; }, dispenso::globalThreadPool());
  dispenso::Future<float> floatFuture([]() { return 99.0f; }, dispenso::globalThreadPool());
  dispenso::Future<int&> refFuture(
      [&value2]() -> int& { return value2; }, dispenso::globalThreadPool());

  auto finalSum = dispenso::when_all(intFuture, voidFuture, floatFuture, refFuture)
                      .then([&value, &value2](auto&& readyFutures) {
                        EXPECT_TRUE(readyFutures.is_ready());

                        auto& tuple = readyFutures.get();
                        EXPECT_EQ(std::get<0>(tuple).get(), 77);
                        EXPECT_EQ(value, 88);
                        EXPECT_EQ(std::get<2>(tuple).get(), 99.0f);
                        EXPECT_EQ(&std::get<3>(tuple).get(), &value2);
                        return std::get<0>(tuple).get() + value + std::get<2>(tuple).get() +
                            std::get<3>(tuple).get();
                      });

  EXPECT_EQ(finalSum.get(), 1288);
}

inline std::unique_ptr<Node> nodeMove(const std::unique_ptr<Node>& current) {
  return std::move(const_cast<std::unique_ptr<Node>&>(current));
}

dispenso::Future<std::unique_ptr<Node>> makeTree(uint32_t depth, std::atomic<uint32_t>& cur) {
  --depth;
  auto node = std::make_unique<Node>();
  node->value = cur.fetch_add(1, std::memory_order_relaxed);
  if (!depth) {
    return dispenso::make_ready_future(std::move(node));
  }

  // TODO(bbudge): Can we make this nicer via unwrapping constructor?
  auto left = dispenso::async([depth, &cur]() { return makeTree(depth, cur); });
  auto right = dispenso::async([depth, &cur]() { return makeTree(depth, cur); });

  return dispenso::when_all(left, right).then([node = std::move(node)](auto&& both) mutable {
    auto& tuple = both.get();
    auto& futFutLeft = std::get<0>(tuple);
    auto& futFutRight = std::get<1>(tuple);
    node->left = nodeMove(futFutLeft.get().get());
    node->right = nodeMove(futFutRight.get().get());
    return std::move(node);
  });
}

void fillVector(std::unique_ptr<Node>& node, std::vector<uint32_t>& values) {
  if (!node) {
    return;
  }
  if (values.size() <= node->value) {
    values.resize(node->value + 1, 0);
  }
  ++values[node->value];
  fillVector(node->left, values);
  fillVector(node->right, values);
}

TEST(Future, WhenAllTreeBuild) {
  std::atomic<uint32_t> val(0);
  auto result = makeTree(6, val);

  std::vector<uint32_t> values(16, 0);

  std::unique_ptr<Node> root = nodeMove(result.get());

  fillVector(root, values);

  for (auto& v : values) {
    EXPECT_EQ(v, 1);
  }
}

dispenso::Future<std::unique_ptr<Node>> makeTreeIters(uint32_t depth, std::atomic<uint32_t>& cur) {
  --depth;
  auto node = std::make_unique<Node>();
  node->value = cur.fetch_add(1, std::memory_order_relaxed);
  if (!depth) {
    return dispenso::make_ready_future(std::move(node));
  }

  // TODO(bbudge): Can we make this nicer via unwrapping constructor?
  auto left = dispenso::async([depth, &cur]() { return makeTree(depth, cur); });
  auto right = dispenso::async([depth, &cur]() { return makeTree(depth, cur); });

  std::array<decltype(left), 2> children = {left, right};

  return dispenso::when_all(std::begin(children), std::end(children))
      .then([node = std::move(node)](auto&& both) mutable {
        auto& vec = both.get();
        EXPECT_EQ(vec.size(), 2);
        auto& futFutLeft = vec[0];
        auto& futFutRight = vec[1];
        node->left = nodeMove(futFutLeft.get().get());
        node->right = nodeMove(futFutRight.get().get());
        return std::move(node);
      });
}

TEST(Future, WhenAllTreeBuildIters) {
  std::atomic<uint32_t> val(0);
  auto result = makeTreeIters(6, val);

  std::vector<uint32_t> values(16, 0);

  std::unique_ptr<Node> root = nodeMove(result.get());

  fillVector(root, values);

  for (auto& v : values) {
    EXPECT_EQ(v, 1);
  }
}

// Pretty convoluted, but there was a bug where taskset wait does not imply the future is finished.
TEST(Future, TaskSetWaitImpliesFinished) {
  std::atomic<int> status(0);
  std::atomic<int> sidelineResult(0);
  dispenso::ConcurrentTaskSet tasks(dispenso::globalThreadPool());

  std::thread waiterThread([&tasks, &status, &sidelineResult]() {
    while (status.load(std::memory_order_acquire) == 0) {
    }
    tasks.wait();
    EXPECT_EQ(sidelineResult.load(std::memory_order_acquire), 1);
  });

  dispenso::Future<int> intFuture(
      [&sidelineResult, &status]() {
        status.store(1, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        sidelineResult.store(1, std::memory_order_release);
        return 77;
      },
      tasks);

  EXPECT_EQ(77, intFuture.get());

  waiterThread.join();
}

TEST(Future, TaskSetWaitImpliesWhenAllFinished) {
  std::vector<dispenso::Future<int>> futures;
  dispenso::TaskSet taskSet(dispenso::globalThreadPool());

  for (size_t i = 0; i < 100; ++i) {
    futures.emplace_back(dispenso::Future<int>([i]() { return i; }, taskSet));
  }

  auto result = dispenso::when_all(taskSet, futures.begin(), futures.end())
                    .then(
                        [](auto&& future) {
                          auto& vec = future.get();
                          int total = 0;
                          for (auto& f : vec) {
                            total += f.get();
                          }
                        },
                        taskSet);

  taskSet.wait();
  EXPECT_TRUE(result.is_ready());
}

TEST(Future, ConcurrentTaskSetWaitImpliesWhenAllFinished) {
  std::vector<dispenso::Future<int>> futures;
  dispenso::ConcurrentTaskSet taskSet(dispenso::globalThreadPool());

  for (size_t i = 0; i < 100; ++i) {
    futures.emplace_back(dispenso::Future<int>([i]() { return i; }, taskSet));
  }

  auto result = dispenso::when_all(taskSet, futures.begin(), futures.end())
                    .then(
                        [](auto&& future) {
                          auto& vec = future.get();
                          int total = 0;
                          for (auto& f : vec) {
                            total += f.get();
                          }
                        },
                        taskSet);

  taskSet.wait();
  EXPECT_TRUE(result.is_ready());
}
