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
