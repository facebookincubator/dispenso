/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispenso/once_function.h>

#include <deque>

#include <gtest/gtest.h>

constexpr size_t kExtraSmall = 8;
constexpr size_t kSmall = 24;
constexpr size_t kMedium = 120;
constexpr size_t kLarge = 248;
constexpr size_t kExtraLarge = 10000;

using dispenso::OnceFunction;

TEST(OnceFunction, Empty) {
  OnceFunction f([]() {});
  f();
}

TEST(OnceFunction, MoveConstructor) {
  OnceFunction f([]() {});
  OnceFunction g(std::move(f));
  g();
}

TEST(OnceFunction, MoveOperator) {
  OnceFunction f([]() {});
  OnceFunction g;
  g = std::move(f);
  g();
}

template <size_t kSize>
void testSize() {
  constexpr size_t kNumElts = kSize - sizeof(int*);
  struct Foo {
    void operator()() {
      int s = 0;
      for (uint8_t b : buf) {
        s += b;
      }
      *sum = s;
    }
    uint8_t buf[kNumElts];
    int* sum;
  } foo;
  for (size_t i = 0; i < kNumElts; ++i) {
    foo.buf[i] = static_cast<uint8_t>(i);
  }
  int answer;
  foo.sum = &answer;
  OnceFunction f(foo);
  OnceFunction g(foo);
  g();
  f();
  int expected = 0;
  for (size_t i = 0; i < kNumElts; ++i) {
    expected += static_cast<int>(i & 255);
  }
  EXPECT_EQ(answer, expected);
}

template <>
void testSize<8>() {
  struct Foo {
    void operator()() {
      int s = 0;
      *sum = s;
    }
    int* sum;
  } foo;
  int answer;
  foo.sum = &answer;
  OnceFunction f(foo);
  OnceFunction g(foo);
  g();
  f();
  int expected = 0;
  EXPECT_EQ(answer, expected);
}

TEST(OnceFunction, ExtraSmall) {
  testSize<kExtraSmall>();
}

TEST(OnceFunction, Small) {
  testSize<kSmall>();
}

TEST(OnceFunction, Medium) {
  testSize<kMedium>();
}

TEST(OnceFunction, Large) {
  testSize<kLarge>();
}

TEST(OnceFunction, ExtraLarge) {
  testSize<kExtraLarge>();
}

TEST(OnceFunction, MoveWithResult) {
  int result = 5;
  OnceFunction f([&result]() { result = 17; });
  EXPECT_EQ(result, 5);
  OnceFunction g(std::move(f));
  EXPECT_EQ(result, 5);
  g();
  EXPECT_EQ(result, 17);
}

template <size_t kNumElts>
void ensureDestructor() {
  int value = 0;
  struct FooWithDestructor {
    void operator()() {
      ++*value;
    }
    ~FooWithDestructor() {
      ++*value;
    }
    uint8_t buf[kNumElts];
    int* value;
  } foo;

  foo.value = &value;

  OnceFunction f(foo);
  f();
  EXPECT_EQ(value, 2);
}

TEST(OnceFunction, EnsureDestructionExtraSmall) {
  ensureDestructor<kExtraSmall>();
}

TEST(OnceFunction, EnsureDestructionSmall) {
  ensureDestructor<kSmall>();
}

TEST(OnceFunction, EnsureDestructionMedium) {
  ensureDestructor<kMedium>();
}

TEST(OnceFunction, EnsureDestructionLarge) {
  ensureDestructor<kLarge>();
}

TEST(OnceFunction, EnsureDestructionExtraLarge) {
  ensureDestructor<kExtraLarge>();
}

template <size_t alignment>
struct EnsureAlign {
  void operator()() {
    uintptr_t bloc = reinterpret_cast<uintptr_t>(&b);
    EXPECT_EQ(0, bloc & (alignment - 1)) << "broken for alignment: " << alignment;
  }

  alignas(alignment) char b = 0;
};

TEST(OnceFunction, EnsureAlignment1) {
  EnsureAlign<1> e;
  OnceFunction f(e);
  f();
}

TEST(OnceFunction, EnsureAlignment2) {
  EnsureAlign<2> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment4) {
  EnsureAlign<4> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment8) {
  EnsureAlign<8> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment16) {
  EnsureAlign<16> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment32) {
  EnsureAlign<32> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment64) {
  EnsureAlign<64> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment128) {
  EnsureAlign<128> e;
  OnceFunction f(e);
  f();
}
TEST(OnceFunction, EnsureAlignment256) {
  EnsureAlign<256> e;
  OnceFunction f(e);
  f();
}
