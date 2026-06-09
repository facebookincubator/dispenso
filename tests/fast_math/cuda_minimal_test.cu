/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Compile-only smoke test: verifies that all fast_math scalar functions can be
// compiled from a .cu file (nvcc) for both host and device.  No kernels are
// launched — the test passes if the translation unit compiles and links.

#include <dispenso/fast_math/fast_math.h>

namespace dfm = dispenso::fast_math;

// Force instantiation of every public scalar function so nvcc checks
// host/device correctness of the entire template set.
// volatile prevents the compiler from optimising calls away entirely.
volatile float gResult = 0.0f;

void instantiateAll() {
  float x = 1.5f;
  float y = 0.5f;
  float r = 0.0f;

  r += dfm::sin(x);
  r += dfm::cos(x);
  r += dfm::tan(x);
  r += dfm::asin(y);
  r += dfm::acos(y);
  r += dfm::atan(x);
  r += dfm::atan2(y, x);
  r += dfm::sinpi(x);
  r += dfm::cospi(x);
  r += dfm::exp(x);
  r += dfm::exp2(x);
  r += dfm::exp10(x);
  r += dfm::expm1(y);
  r += dfm::log(x);
  r += dfm::log2(x);
  r += dfm::log10(x);
  r += dfm::log1p(y);
  r += dfm::sqrt(x);
  r += dfm::cbrt(x);
  r += dfm::hypot(x, y);
  r += dfm::pow(x, y);
  r += dfm::tanh(x);
  r += dfm::erf(y);
  r += dfm::rsqrt(x);
  r += dfm::rsqrt_approx(x);
  r += dfm::rcp(x);
  r += dfm::rcp_approx(x);

  gResult = r;
}

int main() {
  instantiateAll();
  return 0;
}
