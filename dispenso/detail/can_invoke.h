/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>

#include <type_traits>
#include <utility>

namespace dispenso {
namespace detail {

template <class>
using void_t = void;
template <class Sig, class = void>
struct CanInvoke : std::false_type {};
template <class F, class... Args>
struct CanInvoke<F(Args...), void_t<decltype(std::declval<F>()(std::declval<Args>()...))>>
    : std::true_type {};

} // namespace detail
} // namespace dispenso
