# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# A minimal downstream project that consumes an *installed* dispenso the way
# docs/building.md tells users to: find_package(Dispenso) and the imported
# Dispenso::dispenso target, with no include or library paths of its own. CI
# builds and runs it against both a shared and a static install, which checks
# the exported target interface (include directories, link dependencies such
# as Threads, and the package version file) rather than the build tree.
#
#   cmake -S tests/install_consumer -B <dir> -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build <dir>
#   <dir>/install_consumer

cmake_minimum_required(VERSION 3.12)
project(DispensoInstallConsumer LANGUAGES CXX)

find_package(Dispenso REQUIRED)

add_executable(install_consumer main.cpp)
target_link_libraries(install_consumer PRIVATE Dispenso::dispenso)
target_compile_features(install_consumer PRIVATE cxx_std_14)
# Lets main.cpp check the installed headers against the package version file.
target_compile_definitions(install_consumer PRIVATE
  CONSUMER_PACKAGE_VERSION="${Dispenso_VERSION}")
