REM Copyright (c) Meta Platforms, Inc. and affiliates.
REM
REM This source code is licensed under the MIT license found in the
REM LICENSE file in the root directory of this source tree.

@echo off
REM Windows launcher for dispenso benchmarks
REM Sets up Visual Studio build environment and runs run_benchmarks.py
REM Output: benchmark_results.json with system info and benchmark results
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
set PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin
python -u scripts\run_benchmarks.py --build --high-priority --min-time 1.5 -o benchmark_results.json --cmake-args="-G" --cmake-args="Visual Studio 18 2026" --cmake-args="-A" --cmake-args="x64" --cmake-args="-DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows" --cmake-args="-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON" %*
