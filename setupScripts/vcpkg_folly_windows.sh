#!/bin/bash
cd ../../thirdparty/vcpkg
./vcpkg install glog:x64-windows
./vcpkg double-conversion:x64-windows
./vcpkg double-conversion:x64-windows-static
./vcpkg install libevent:x64-windows-static
./vcpkg install folly:x64-windows-static
./vcpkg install folly:x64-windows
cd ../../dispenso/setupScripts/