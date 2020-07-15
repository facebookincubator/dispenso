#!/bin/bash
cd ../../thirdparty/vcpkg
./vcpkg install folly:x64-windows-static
xcopy "installed\\x64-windows-static"  "installed\\x64-windows" //Y //E
cd ../../dispenso/setupScripts/