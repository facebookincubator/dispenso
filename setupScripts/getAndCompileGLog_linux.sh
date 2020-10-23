#!/bin/bash

#create 3rd party directory  outside of dispenso directory
cd ../../
mkdir thirdparty
cd thirdparty

# Check out the library.
git clone https://github.com/google/glog.git
# Go to the library root directory and build
cd glog
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build
sudo cmake --build build --target install
