#!/bin/bash

#create 3rd party directory  outside of dispenso directory
cd ../../
mkdir thirdparty
cd thirdparty

# Check out the library.
git clone https://github.com/wjakob/tbb.git
# Go to the library root directory and build
cd tbb
cmake .

#manually open vs project and build