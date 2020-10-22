#!/bin/bash

#create 3rd party directory  outside of dispenso directory
cd ../../
mkdir thirdparty
cd thirdparty

wget https://github.com/google/googletest/archive/release-1.10.0.tar.gz

tar xf release-1.10.0.tar.gz
cd googletest-release-1.10.0
cmake .
make

sudo cp -a googletest/include/gtest /usr/include
sudo cp lib/*.a /usr/lib/
sudo cp -r googlemock/include/gmock /usr/include/
