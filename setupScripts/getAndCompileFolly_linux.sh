#!/bin/bash

#create 3rd party directory  outside of dispenso directory
cd ../../
mkdir thirdparty
cd thirdparty

# Check out the library.
git clone https://github.com/facebook/folly.git
# Go to the library root directory and build
cd folly

sudo apt-get install \
    g++ \
    cmake \
    libboost-all-dev \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libiberty-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    libssl-dev \
    pkg-config \
    libunwind-dev

#build fmt from source
git clone https://github.com/fmtlib/fmt.git && cd fmt

mkdir _build && cd _build
cmake ..

make -j$(nproc)
sudo make install

#go back to folly dir
cd ../../

#build folly
mkdir _build && cd _build
cmake ..
make -j $(nproc)
sudo make install # with either sudo or DESTDIR as necessary

