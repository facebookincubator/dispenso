#!/bin/bash

usage='An argument specifing package manager is needed; eg. (apt-get, dnf)'

if [ "$#" -ne 1 ]; then
    echo "$usage"

else
    #create 3rd party directory  outside of dispenso directory
    cd ../../
    mkdir thirdparty
    cd thirdparty

    # Check out the library.
    git clone https://github.com/facebook/folly.git
    # Go to the library root directory and build
    cd folly

    if [[ $1 != "apt-get" ]]; then
        sudo $1 install \
         gcc \
         gcc-c++ \
         automake \
         boost-devel \
         libtool \
         lz4-devel \
         libevent-devel-2.1.8-7.fc31.i686 \
         lzma-sdk-devel-4.6.5-24.fc31.x86_64 \
         snappy-devel \
         zlib-devel \
         glog-devel \
         gflags-devel \
         scons \
         double-conversion-devel \
         openssl-devel \
         libevent-devel
    else
        sudo $1 install \
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
    fi

    #build fmt from source
    git clone https://github.com/fmtlib/fmt.git
    cd fmt

    mkdir _build
    cd _build
    cmake ..

    echo "building fmt at $PWD"
    make -j$(nproc)
    sudo make install

    #go back to folly dir
    cd ../../

    #build folly
    mkdir _build
    cd _build
    cmake ..
    echo "building folly at $PWD"
    make -j $(nproc)
    sudo echo "installing folly at $PWD"
    sudo make install # with either sudo or DESTDIR as necessary
fi

