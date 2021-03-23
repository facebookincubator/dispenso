#!/bin/bash

#create 3rd party directory  outside of dispenso directory
cd ../../
mkdir thirdparty
cd thirdparty

# Check out the library.
git clone https://github.com/google/googletest.git
