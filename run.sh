#!/bin/bash

set -e
set -x 

rm -rf build
mkdir build 
pushd build

conan install .. --profile=clang
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .

popd
build/bin/fishin
