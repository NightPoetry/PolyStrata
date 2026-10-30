#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH=/opt/rocm/bin:$PATH
cmake -S . -B build-906 -G "Unix Makefiles" -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DSTRATA_ENABLE_TESTS=OFF -DSTRATA_ENABLE_BENCH=OFF \
  -DSTRATA_GGML_DIR=/home/x99/poly-salvage/third_party/llama.cpp \
  > /tmp/poly-build.log 2>&1
make -C build-906 -j32 strata >> /tmp/poly-build.log 2>&1
echo "BUILD_OK" >> /tmp/poly-build.log
ls -lh build-906/strata >> /tmp/poly-build.log 2>&1
