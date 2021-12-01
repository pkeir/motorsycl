#!/bin/bash

CXXFLAGS="-std=c++20"
GXX=$MYGCC/bin/g++
CLANG=$MYCLANG/bin/clang++
INC="-I $MOTORSYCL_INCLUDE"
COUNT=1

function doit()
{
  echo -e "**********\n"$FILES
  for f in ${FILES[@]}; do
#    echo "$COUNT Processing $f ..." && COUNT=$((COUNT+1))
    printf -v CMDSTRING "$TOKENSTRING" "$f"
    echo -e "\n"$CMDSTRING
    rm -rf ./a.out
    eval $CMDSTRING
    ./a.out
  done
}

FILES="usm_shortcuts.cpp containers.cpp maths.cpp host_accessor.cpp auto_lambda.cpp item.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp simple_2d_nd_range.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# Used by both GXX and CLANG below
OLD_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MYGCC/lib64:$LD_LIBRARY_PATH

# Motörsycl using Clang (using recent GCC's libstdc++ for sycl::bit_cast)
TOKENSTRING="${CLANG} -std=c++20 -Wno-deprecated-declarations -isystem $MYGCC/include/c++/* -isystem $MYGCC/include/c++/*/x86_64-pc-linux-gnu ${INC} %s -L$MYGCC/lib64 -L$MYGCC/lib/gcc/x86_64-pc-linux-gnu/* -ltbb"
doit

# Motörsycl using GCC (A recent GCC build - say 2021)
# The -DTBB_SUPPRESS_DEPRECATED_MESSAGES tells the system TBB's _deprecated_header_message_guard.h not to define __TBB_show_deprecated_header_message and produce warning messages, over libstdc++'s inclusion of deprecated header tbb/task.h
TOKENSTRING="${GXX} -std=c++20 -DTBB_SUPPRESS_DEPRECATED_MESSAGES -Wno-deprecated-declarations ${INC} %s -ltbb"
doit

# Restore; just in case
export LD_LIBRARY_PATH=$OLD_PATH

FILES="usm_shortcuts.cpp containers.cpp maths.cpp host_accessor.cpp auto_lambda.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp simple_2d_nd_range.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# MotörSYCL (MotorSYCL) using NVIDIA nvc++
if [[ -v MYNVCPP ]]; then
  TOKENSTRING="$MYNVCPP/bin/nvc++ -stdpar -std=c++20 ${INC} %s"
  doit
fi
FILES="usm_shortcuts.cpp containers.cpp maths.cpp host_accessor.cpp auto_lambda.cpp item.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp simple_2d_nd_range.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# Intel's SYCL (DPCPP) -fsycl-unnamed-lambda is also useful (pre-2020-default)
if [[ -v MYDPCPP ]]; then
if [[ -v CUDA_10_2 ]]; then
  TOKENSTRING="$MYDPCPP/bin/clang++ --cuda-path=$CUDA_10_2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda %s"
else
  TOKENSTRING="$MYDPCPP/bin/clang++ -fsycl %s"
fi # CUDA_10_2
doit
fi # MYDPCPP

FILES="maths.cpp auto_lambda.cpp simple_2d_test.cpp three_kernels.cpp qw.cpp simple_2d_nd_range.cpp vectors.cpp single_task.cpp"

# Codeplay's ComputeCpp
if [[ -v COMPUTECPP_DIR ]]; then
  TOKENSTRING="$COMPUTECPP_DIR/bin/compute++ -O2 -mllvm -intelspirmetadata -inline-threshold=1000 -Wno-unused-command-line-argument -sycl-driver -sycl-target spir64 -no-serial-memop -I $COMPUTECPP_DIR/include -std=c++17 -DSYCL_LANGUAGE_VERSION=202001 -L $COMPUTECPP_DIR/lib -lComputeCpp %s"
  doit
fi

FILES="maths.cpp host_accessor.cpp auto_lambda.cpp item.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# triSYCL using Clang
if [[ -v TRISYCL_INCLUDE ]]; then
  TOKENSTRING="clang++ -std=c++17 -I $TRISYCL_INCLUDE %s -pthread"
  doit
fi

