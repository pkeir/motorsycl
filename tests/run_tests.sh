#!/bin/bash

CXXFLAGS="-std=c++20"
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

FILES="three-simple-queues.cpp usm_shortcuts.cpp containers.cpp maths.cpp host_accessor.cpp auto_lambda.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp simple_2d_nd_range.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# MotörSYCL (MotorSYCL) using NVIDIA nvc++ (-w turns off warnings)
if [[ -v MYNVCPP ]]; then
  TOKENSTRING="$MYNVCPP/bin/nvc++ -stdpar -acc -std=c++20 -w ${INC} %s"
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
  TOKENSTRING="$COMPUTECPP_DIR/bin/compute++ -O2 -mllvm -inline-threshold=1000 -intelspirmetadata -Wno-unused-command-line-argument -sycl-driver -sycl-target spir64 -no-serial-memop -I $COMPUTECPP_DIR/include -std=c++17 -DSYCL_LANGUAGE_VERSION=202001 -L $COMPUTECPP_DIR/lib -lComputeCpp %s"
  doit
fi

FILES="maths.cpp host_accessor.cpp auto_lambda.cpp item.cpp simple_2d_test.cpp three_kernels.cpp unused_buffer.cpp qw.cpp vectors.cpp multi_ptr.cpp single_task.cpp"

# triSYCL using Clang
if [[ -v TRISYCL_INCLUDE ]]; then
  TOKENSTRING="clang++ -std=c++17 -I $TRISYCL_INCLUDE -I $MDSPAN_INCLUDE %s -pthread"
  doit
fi

