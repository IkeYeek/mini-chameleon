#!/usr/bin/env sh

module load build/cmake/3.15.3
module load compiler/gcc/9.3.0
module load mpi/openmpi/4.0.3-mlx
module load compiler/cuda/10.2
module load linalg/mkl/2020_update4

module load trace/fxt/0.3.9
module load trace/eztrace/1.1-8

# Choose the right one
module load runtime/starpu/1.3.8/mpi
#module load runtime/starpu/1.3.8/mpi-fxt
#module load runtime/starpu/1.3.8/mpi-cuda
#module load runtime/starpu/1.3.8/mpi-cuda-fxt

