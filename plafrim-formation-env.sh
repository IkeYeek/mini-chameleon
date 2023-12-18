#!/usr/bin/env sh

# Outils de base
module load build/cmake/3.21.3
module load tools/git/2.30.0

# Compilateur
module load compiler/gcc/12.2.0
module load mpi/openmpi/4.0.3-mlx
module load compiler/cuda/11.7

# BLAS / LAPACK avec Intel MKL
module load linalg/mkl/2020_update4

# Gestion du binding des threads et traces (dépendences de StarPU)
module load hardware/hwloc/2.5.0
module load trace/fxt/0.3.13
module load trace/eztrace/1.1-9

# Module StarPU pour la fin du projet (Choose the right one)
#module load runtime/starpu/1.3.8/mpi
module load runtime/starpu/1.3.8/mpi-fxt
#module load runtime/starpu/1.3.8/mpi-cuda
#module load runtime/starpu/1.3.8/mpi-cuda-fxt
