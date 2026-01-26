#!/bin/bash
#SBATCH -C miriel
#SBATCH --exclusive


guix environment --pure mini-chameleon bash gcc-toolchain -- bash --norc

OMP_NUM_THREADS=6 MPI_VER=summa-vendor mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 131072 -N 131072 -K 131072 -b 2048
OMP_NUM_THREADS=6 MPI_VER=summa-vendor mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 131072 -N 131072 -K 131072 -b 4096
