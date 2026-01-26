#!/bin/bash
#SBATCH -C miriel
#SBATCH --exclusive


guix environment --pure mini-chameleon bash gcc-toolchain -- bash --norc

OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-chameleon/build/debug/testings/perf_dgemm -v mpi -M 4096 -N 4096 -K 4096 -b 2048
OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-chameleon/build/debug/testings/perf_dgemm -v mpi -M 8192 -N 8192 -K 8192 -b 2048
OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-chameleon/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 2048
OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-chameleon/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 2048
