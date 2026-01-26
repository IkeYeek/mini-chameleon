#!/bin/bash
#SBATCH -C miriel
#SBATCH --exclusive


guix environment --pure mini-chameleon bash gcc-toolchain -- bash -c "
    OMP_NUM_THREADS=$Users_Threads \
    MPI_VER=$Users_Mpi \
    mpirun --mca mtl psm2 --map-by numa \
    /home/cisd-jonquir/mini-chameleon/build/debug/testings/perf_dgemm \
    -v mpi \
    -M $Users_M \
    -N $Users_N \
    -K $Users_K \
    -b $Users_B
"