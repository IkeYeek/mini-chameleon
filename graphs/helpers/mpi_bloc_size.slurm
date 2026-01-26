#!/bin/bash
#SBATCH -C miriel
#SBATCH --exclusive


guix environment --pure mini-chameleon bash gcc-toolchain -- bash --norc

for i in {1..3}; do

    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 4096 -N 4096 -K 4096 -b 512
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 8192 -N 8192 -K 8192 -b 512
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 512
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 512
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 65536 -N 65536 -K 65536 -b 512

    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 8192 -N 8192 -K 8192 -b 1024
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 4096 -N 4096 -K 4096 -b 1024
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 1024
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 1024
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 65536 -N 65536 -K 65536 -b 1024

    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 4096 -N 4096 -K 4096 -b 2048
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 8192 -N 8192 -K 8192 -b 2048
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 2048
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 2048
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 65536 -N 65536 -K 65536 -b 2048

    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 8192 -N 8192 -K 8192 -b 4096   
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 4096
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 4096
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 65536 -N 65536 -K 65536 -b 4096
    
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 32768 -N 32768 -K 32768 -b 8192    
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 16384 -N 16384 -K 16384 -b 8192
    OMP_NUM_THREADS=6 MPI_VER=summa mpirun --mca mtl psm2 --map-by numa /home/cisd-jonquir/mini-mpi/build/debug/testings/perf_dgemm -v mpi -M 65536 -N 65536 -K 65536 -b 8192
    

    
done
