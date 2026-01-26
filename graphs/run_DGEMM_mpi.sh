#!/bin/bash
FILE=$(date '+%Y-%m-%d-%H:%M:%S')
mkdir -p "$FILE"
# 1. Nodes
# 2. Tasks Per Node
# 3. Threads
# 4. MPI Version
# 5. M (Matrix Height)
# 6. N (Matrix Width)
# 7. K (Matrix Depth)
# 8. B (Block Size)

# Note: I finaly made custom scripts in /helpers
# Just set nodes and task per nodes here and change the helper script

EXPERIMENTS=(
    #"1 1 8 summa 16384 16384 16384 128"
    #"1 2 8 summa 16384 16384 16384 128"
    #"1 3 8 summa 16384 16384 16384 128"
    #"1 4 8 summa 16384 16384 16384 128"   
    "4 4 8 summa 16384 16384 16384 128"   
)

TASKS_PER_CORE=1

for exp in "${EXPERIMENTS[@]}"; do
    read -r NODES TPN THREADS MPI_VER MAT_M MAT_N MAT_K BLOCK_SIZE <<< "$exp"
    
    TOTAL_TASKS=$(( NODES * TPN ))
    JOB_ID="${MPI_VER}_S${MAT_M}_N${NODES}_T${TPN}_th${THREADS}_B${BLOCK_SIZE}"
    OUT_FILE="$FILE/bench_${JOB_ID}.raw"

    sbatch \
      --nodes=$NODES \
      --exclude=miriel023,miriel025 \
      --ntasks=$TOTAL_TASKS \
      --ntasks-per-node=$TPN \
      --ntasks-per-core=1 \
      --constraint=miriel \
      --exclusive \
      --output="$OUT_FILE" \
      --error="$OUT_FILE" \
      --export=ALL,Users_Threads=$THREADS,Users_Mpi=$MPI_VER,Users_M=$MAT_M,Users_N=$MAT_N,Users_K=$MAT_K,Users_B=$BLOCK_SIZE \
      helpers/mpi_strong.slurm
done
