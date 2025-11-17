#!/bin/bash

TEST_PATH="../build/debug/testings/perf_dgemm"
BLOCK_SIZES="64"
NUM_THREADS="4 8 16"
VARIANTS="omp"
SEQ_VARIANT="goto"
SIDE_SIZES="256 320 384 512 640 832 1024 1280 1600 1984 2496 3136 3904 4864"
ITER=5
FILE=$(date '+%Y-%m-%d-%H:%M:%S')
RFILES="../r_scripts/template.R ../r_scripts/template_simple.R ../r_scripts/bar.R"
RESUMING=false

need_run() {
    if [ "$RESUMING" = false ]; then
        return 0
    fi
    if [ ! -f "$1" ]; then
        return 0
    fi
    local lines=$(wc -l < "$1")
    if [ "$lines" -eq "$2" ]; then
        return 1
    else
        return 0
    fi
}

current_dir=$(pwd)
if [[ $current_dir == *graphs ]]; then
  echo "Compiling project"
else
    echo "Not running from graphs/ folder (cd into graphs)."
    exit 1
fi

source ./helpers/project_compile.sh

if [[ "$1" == "-r" ]]; then
    FILE="$2"
    RESUMING=true
    echo "Resuming previous run in folder $FILE"
fi


echo "Creating GUIX shell"

mkdir -p "$FILE"


for size in $SIDE_SIZES; do
  echo "Launching: BS=$bs, Variant=$SEQ_VARIANT, MxNxK=${size}x${size}x${size}, ITER=$ITER, Threads=1"
  OUTPUT_FILE="gemm-${SEQ_VARIANT}-${size}x${size}x${size}-SEQ.raw"
          
  if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
    OMP_NUM_THREADS=1 SEQ_VER=$SEQ_VARIANT BLOCKSIZE=$bs $TEST_PATH -v seq -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
  else
    echo "Skipping experiment (output file exists and complete)"
  fi
  for var in $VARIANTS; do
      for num_threads in $NUM_THREADS; do
        for bs in $BLOCK_SIZES; do
          echo "Launching: BS=$bs, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER, Threads=$num_threads"
          OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}-${bs}-${num_threads}.raw"
          
          if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
            OMP_NUM_THREADS=$num_threads OMP_PLACES=cores BLOCKSIZE=$bs $TEST_PATH -v $var -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
          else
            echo "Skipping experiment (output file exists and complete)"
          fi
        done
      done
  done
done

echo "All experiments complete. Exiting GUIX for Visualisation"
EOF
cd "$FILE"
for file in *.raw; do
  echo "Processing text file: $file"
  python3 ../formatter.py -i "$file" -al -o "gemm_square.out"
done
for file in *.out; do
  echo "Processing text file: $file"
   for script in $RFILES; do
    guix shell r r-ggplot2 r-dplyr r-tidyr -- Rscript "$script" "$file"
  done
done
