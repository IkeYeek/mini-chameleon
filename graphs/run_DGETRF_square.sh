#!/bin/bash

TEST_PATH="../build/debug/testings/perf_dgetrf"
BLOCK_SIZES="1 64 128 256"
VARIANTS="vendor"
SIDE_SIZES="256 512 1024 2048 4096"
ITER=3
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

echo "Compiling project"

source ./helpers/project_compile.sh

if [[ "$1" == "-r" ]]; then
    FILE="$2"
    RESUMING=true
    echo "Resuming previous run in folder $FILE"
fi


echo "Creating GUIX shell"

mkdir -p "$FILE"


for size in $SIDE_SIZES; do
  for var in $VARIANTS; do
      if [ "$var" = "vendor" ]; then
        echo "Launching: BS=N/A, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
        OUTPUT_FILE="getrf-${var}-${size}x${size}.raw"

        if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
          OMP_NUM_THREADS=1 $TEST_PATH -v vendor -i $ITER -M $size -N $size > "$FILE/$OUTPUT_FILE"
        else
          echo "Skipping experiment (output file exists and complete)"
        fi
      else
        for blocksize in $BLOCK_SIZES; do
          echo "Launching: BS=$blocksize, Variant=$var, MxN=${size}x${size}, ITER=$ITER"
          OUTPUT_FILE="getrf-${var}-${size}x${size}-bs_${blocksize}.raw"
          if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
            BLOCKSIZE=$blocksize $TEST_PATH -v $var -i $ITER -M $size -N $size > "$FILE/$OUTPUT_FILE"
          else
            echo "Skipping experiment (output file exists and complete)"
          fi
        done
      fi
  done
done

echo "All experiments complete. Exiting GUIX for Visualisation"

cd "$FILE"
for file in *.raw; do
  echo "Processing text file: $file"
  python3 ../formatter_dgetrf.py -i "$file" -al -o "getrf_omp_square.out"
done
for file in *.out; do
  echo "Processing text file: $file"
   for script in $RFILES; do
    guix shell r r-ggplot2 r-dplyr r-tidyr -- Rscript "$script" "$file"
  done
done
