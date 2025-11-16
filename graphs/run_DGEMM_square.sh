#!/bin/bash

TEST_PATH="../build/debug/testings/perf_dgemm"
BLOCK_SIZES="64"
VARIANTS="scalaire bloc avx2_bloc vendor"
SIDE_SIZES="256 320 384 512 640 832 1024 1280 1600 1984 2496 3136 3904 4864 6080 7616 9536"
ITER=5
FILE=$(date '+%Y-%m-%d-%H:%M:%S')
RFILES="../r_scripts/template.R ../r_scripts/template_simple.R"
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
    
    if [ "$var" = "scalaire" ]; then
      echo "Launching: BS=N/A, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
      OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}.raw"
      
      if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
          SEQ_VER=$var $TEST_PATH -v seq -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
      else
          echo "Skipping experiment (output file exists and complete)"
      fi
      

    else
      if [ "$var" = "vendor" ]; then
        for bs in $BLOCK_SIZES; do
          echo "Launching: BS=N/A, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
          OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}.raw"

          if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
            $TEST_PATH -v vendor -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
          else
            echo "Skipping experiment (output file exists and complete)"
          fi
        done
      else
        for bs in $BLOCK_SIZES; do
          echo "Launching: BS=$bs, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
          OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}-${bs}.raw"
          
          if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
            SEQ_VER=$var BLOCKSIZE=$bs $TEST_PATH -v seq -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
          else
            echo "Skipping experiment (output file exists and complete)"
          fi
        done
      fi
      
    fi
    
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
    guix shell r r-ggplot2 r-dplyr -- Rscript "$script" "$file"
  done
done
