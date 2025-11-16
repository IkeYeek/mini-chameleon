#!/bin/bash

export TEST_PATH="../build/debug/testings/perf_dgemm"
export BLOCK_SIZES="64"
export VARIANTS="scalaire bloc avx2_bloc"
export SIDE_SIZES="512 1024 2048"
export ITER="5"
export FILE=$(date '+%Y-%m-%d-%H:%M:%S')
RFILES="../r_scripts/template.R ../r_scripts/template_simple.R"

# If you want to edit:
# Comment guix shell command
# To enable bash lang syntax highlight

echo "Compiling project"

source ./helpers/project_compile.sh

echo "Creating GUIX shell"

guix shell --pure -D mini-chameleon bash gcc-toolchain -- \
env -i \
  FILE="$FILE" \
  TEST_PATH="$TEST_PATH" \
  BLOCK_SIZES="$BLOCK_SIZES" \
  VARIANTS="$VARIANTS" \
  SIDE_SIZES="$SIDE_SIZES" \
  ITER="$ITER" \
  bash <<'EOF'


mkdir -p "$FILE"


for size in $SIDE_SIZES; do
  for var in $VARIANTS; do
    
    if [ "$var" = "scalaire" ]; then
      echo "Launching: BS=N/A, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
      OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}"
      
      SEQ_VER=$var $TEST_PATH -v seq -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"

    else
      for bs in $BLOCK_SIZES; do
        echo "Launching: BS=$bs, Variant=$var, MxNxK=${size}x${size}x${size}, ITER=$ITER"
        OUTPUT_FILE="gemm-${var}-${size}x${size}x${size}-${bs}"
        
        SEQ_VER=$var BLOCKSIZE=$bs $TEST_PATH -v seq -i $ITER -M $size -N $size -K $size > "$FILE/$OUTPUT_FILE"
      done
    fi
    
  done
done

echo "All experiments complete. Exiting GUIX for Visualisation"
EOF
cd "$FILE"
for file in *; do
  echo "Processing text file: $file"
  python3 ../formatter.py -i "$file" -al -o "gemm_square.out"
done
for file in *.out; do
  echo "Processing text file: $file"
   for script in $RFILES; do
    guix shell r r-ggplot2 r-dplyr -- Rscript "$script" "$file"
  done
done




