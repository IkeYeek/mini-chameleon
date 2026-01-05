#!/bin/bash

export TEST_PATH="../build/debug/testings/perf_dgemm"
export BLOCK_SIZES="16 32 64"
export VARIANTS="goto omp omp-t"
export M_SIZES="256 512 1024 2048 4096 8192"
export N_SIZES="256 512 1024 2048 4096 8192"
export K_SIZES="256 512 1024 2048 4096 8192"
export ITER="5"
export FILE=$(date '+%Y-%m-%d-%H:%M:%S')
RFILES="../r_scripts/template.R ../r_scripts/template_simple.R"

# If you want to edit:
# Comment guix shell command
# To enable bash lang syntax highlight

echo "Creating GUIX shell"

guix shell --pure -D mini-chameleon bash gcc-toolchain -- \
env -i \
  FILE="$FILE" \
  TEST_PATH="$TEST_PATH" \
  BLOCK_SIZES="$BLOCK_SIZES" \
 VARIANTS="$VARIANTS" \
  M_SIZES="$M_SIZES" \
  N_SIZES="$N_SIZES" \
  K_SIZES="$K_SIZES" \
  ITER="$ITER" \
  bash <<'EOF'


mkdir -p "$FILE"


for m in $M_SIZES; do
  for n in $N_SIZES; do
    for k in $K_SIZES; do
      for var in $VARIANTS; do
        
        if [ "$var" = "goto" ]; then
          echo "Launching: BS=N/A, Variant=$var, MxNxK=${m}x${n}x${k}, ITER=$ITER"
          OUTPUT_FILE="gemm-${var}-${m}x${n}x${k}"
          
          SEQ_VER=$var $TEST_PATH -v seq -i $ITER -M $m -N $n -K $k > "$FILE/$OUTPUT_FILE"

        else
          for bs in $BLOCK_SIZES; do
            echo "Launching: BS=$bs, Variant=$var, MxNxK=${m}x${n}x${k}, ITER=$ITER"
            OUTPUT_FILE="gemm-${var}-${m}x${n}x${k}-${bs}"
            
            SEQ_VER=goto $TEST_PATH -v $var -i $ITER -M $m -N $n -K $k -b $bs> "$FILE/$OUTPUT_FILE"
          done
        fi
        
      done
    done
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




