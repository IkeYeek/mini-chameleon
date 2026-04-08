#!/bin/bash

TEST_PATH="../build/debug/testings/perf_dgemm"
BLOCK_SIZES="64"
VARIANTS="scalaire bloc avx2_bloc vendor"
M_SIZES="512 1024"
N_SIZES="512 1024"
K_SIZES="512 1024"
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

echo "Compiling project"

source ./helpers/project_compile.sh

if [[ "$1" == "-r" ]]; then
    FILE="$2"
    RESUMING=true
    echo "Resuming previous run in folder $FILE"
fi


echo "Creating GUIX shell"

mkdir -p "$FILE"


for m in $M_SIZES; do
  for n in $N_SIZES; do
    for k in $K_SIZES; do
      for var in $VARIANTS; do
        
        if [ "$var" = "scalaire" ]; then
          echo "Launching: BS=N/A, Variant=$var, MxNxK=${m}x${n}x${k}, ITER=$ITER"
          OUTPUT_FILE="gemm-${var}-${m}x${n}x${k}.raw"
          
          if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
              SEQ_VER=$var $TEST_PATH -v seq -i $ITER -M $m -N $n -K $k > "$FILE/$OUTPUT_FILE"
          else
              echo "Skipping experiment (output file exists and complete)"
          fi
          

        else
          if [ "$var" = "vendor" ]; then
            for bs in $BLOCK_SIZES; do
              echo "Launching: BS=N/A, Variant=$var, MxNxK=${m}x${n}x${k}, ITER=$ITER"
              OUTPUT_FILE="gemm-${var}-${m}x${n}x${k}.raw"

              if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
                $TEST_PATH -v vendor -i $ITER -M $m -N $n -K $k > "$FILE/$OUTPUT_FILE"
              else
                echo "Skipping experiment (output file exists and complete)"
              fi
            done
          else
            for bs in $BLOCK_SIZES; do
              echo "Launching: BS=$bs, Variant=$var, MxNxK=${m}x${n}x${k}, ITER=$ITER"
              OUTPUT_FILE="gemm-${var}-${m}x${n}x${k}-${bs}.raw"
              
              if need_run "$FILE/$OUTPUT_FILE" $((ITER + 1)); then
                SEQ_VER=$var BLOCKSIZE=$bs $TEST_PATH -v seq -i $ITER -M $m -N $n -K $k > "$FILE/$OUTPUT_FILE"
              else
                echo "Skipping experiment (output file exists and complete)"
              fi
            done
          fi
          
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
