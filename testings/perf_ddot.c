/**
 *
 * @file perf_ddot.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary to assess the performance of a DDOT implementation
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "algonum.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  option_t *options = &global_options;
  int i, rep;

  algonum_init(argc, argv, options, ALGO_DDOT);
  FILE *output = fopen("./eval_ddot.txt", "w+");

  for (rep = 0; rep < 10; rep++)
    for (i = 0; i < options->iter; i++) {
      testone_ddot_real(options->fct->fctptr, 2 << i, options->check, output);
    }
  fclose(output);

  algonum_exit(options, ALGO_DDOT);

  return EXIT_SUCCESS;
}
