/**
 *
 * @file check_dgemm.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary checking the numerical validity of the dgemm function
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include "algonum.h"

int main( int argc, char **argv )
{
    dplrnt_tiled_fct_t tested_tiled_dplrnt = dplrnt_tiled;
    option_t *options = &global_options;

    algonum_init( argc, argv, options, ALGO_GEMM );

#if defined(ENABLE_STARPU)
    if ( options->fct->starpu ) {
        tested_tiled_dplrnt = dplrnt_tiled_starpu;
    }
#endif

    if ( options->fct->tiled ) {
        testall_dgemm_tiled( tested_tiled_dplrnt,
                              options->fct->fctptr );
    }
    else {
        testall_dgemm( options->fct->fctptr );
    }

    algonum_exit( options, ALGO_GEMM );
    return EXIT_SUCCESS;
}
