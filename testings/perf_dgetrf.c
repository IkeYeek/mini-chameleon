/**
 *
 * @file perf_dgetrf.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary to assess the performance of a GETRF implementation
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
    int i;

    algonum_init( argc, argv, options, ALGO_GETRF );

#if defined(ENABLE_STARPU)
    if ( options->fct->starpu ) {
        tested_tiled_dplrnt = dplrnt_tiled_starpu;
    }
#endif

    for( i=0; i<options->iter; i++ ) {
        if ( options->fct->tiled ) {
            testone_dgetrf_tiled( tested_tiled_dplrnt,
                                  options->fct->fctptr,
                                  options->M, options->N,
                                  options->b, options->check );
        }
#if defined(ENABLE_CUDA)
        else if ( options->fct->cuda ) {
            testone_dgetrf_cuda(
                options->fct->fctptr,
                options->M, options->N, options->check );
        }
#endif
        else {
            testone_dgetrf(
                options->fct->fctptr,
                options->M, options->N, options->check );
        }
    }

    algonum_exit( options, ALGO_GETRF );

    return EXIT_SUCCESS;
}
