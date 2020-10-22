/**
 *
 * @file perf_dgemm.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary to assess the performance of a GEMM implementation
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include "algonum.h"

int main( int argc, char **argv )
{
    dplrnt_tiled_fct_t tested_tiled_dplrnt = dplrnt_tiled;
    option_t options;
    int i, check = 0;

    parse_opts( argc, argv, &options, ALGO_GEMM );

    if ( options.fct == NULL ) {
        fprintf( stderr, "Need to define a version to test\n" );
        print_usage( argv[0], ALGO_GEMM );
        exit(1);
    }
    else {
        printf( "Test: %s\n", options.fct->helper );
    }

#if defined(ENABLE_STARPU)
    if ( options.fct->starpu ) {
        my_starpu_init();
        tested_tiled_dplrnt = dplrnt_tiled_starpu;
    }
#endif

    for( i=0; i<options.iter; i++ ) {
        if ( options.fct->tiled ) {
            testone_dgemm_tiled( tested_tiled_dplrnt,
                                 options.fct->fctptr,
                                 options.transA, options.transB,
                                 options.M, options.N, options.K,
                                 options.b, check );
        }
        else {
            testone_dgemm( options.fct->fctptr,
                           options.transA, options.transB,
                           options.M, options.N, options.K, check );
        }
    }

#if defined(ENABLE_STARPU)
    if ( options.fct->starpu ) {
        my_starpu_exit();
    }
#endif

    return EXIT_SUCCESS;
}

