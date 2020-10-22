/**
 *
 * @file check_dgemm.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary checking the numerical validity of the dgemm function
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

    if ( options.fct->tiled ) {
        testall_dgemm_tiled( tested_tiled_dplrnt,
                              options.fct->fctptr );
    }
    else {
        testall_dgemm( options.fct->fctptr );
    }

#if defined(ENABLE_STARPU)
    if ( options.fct->starpu ) {
        my_starpu_exit();
    }
#endif

    return EXIT_SUCCESS;
}
