/**
 *
 * @file warm_ddot.c
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
#include <stdlib.h>
#include <stdio.h>
#include "algonum.h"

int main( int argc, char **argv )
{
    option_t *options = &global_options;

    algonum_init( argc, argv, options, ALGO_DDOT );

    if ( options->fct->mpi || options->fct->tiled || options->fct->starpu ) {
        fprintf( stderr, "this benchmark is meant to bench only the sequential versions of your code\n" );
        algonum_exit( options, ALGO_DDOT );
        return EXIT_FAILURE;
    }

    testwarm_ddot( options->fct->fctptr, options->N );

    algonum_exit( options, ALGO_DDOT );

    return EXIT_SUCCESS;
}
