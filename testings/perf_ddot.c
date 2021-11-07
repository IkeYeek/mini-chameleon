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
#include <stdlib.h>
#include <stdio.h>
#include "algonum.h"

int main( int argc, char **argv )
{
    option_t *options = &global_options;
    int i;

    algonum_init( argc, argv, options, ALGO_DDOT );

    for( i=0; i<options->iter; i++ ) {
        testone_ddot( options->fct->fctptr, options->N, options->check );
    }

    algonum_exit( options, ALGO_DDOT );

    return EXIT_SUCCESS;
}
