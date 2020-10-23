/**
 *
 * @file perf_ddot.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary to assess the performance of a DDOT implementation
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
    option_t options;
    int i, check = 0;

    parse_opts( argc, argv, &options, ALGO_DDOT );

    if ( options.fct == NULL ) {
        fprintf( stderr, "Need to define a version to test\n" );
        print_usage( argv[0], ALGO_DDOT );
        exit(1);
    }
    else {
        printf( "Test: %s\n", options.fct->helper );
    }

    for( i=0; i<options.iter; i++ ) {
        testone_ddot( options.fct->fctptr, options.N, check );
    }

    return EXIT_SUCCESS;
}

