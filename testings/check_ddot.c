/**
 *
 * @file check_ddot.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Binary checking the numerical validity of the ddot function
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

    testall_ddot( options->fct->fctptr );

    algonum_exit( options, ALGO_DDOT );

    return EXIT_SUCCESS;
}
