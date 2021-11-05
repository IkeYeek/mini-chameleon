/**
 *
 * @file ddot_seq.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Sequential version of the dot product.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"

double ddot_seq( int N, const double *X, int incX,
                        const double *Y, int incY )
{
    return 0.;
}

/* To make sure we use the right prototype */
static ddot_fct_t valid_ddot_seq __attribute__ ((unused)) = ddot_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_ddot_seq;

/**
 * @brief Registration function
 */
void ddot_seq_init( void ) __attribute__( ( constructor ) );
void
ddot_seq_init( void )
{
    fct_ddot_seq.mpi    = 0;
    fct_ddot_seq.tiled  = 0;
    fct_ddot_seq.starpu = 0;
    fct_ddot_seq.name   = "seq";
    fct_ddot_seq.helper = "Sequential version of DDOT";
    fct_ddot_seq.fctptr = ddot_seq;
    fct_ddot_seq.next   = NULL;

    register_fct( &fct_ddot_seq, ALGO_DDOT );
}
