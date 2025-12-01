/**
 *
 * @file ddot_vendor.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief vendoruential version of the dot product.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"

double ddot_vendor( int N, const double *X, int incX,
                        const double *Y, int incY )
{
    cblas_ddot(N, X, incX, Y, incY);
    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static ddot_fct_t valid_ddot_vendor __attribute__ ((unused)) = ddot_vendor;

/* Declare the variable that will store the information about this version */
fct_list_t fct_ddot_vendor;

/**
 * @brief Registration function
 */
void ddot_vendor_init( void ) __attribute__( ( constructor ) );
void
ddot_vendor_init( void )
{
    fct_ddot_vendor.mpi    = 0;
    fct_ddot_vendor.tiled  = 0;
    fct_ddot_vendor.starpu = 0;
    fct_ddot_vendor.name   = "vendor";
    fct_ddot_vendor.helper = "vendor version of DDOT";
    fct_ddot_vendor.fctptr = ddot_vendor;
    fct_ddot_vendor.next   = NULL;

    register_fct( &fct_ddot_vendor, ALGO_DDOT );
}
