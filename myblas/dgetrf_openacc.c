/**
 *
 * @file dgetrf_openacc.c
 *
 * @copyright 2019-2023 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP tile-based implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @date 2023-12-18
 *
 */
#include "myblas.h"

int
dgetrf_openacc( CBLAS_LAYOUT layout,
                int M, int N, int b, double *A )
{
    return ALGONUM_NOT_IMPLEMENTED;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_openacc __attribute__ ((unused)) = dgetrf_openacc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_openacc;

/**
 * @brief Registration function
 */
void dgetrf_openacc_init( void ) __attribute__( ( constructor ) );
void
dgetrf_openacc_init( void )
{
    memset( fct_dgemm_openacc, 0, sizeof( fct_list_t ) );
    fct_dgetrf_openacc.openacc = 1;
    fct_dgetrf_openacc.tiled   = 0;
    fct_dgetrf_openacc.name    = "openacc";
    fct_dgetrf_openacc.helper  = "OpenACC implementation of the dgetrf";
    fct_dgetrf_openacc.fctptr  = dgetrf_openacc;
    fct_dgetrf_openacc.next    = NULL;

    register_fct( &fct_dgetrf_openacc, ALGO_GETRF );
}
