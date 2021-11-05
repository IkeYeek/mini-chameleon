/**
 *
 * @file dgetrf_omp.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include <assert.h>

int
dgetrf_omp( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;

    return ALGONUM_NOT_IMPLEMENTED; /* Success */
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_omp __attribute__ ((unused)) = dgetrf_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_omp;

/**
 * @brief Registration function
 */
void dgetrf_omp_init( void ) __attribute__( ( constructor ) );
void
dgetrf_omp_init( void )
{
    fct_dgetrf_omp.mpi    = 0;
    fct_dgetrf_omp.tiled  = 0;
    fct_dgetrf_omp.starpu = 0;
    fct_dgetrf_omp.name   = "omp";
    fct_dgetrf_omp.helper = "OpenMP implementation of the dgetrf";
    fct_dgetrf_omp.fctptr = dgetrf_omp;
    fct_dgetrf_omp.next   = NULL;

    register_fct( &fct_dgetrf_omp, ALGO_GETRF );
}
