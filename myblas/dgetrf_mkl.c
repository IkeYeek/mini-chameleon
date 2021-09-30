/**
 *
 * @file dgetrf_mkl.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Register the MKL function for references
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"

int dgetrf_mkl( CBLAS_LAYOUT layout,
                int m, int n, double *A, int lda )
{
    int minMN = ( m < n ) ? m : n;
    int *ipiv = malloc( minMN * sizeof(int) );
    int rc;

    rc = LAPACKE_dgetrf_work( LAPACK_COL_MAJOR, m, n, A, lda, ipiv );
    assert( rc == 0 );

#if !defined(NDEBUG)
    {
        int i;
        for(i=0; i<minMN; i++) {
            assert( ipiv[i] == i+1 );
        }
    }
#endif
    free( ipiv );

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_mkl __attribute__ ((unused)) = dgetrf_mkl;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_mkl;

/**
 * @brief Registration function
 */
void dgetrf_mkl_init( void ) __attribute__( ( constructor ) );
void
dgetrf_mkl_init( void )
{
    fct_dgetrf_mkl.mpi    = 0;
    fct_dgetrf_mkl.tiled  = 0;
    fct_dgetrf_mkl.starpu = 0;
    fct_dgetrf_mkl.name   = "mkl";
    fct_dgetrf_mkl.helper = "MKL implementation of the dgetrf";
    fct_dgetrf_mkl.fctptr = dgetrf_mkl;
    fct_dgetrf_mkl.next   = NULL;

    register_fct( &fct_dgetrf_mkl, ALGO_GETRF );
}
