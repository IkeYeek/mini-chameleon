/**
 *
 * @file dgemm_omp.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the OpenMP version
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"

int dgemm_omp( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )
{
    int m, n, k;

    if ( transA == CblasNoTrans ) {
        if ( transB == CblasNoTrans ) {
	    return ALGONUM_NOT_IMPLEMENTED /* Not implemented */;
        }
        else {
	    return ALGONUM_NOT_IMPLEMENTED /* Not implemented */;
        }
    }
    else {
        if ( transB == CblasNoTrans ) {
	    return ALGONUM_NOT_IMPLEMENTED /* Not implemented */;
        }
        else {
	    return ALGONUM_NOT_IMPLEMENTED /* Not implemented */;
        }
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_omp __attribute__ ((unused)) = dgemm_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_omp;

/**
 * @brief Registration function
 */
void dgemm_omp_init( void ) __attribute__( ( constructor ) );
void
dgemm_omp_init( void )
{
    fct_dgemm_omp.mpi    = 0;
    fct_dgemm_omp.tiled  = 0;
    fct_dgemm_omp.starpu = 0;
    fct_dgemm_omp.name   = "omp";
    fct_dgemm_omp.helper = "Scalar OpenMP implementation of the dgemm";
    fct_dgemm_omp.fctptr = dgemm_omp;
    fct_dgemm_omp.next   = NULL;

    register_fct( &fct_dgemm_omp, ALGO_GEMM );
}
