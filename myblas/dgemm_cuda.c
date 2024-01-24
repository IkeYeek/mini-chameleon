/**
 *
 * @file dgemm_cuda.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the CUDA version of the dgemm.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"
#include <string.h>

int dgemm_cuda  ( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )
{
    if ( transA == CblasNoTrans ) {
        if ( transB == CblasNoTrans ) {
            return ALGONUM_NOT_IMPLEMENTED;
        }
        else {
            return ALGONUM_NOT_IMPLEMENTED;
        }
    }
    else {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_cuda __attribute__ ((unused)) = dgemm_cuda;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_cuda;

/**
 * @brief Registration function
 */
void dgemm_cuda_init( void ) __attribute__( ( constructor ) );
void
dgemm_cuda_init( void )
{
    memset( &fct_dgemm_cuda, 0, sizeof( fct_list_t ) );
    fct_dgemm_cuda.cuda    = 1;
    fct_dgemm_cuda.tiled   = 0;
    fct_dgemm_cuda.name    = "cuda";
    fct_dgemm_cuda.helper  = "CUDA implementation of the dgemm";
    fct_dgemm_cuda.fctptr  = dgemm_cuda;
    fct_dgemm_cuda.next    = NULL;

    register_fct( &fct_dgemm_cuda, ALGO_GEMM );
}
