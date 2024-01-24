/**
 *
 * @file dgemm_cublas.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Register the cublas function for references
 *
 * This version of the DGEMM should not be modified. It is provided to
 * you in order to have a reference point for the sequential
 * algorithm.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"
#include <string.h>

int dgemm_cublas  ( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )

{
    cublasStatus_t rc;
    rc = cublasDgemm( my_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
    // TODO: FIXME: use get_cublas_trans instead of bypassing transA and transB
    // rc = cublasDgemm( my_cublas_handle, get_cublas_trans( transA ), get_cublas_trans( transB ),
                      M, N, K,
                      &alpha, A, lda, B, ldb, &beta, C, ldc );
    cudaDeviceSynchronize();
    return (rc == CUBLAS_STATUS_SUCCESS) ? ALGONUM_SUCCESS : ALGONUM_FAIL;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_cublas __attribute__ ((unused)) = dgemm_cublas;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_cublas;

/**
 * @brief Registration function
 */
void dgemm_cublas_init( void ) __attribute__( ( constructor ) );
void
dgemm_cublas_init( void )
{
    memset( &fct_dgemm_cublas, 0, sizeof( fct_list_t ) );
    fct_dgemm_cublas.cuda   = 1;
    fct_dgemm_cublas.tiled  = 0;
    fct_dgemm_cublas.name   = "cublas";
    fct_dgemm_cublas.helper = "cuBLAS implementation of the dgemm";
    fct_dgemm_cublas.fctptr = dgemm_cublas;
    fct_dgemm_cublas.next   = NULL;

    register_fct( &fct_dgemm_cublas, ALGO_GEMM );
}
