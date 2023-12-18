/**
 *
 * @file dgemm_tiled_cuda.c
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

int dgemm_tiled_cuda( CBLAS_LAYOUT layout,
                      CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                      int M, int N, int K, int b,
                      double alpha, const double **A,
                                    const double **B,
                      double beta,        double **C )
{
#if !defined(ENABLE_CUDA)
    return ALGONUM_NOT_IMPLEMENTED;
#else
    const double *Aptr, *Bptr;
    double        wsA[b*b], wsB[b*b];
    double        lbeta;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );

    int m, n, k, mm, nn, kk;
    int ownerA, ownerB, ownerC;

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
#endif
}

/* To make sure we use the right prototype */
static dgemm_tiled_fct_t valid_dgemm_tiled_cuda __attribute__ ((unused)) = dgemm_tiled_cuda;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_cuda;

/**
 * @brief Registration function
 */
void dgemm_tiled_cuda_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_cuda_init( void )
{
    fct_dgemm_tiled_cuda.cuda   = 1;
    fct_dgemm_tiled_cuda.tiled  = 1;
    fct_dgemm_tiled_cuda.starpu = 0;
    fct_dgemm_tiled_cuda.name   = "cuda";
    fct_dgemm_tiled_cuda.helper = "CUDA tiled implementation of the dgemm";
    fct_dgemm_tiled_cuda.fctptr = dgemm_tiled_cuda;
    fct_dgemm_tiled_cuda.next   = NULL;

    register_fct( &fct_dgemm_tiled_cuda, ALGO_GEMM );
}
