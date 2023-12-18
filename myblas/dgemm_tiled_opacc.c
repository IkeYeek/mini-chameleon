/**
 *
 * @file dgemm_tiled_openacc.c
 *
 * @copyright 2019-2023 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the OpenMP version
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @date 2023-12-18
 *
 */
#include "myblas.h"

int dgemm_tiled_openacc( CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                         int M, int N, int K, int b,
                         double alpha, const double **A,
                                       const double **B,
                         double beta,        double **C )
{
#if !defined(ENABLE_OPACC)
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
static dgemm_tiled_fct_t valid_dgemm_tiled_openacc __attribute__ ((unused)) = dgemm_tiled_openacc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_openacc;

/**
 * @brief Registration function
 */
void dgemm_tiled_openacc_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_openacc_init( void )
{
    fct_dgemm_tiled_openacc.openacc = 1;
    fct_dgemm_tiled_openacc.tiled   = 1;
    fct_dgemm_tiled_openacc.starpu  = 0;
    fct_dgemm_tiled_openacc.name    = "openacc";
    fct_dgemm_tiled_openacc.helper  = "OpenACC tiled implementation of the dgemm";
    fct_dgemm_tiled_openacc.fctptr  = dgemm_tiled_openacc;
    fct_dgemm_tiled_openacc.next    = NULL;

    register_fct( &fct_dgemm_tiled_openacc, ALGO_GEMM );
}
