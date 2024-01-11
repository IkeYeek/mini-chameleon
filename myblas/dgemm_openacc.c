/**
 *
 * @file dgemm_openacc.c
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

int dgemm_openacc( CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                   int M, int N, int K, int b,
                   double alpha, const double *A,
                                 const double *B,
                   double beta,        double *C )
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
static dgemm_fct_t valid_dgemm_openacc __attribute__ ((unused)) = dgemm_openacc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_openacc;

/**
 * @brief Registration function
 */
void dgemm_openacc_init( void ) __attribute__( ( constructor ) );
void
dgemm_openacc_init( void )
{
    memset( fct_dgemm_openacc, 0, sizeof( fct_list_t ) );
    fct_dgemm_openacc.openacc = 1;
    fct_dgemm_openacc.tiled   = 0;
    fct_dgemm_openacc.name    = "openacc";
    fct_dgemm_openacc.helper  = "OpenACC implementation of the dgemm";
    fct_dgemm_openacc.fctptr  = dgemm_openacc;
    fct_dgemm_openacc.next    = NULL;

    register_fct( &fct_dgemm_openacc, ALGO_GEMM );
}
