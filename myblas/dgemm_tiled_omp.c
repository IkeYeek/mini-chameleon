/**
 *
 * @file dgemm_tiled_omp.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the OpenMP version
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "algonum.h"

int dgemm_tiled_omp( CBLAS_LAYOUT layout,
                    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                    int M, int N, int K, int b,
                    double alpha, const double **A,
                                  const double **B,
                    double beta,        double **C )
{
    return 1 /* Not implemented */;
}

/* To make sure we use the right prototype */
static dgemm_tiled_fct_t valid_dgemm_tiled_omp __attribute__ ((unused)) = dgemm_tiled_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_omp;

/**
 * @brief Registration function
 */
void dgemm_tiled_omp_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_omp_init( void )
{
    fct_dgemm_tiled_omp.tiled  = 1;
    fct_dgemm_tiled_omp.starpu = 0;
    fct_dgemm_tiled_omp.name   = "omp-t";
    fct_dgemm_tiled_omp.helper = "Tiled OpenMP implementation of the dgemm";
    fct_dgemm_tiled_omp.fctptr = dgemm_tiled_omp;
    fct_dgemm_tiled_omp.next   = NULL;

    register_fct( &fct_dgemm_tiled_omp, ALGO_GEMM );
}
