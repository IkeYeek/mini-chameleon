/**
 *
 * @file dgetrf_cuda.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the CUDA version of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include <string.h>

int
dgetrf_cuda( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    return ALGONUM_NOT_IMPLEMENTED;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_cuda __attribute__ ((unused)) = dgetrf_cuda;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_cuda;

/**
 * @brief Registration function
 */
void dgetrf_cuda_init( void ) __attribute__( ( constructor ) );
void
dgetrf_cuda_init( void )
{
    memset( &fct_dgetrf_cuda, 0, sizeof( fct_list_t ) );
    fct_dgetrf_cuda.cuda   = 1;
    fct_dgetrf_cuda.tiled  = 0;
    fct_dgetrf_cuda.name   = "cuda";
    fct_dgetrf_cuda.helper = "CUDA implementation of the dgetrf";
    fct_dgetrf_cuda.fctptr = dgetrf_cuda;
    fct_dgetrf_cuda.next   = NULL;

    register_fct( &fct_dgetrf_cuda, ALGO_GETRF );
}
