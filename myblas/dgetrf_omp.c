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

static int dgetrf_omp_block_size = 8;

int
dgetrf_omp( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int small_dim = M<N ? M :N;
    for (int k = 0; k < small_dim; k += dgetrf_omp_block_size) {
        int size = (small_dim - k < dgetrf_omp_block_size) ? (small_dim - k) : dgetrf_omp_block_size;

        dgetrf_seq(layout, size, size, &A[k + k * lda], lda);

        if (k + size < N) {
            cblas_dtrsm(layout,
                        CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        size, N - (k + size), 1.0,
                        &A[k + k * lda], lda,
                        &A[k + (k + size) * lda], lda);
        }

        if (k + size < M) {
            cblas_dtrsm(layout,
                        CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                        M - (k + size), size, 1.0,
                        &A[k + k * lda], lda,
                        &A[k + size + k * lda], lda);
        }

        if (k + size < M && k + size < N) {
            dgemm_omp(layout,
                      CblasNoTrans, CblasNoTrans,
                      M - (k + size), N - (k + size), size,
                      -1.0,
                      &A[k + size + k * lda], lda,
                      &A[k + (k + size) * lda], lda,
                      1.0,
                      &A[k + size + (k + size) * lda], lda);
        }
    }

    return ALGONUM_SUCCESS; /* Success */
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
    dgetrf_omp_block_size = myblas_getenv_value_int("BLOCKSIZE_OMP", 64);
}
