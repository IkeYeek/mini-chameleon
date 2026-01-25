/**
 *
 * @file dgetrf_seq.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Basic sequential implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include <assert.h>

static int dgetrf_seq_block_size = -1;

int
dgetrf_scalaire( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;

    for( k=0; k<K; k++ ) {
        for( m=k+1; m<M; m++ ) {
            A[ lda * k + m ] = A[ lda * k + m ] / A[ lda * k + k ];
            for( n=k+1; n<N; n++ ) {
                A[ lda * n + m ] = A[ lda * n + m ] - A[ lda * k + m ] * A[ lda * n + k ];
            }
        }
    }

    return ALGONUM_SUCCESS; /* Success */
}

int
dgetrf_block( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int small_dim = M<N ? M :N;
    for (int k = 0; k < small_dim; k += dgetrf_seq_block_size) {
        int size = (small_dim - k < dgetrf_seq_block_size) ? (small_dim - k) : dgetrf_seq_block_size;

        dgetrf_scalaire(layout, size, size, &A[k + k * lda], lda);

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
            dgemm_seq(layout,
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


int dgetrf_seq( CBLAS_LAYOUT layout, int M, int N, double *A, int lda ) {
  if (dgetrf_seq_block_size > 1) {
    dgetrf_block(layout, M,  N, A, lda );
  } else {
    dgetrf_scalaire( layout, M, N, A, lda );
  }

  return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_seq __attribute__ ((unused)) = dgetrf_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_seq;

/**
 * @brief Registration function
 */
void dgetrf_seq_init( void ) __attribute__( ( constructor ) );
void
dgetrf_seq_init( void )
{
    fct_dgetrf_seq.mpi    = 0;
    fct_dgetrf_seq.tiled  = 0;
    fct_dgetrf_seq.starpu = 0;
    fct_dgetrf_seq.name   = "seq";
    fct_dgetrf_seq.helper = "Basic sequential implementation of the dgetrf";
    fct_dgetrf_seq.fctptr = dgetrf_seq;
    fct_dgetrf_seq.next   = NULL;

    register_fct( &fct_dgetrf_seq, ALGO_GETRF );
    dgetrf_seq_block_size = myblas_getenv_value_int("BLOCKSIZE", 32);
}
