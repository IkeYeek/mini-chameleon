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

#define BLOCK_SIZE 32

int
dgetrf_seq( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
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
    for (int k=0; k<N; k+=BLOCK_SIZE){
        dgetrf_seq(layout, BLOCK_SIZE, BLOCK_SIZE, A+k, lda);
        for (int i=k+BLOCK_SIZE; i<M; i+=BLOCK_SIZE){
            //TRSM
            //cblas_dtrsm(layout, side, uplo, trans, diag, M, N, )
            cblas_dtrsm(layout, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M - i, BLOCK_SIZE, 1.0, A + k, lda, A + i, lda);
        }
        for (int j=k+BLOCK_SIZE; j<N; j+=BLOCK_SIZE){
            //TRSM
            cblas_dtrsm(layout, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, BLOCK_SIZE, N - j, 1.0, A + k, lda, A + j * lda, lda);
        }
        dgemm_seq(layout, CblasNoTrans, CblasNoTrans, M-(k+BLOCK_SIZE), N-(k+BLOCK_SIZE), k, -1.0, A+k+BLOCK_SIZE, lda, A+lda*(k+BLOCK_SIZE), lda, 1.0, A+lda*(k+BLOCK_SIZE)+k+BLOCK_SIZE, lda);
    }

    return ALGONUM_SUCCESS; /* Success */
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
    fct_dgetrf_seq.fctptr = dgetrf_block;
    fct_dgetrf_seq.next   = NULL;

    register_fct( &fct_dgetrf_seq, ALGO_GETRF );
}
