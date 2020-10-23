/**
 *
 * @file common.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template for the initial sequential GEMM function
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "algonum.h"

int dgemm_seq( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )
{
    int m, n, k;

    if ( transA == CblasNoTrans ) {
        if ( transB == CblasNoTrans ) {
	    for( m=0; m<M; m++ ) {
		for( n=0; n<N; n++ ) {
		    C[ ldc * n + m ] = beta * C[ ldc * n + m ];
		    for( k=0; k<K; k++ ) {
                        C[ ldc * n + m ] += alpha * A[ lda * k + m ] * B[ ldb * n + k ];
                    }
                }
            }
        }
        else {
	    for( m=0; m<M; m++ ) {
		for( n=0; n<N; n++ ) {
		    C[ ldc * n + m ] = beta * C[ ldc * n + m ];
		    for( k=0; k<K; k++ ) {
                        C[ ldc * n + m ] += alpha * A[ lda * k + m ] * B[ ldb * k + n ];
                    }
                }
            }
        }
    }
    else {
        if ( transB == CblasNoTrans ) {
	    for( m=0; m<M; m++ ) {
		for( n=0; n<N; n++ ) {
		    C[ ldc * n + m ] = beta * C[ ldc * n + m ];
		    for( k=0; k<K; k++ ) {
			C[ ldc * n + m ] += alpha * A[ lda * m + k ] * B[ ldb * n + k ];
                    }
                }
            }
        }
        else {
	    for( m=0; m<M; m++ ) {
		for( n=0; n<N; n++ ) {
		    C[ ldc * n + m ] = beta * C[ ldc * n + m ];
		    for( k=0; k<K; k++ ) {
                        C[ ldc * n + m ] += alpha * A[ lda * m + k ] * B[ ldb * k + n ];
                    }
                }
            }
        }
    }

    return 0;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_seq __attribute__ ((unused)) = dgemm_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_seq;

/**
 * @brief Registration function
 */
void dgemm_seq_init( void ) __attribute__( ( constructor ) );
void
dgemm_seq_init( void )
{
    fct_dgemm_seq.tiled  = 0;
    fct_dgemm_seq.starpu = 0;
    fct_dgemm_seq.name   = "seq";
    fct_dgemm_seq.helper = "Sequential version of DGEMM";
    fct_dgemm_seq.fctptr = dgemm_seq;
    fct_dgemm_seq.next   = NULL;

    register_fct( &fct_dgemm_seq, ALGO_GEMM );
}
