/**
 *
 * @file dgemm_tiled_omp.c
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

int dgemm_tiled_omp( CBLAS_LAYOUT layout,
                     CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int M, int N, int K, int b,
                     double alpha, const double **A,
                                   const double **B,
                     double beta,        double **C )
{
    double lbeta;
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );
    int m, n, k, mm, nn, kk;

    if ( transA == CblasNoTrans ) {
        if ( transB == CblasNoTrans ) {
            for( m=0; m<MT; m++ ) {
                mm = m == (MT-1) ? M - m * b : b;

                for( n=0; n<NT; n++ ) {
                    nn = n == (NT-1) ? N - n * b : b;

                    for( k=0; k<KT; k++ ) {
                        kk = k == (KT-1) ? K - k * b : b;

                        lbeta = (k == 0) ? beta : 1.;
                        dgemm_seq( CblasColMajor, transA, transB,
                                   mm, nn, kk,
                                   alpha, A[ MT * k + m ], b,
                                          B[ KT * n + k ], b,
                                   lbeta, C[ MT * n + m ], b );
                    }
                }
            }
        }
        else {
            for( m=0; m<MT; m++ ) {
                mm = m == (MT-1) ? M - m * b : b;

                for( n=0; n<NT; n++ ) {
                    nn = n == (NT-1) ? N - n * b : b;

                    for( k=0; k<KT; k++ ) {
                        kk = k == (KT-1) ? K - k * b : b;

                        lbeta = (k == 0) ? beta : 1.;
                        dgemm_seq( CblasColMajor, transA, transB,
                                   mm, nn, kk,
                                   alpha, A[ MT * k + m ], b,
                                          B[ NT * k + n ], b,
                                   lbeta, C[ MT * n + m ], b );
                    }
                }
            }
        }
    }
    else {
        if ( transB == CblasNoTrans ) {
            for( m=0; m<MT; m++ ) {
                mm = m == (MT-1) ? M - m * b : b;

                for( n=0; n<NT; n++ ) {
                    nn = n == (NT-1) ? N - n * b : b;

                    for( k=0; k<KT; k++ ) {
                        kk = k == (KT-1) ? K - k * b : b;

                        lbeta = (k == 0) ? beta : 1.;
                        dgemm_seq( CblasColMajor, transA, transB,
                                   mm, nn, kk,
                                   alpha, A[ KT * m + k ], b,
                                          B[ KT * n + k ], b,
                                   lbeta, C[ MT * n + m ], b );
                    }
                }
            }
        }
        else {
            for( m=0; m<MT; m++ ) {
                mm = m == (MT-1) ? M - m * b : b;

                for( n=0; n<NT; n++ ) {
                    nn = n == (NT-1) ? N - n * b : b;

                    for( k=0; k<KT; k++ ) {
                        kk = k == (KT-1) ? K - k * b : b;

                        lbeta = (k == 0) ? beta : 1.;
                        dgemm_seq( CblasColMajor, transA, transB,
                                   mm, nn, kk,
                                   alpha, A[ KT * m + k ], b,
                                          B[ NT * k + n ], b,
                                   lbeta, C[ MT * n + m ], b );
                    }
                }
            }
        }
    }

    return ALGONUM_SUCCESS;
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
    fct_dgemm_tiled_omp.mpi    = 0;
    fct_dgemm_tiled_omp.tiled  = 1;
    fct_dgemm_tiled_omp.starpu = 0;
    fct_dgemm_tiled_omp.name   = "omp-t";
    fct_dgemm_tiled_omp.helper = "Tiled OpenMP implementation of the dgemm";
    fct_dgemm_tiled_omp.fctptr = dgemm_tiled_omp;
    fct_dgemm_tiled_omp.next   = NULL;

    register_fct( &fct_dgemm_tiled_omp, ALGO_GEMM );
}
