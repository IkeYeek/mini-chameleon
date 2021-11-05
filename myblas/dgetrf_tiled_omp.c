/**
 *
 * @file dgetrf_tiled_omp.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP tile-based implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"

int
dgetrf_tiled_omp( CBLAS_LAYOUT layout,
                  int M, int N, int b, double **A )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_imin( MT, NT );
    int m, n, k;

    for( k=0; k<KT; k++) {
        int mk = k == (MT-1) ? M - k * b : b;
        int nk = k == (NT-1) ? N - k * b : b;
        int kk = my_imin( mk, nk );

        dgetrf_seq( LAPACK_COL_MAJOR, mk, nk,
                    A[ MT * k + k ], b );

        for( n=k+1; n<NT; n++) {
            int nn = n == (NT-1) ? N - n * b : b;

            cblas_dtrsm( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                         kk, nn,
                         1., A[ MT * k + k ], b,
                             A[ MT * n + k ], b );
        }
        for( m=k+1; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

            cblas_dtrsm( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                         mm, kk,
                         1., A[ MT * k + k ], b,
                             A[ MT * k + m ], b );

            for( n=k+1; n<NT; n++) {
                int nn = n == (NT-1) ? N - n * b : b;

                dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                           mm, nn, kk,
                           -1., A[ MT * k + m ], b,
                                A[ MT * n + k ], b,
                           1.,  A[ MT * n + m ], b );
            }
        }
    }

    return ALGONUM_SUCCESS; /* Success */
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_omp __attribute__ ((unused)) = dgetrf_tiled_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_omp;

/**
 * @brief Registration function
 */
void dgetrf_tiled_omp_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_omp_init( void )
{
    fct_dgetrf_tiled_omp.mpi    = 0;
    fct_dgetrf_tiled_omp.tiled  = 1;
    fct_dgetrf_tiled_omp.starpu = 0;
    fct_dgetrf_tiled_omp.name   = "omp-t";
    fct_dgetrf_tiled_omp.helper = "OpenMP tile-based implementation of the dgetrf";
    fct_dgetrf_tiled_omp.fctptr = dgetrf_tiled_omp;
    fct_dgetrf_tiled_omp.next   = NULL;

    register_fct( &fct_dgetrf_tiled_omp, ALGO_GETRF );
}
