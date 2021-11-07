/**
 *
 * @file check_dgetrf.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm_tiled variants.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum_int.h"

int
check_dgetrf( int M, int N,
              double *LU, double *A, int lda )
{
    int info_solution;
    double Anorm, Rnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    double *L, *U;
    int ldl, ldu;
    int minMN = ( M < N ) ? M : N;

    ldl = max( 1, lda );
    ldu = max( 1, minMN );

    /* Calculates the dimensions according to the transposition */
    L = LU;
    U = malloc( minMN * N * sizeof(double) );

    /* Set before copying to get the non unit diagonal */
    LAPACKE_dlaset_work( LAPACK_COL_MAJOR, 'A', minMN, N, 0., 1.,  U, ldu );
    LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'U', minMN, N, LU, lda, U, ldu );

    /* Copy L before setting to force the diagonal to 1. */
    LAPACKE_dlaset_work( LAPACK_COL_MAJOR, 'U', M, minMN, 0., 1., L, ldl );

    Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', M, N, A, lda, NULL );

    /* Makes the multiplication with the core function */
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, minMN,
                 -1, L, ldl, U, ldu, 1., A, lda );

    /* Calculates the norm with the core function's result */
    Rnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, A, lda, NULL );

    result = (Anorm * minMN * eps);
    if ( result > 0. ) {
        result = Rnorm / result;
    }

    /* Verifies if the result is inside a threshold */
    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        /* fprintf(stderr, "M= %4d, N= %4d, Anorm= %le, Rnorm= %le, result= %le\n", */
        /*         M, N, Anorm, Rnorm, result ); */
        info_solution = ALGONUM_FAIL;
    }
    else {
        info_solution = ALGONUM_SUCCESS;
    }

    free( U );

    return info_solution;
}
