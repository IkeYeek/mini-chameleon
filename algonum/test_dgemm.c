/**
 *
 * @file test_dgemm.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm variants on lapack format.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum_int.h"

/**
 *  @brief Function to test one single case of dgemm.
 *
 *  Perform and check eventually the result of one of the following operation
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m-by-k matrix, op( B ) a k-by-n matrix and C an m-by-n matrix.
 *
 * @param[in] gemm
 *          The function pointer to a gemm operation on lapack format to test.
 *
 * @param[in] transA
 *          - CblasNoTrans:   A is not transposed,
 *          - CblasTrans:     A is transposed,
 *
 * @param[in] transB
 *          - CblasNoTrans:   B is not transposed,
 *          - CblasTrans:     B is transposed,
 *
 * @param[in] M
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] N
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] K
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] check
 *          Boolean to enable/disable the numerical check.
 *
 * @retval 0, on success
 * @retval 1, on failure
 *
 */
int
testone_dgemm( dgemm_fct_t     dgemm,
               CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB,
               int M, int N, int K, int check )
{
    int     Am, An, Bm, Bn;
    int     rc = 0;
    double *A, *B, *C;
    int     lda, ldb, ldc;
    double  alpha, beta;
    int     seedA = random();
    int     seedB = random();
    int     seedC = random();
    perf_t  start, stop;

    double gflops;
    double flops = flops_dgemm( M, N, K );

    /* Compute the dimension of A and B */
    if ( transA == CblasNoTrans ) {
        Am = M;
        An = K;
    }
    else {
        Am = K;
        An = M;
    }
    lda = max( Am, 1 );

    if ( transB == CblasNoTrans ) {
        Bm = K;
        Bn = N;
    }
    else {
        Bm = N;
        Bn = K;
    }
    ldb = max( Bm, 1 );
    ldc = max( M, 1 );

    /* Initialize alpha and beta */
    CORE_dplrnt( 0, 1, 1, &alpha, lda, 1, 0, 0, random() );
    CORE_dplrnt( 0, 1, 1, &beta,  ldb, 1, 0, 0, random() );

    /* Allocate A, B, and C */
    A = malloc( lda * An * sizeof(double) );
    B = malloc( ldb * Bn * sizeof(double) );
    C = malloc( ldc * N  * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( 0, Am, An, A, lda, Am, 0, 0, seedA );
    CORE_dplrnt( 0, Bm, Bn, B, ldb, An, 0, 0, seedB );
    CORE_dplrnt( 0, M,  N,  C, ldc, M,  0, 0, seedC );

    /* Calculate the product */
    perf( &start );
    rc = dgemm( CblasColMajor, transA, transB, M, N, K,
                alpha, A, lda, B, ldb,
                beta, C, ldc );
    perf( &stop );

    if ( rc ) {
        fprintf( stderr,
                 "tA=%s tB=%s M= %4d N= %4d K= %4d: Not Supported or not implemented\n",
                 (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                 (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                 M, N, K );
        return rc;
    }

    perf_diff( &start, &stop );
    if ( flops > 0. ) {
        gflops = perf_gflops( &stop, flops );
    }
    else {
        gflops = 0.;
    }

    /* Check the solution */
    if ( check ) {
        double *Cinit = malloc( ldc * N  * sizeof(double) );
        CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, seedC );

        rc = check_dgemm( transA, transB, M, N, K,
                          alpha, A, lda, B, ldb,
                          beta, Cinit, C, ldc );

        if ( rc ) {
            fprintf( stderr,
                     "tA=%s tB=%s M= %4d N= %4d K= %4d alpha= %e, beta= %e: FAILED\n",
                     (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                     (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                     M, N, K, alpha, beta );
        }
        free( Cinit );
    }
    else {
        printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: %le GFlop/s\n",
                (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                M, N, K, gflops );
    }

    free( A );
    free( B );
    free( C );

    return rc;
}

/**
 * @brief Function to test a series of tests on a gemm function.
 *
 * @param[in] dgemm
 *          The function pointer to a gemm operation on lapack format to test.
 *
 * @retval 0, on success
 * @retval The number of failures otherwise.
 *
 */
int
testall_dgemm( dgemm_fct_t dgemm )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_K[] = { 0, 1, 8, 24, 47 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_K = sizeof( all_K ) / sizeof( int );

    int im, in, ik, m, n, k;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = 4 * nb_M * nb_N * nb_K;
    CBLAS_TRANSPOSE tA, tB;

    for ( tA = CblasNoTrans; tA <= CblasTrans; tA ++ ) {
        for ( tB = CblasNoTrans; tB <= CblasTrans; tB ++ ) {
            for( im = 0; im < nb_M; im ++ ) {
                m = all_M[im];
                for( in = 0; in < nb_N; in ++ ) {
                    n = all_N[in];
                    for( ik = 0; ik < nb_K; ik ++ ) {
                        k = all_K[ik];

                        nbfailed += testone_dgemm( dgemm, tA, tB, m, n, k, 1 );
                        nbpassed++;
                        fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
                    }
                }
            }
        }
    }

    if ( nbfailed > 0 ) {
        fprintf( stdout, "\n %4d tests failed out of %d\n",
                 nbfailed, nbtests );
    }
    else {
        fprintf( stdout, "\n Congratulations all %4d tests succeeded\n",
                 nbtests );
    }
    return nbfailed;
}
