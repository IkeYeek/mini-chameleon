/**
 *
 * @file test_dgemm_tiled.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm variants in tile format.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum_int.h"

/**
 *  @brief Function to test one single case of tiled dgemm.
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
 * @param[in] dplrnt
 *          The function pointer to a dplrnt operation on tiled format to initialize the matrices.
 *
 * @param[in] dgemm
 *          The function pointer to a dgemm operation on tiled format to test.
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
testone_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                     dgemm_tiled_fct_t  dgemm,
                     CBLAS_TRANSPOSE transA,
                     CBLAS_TRANSPOSE transB,
                     int M, int N, int K, int b,
                     int check )
{
    int     Am, An, Bm, Bn;
    int     rc = 0;
    double **Atile, **Btile, **Ctile;
    int     lda, ldb, ldc;
    double  alpha, beta;
    int     seedA = 4972;
    int     seedB = 330;
    int     seedC = 93047;
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
    lda = max( lda, my_iceil( Am, b ) * b );

    if ( transB == CblasNoTrans ) {
        Bm = K;
        Bn = N;
    }
    else {
        Bm = N;
        Bn = K;
    }
    ldb = max( Bm, 1 );
    ldb = max( ldb, my_iceil( Bm, b ) * b );
    ldc = max( M, 1 );
    ldc = max( ldc, my_iceil( M,  b ) * b );

    /* Initialize alpha and beta */
    CORE_dplrnt( 0, 1, 1, &alpha, lda, 1, 0, 0, 342 );
    CORE_dplrnt( 0, 1, 1, &beta,  ldb, 1, 0, 0, 651 );

    /* Allocate A, B, and C */
    Atile = lapack2tile( Am, An, b, NULL, lda );
    Btile = lapack2tile( Bm, Bn, b, NULL, ldb );
    Ctile = lapack2tile( M,  N,  b, NULL, ldc );

    /* Fill the matrices with random values */
    dplrnt( 0., Am, An, b, Atile, seedA );
    dplrnt( 0., Bm, Bn, b, Btile, seedB );
    dplrnt( 0., M,  N,  b, Ctile, seedC );

    /* Calculate the product */
    perf( &start );
    rc = dgemm( CblasColMajor, transA, transB, M, N, K, b,
                alpha, (const double **)Atile, (const double **)Btile, beta, Ctile );
    perf( &stop );

    if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
        if ( global_options.mpirank == 0 ) {
            printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: " ALGONUM_COLOR_ORANGE "Not supported or not implemented\n" ALGONUM_COLOR_RESET,
                    (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                    (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                    M, N, K );
        }

        tileFree( Am, An, b, Atile );
        tileFree( Bm, Bn, b, Btile );
        tileFree( M,  N,  b, Ctile );

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
        double *A, *B, *C, *Cinit;

        /* Create the matrices for the test */
        if ( global_options.mpirank == 0 ) {
            A = malloc( lda * An * sizeof(double) );
            B = malloc( ldb * Bn * sizeof(double) );
            C = malloc( ldc * N  * sizeof(double) );
        }

        /* Fill the matrices with the same random values */
        tile2lapack( Am, An, b, (const double **)Atile, A, lda );
        tile2lapack( Bm, Bn, b, (const double **)Btile, B, ldb );
        tile2lapack( M,  N,  b, (const double **)Ctile, C, ldc );

        /* Create the original C to compare with */
        if ( global_options.mpirank == 0 ) {
            Cinit = malloc( ldc * N  * sizeof(double) );
            CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, seedC );

            rc = check_dgemm( transA, transB, M, N, K,
                              alpha, A, lda, B, ldb,
                              beta, Cinit, C, ldc );

            if ( rc == ALGONUM_SUCCESS ) {
                printf( "tA=%s tB=%s M= %4d N= %4d K= %4d alpha= %e beta= %e: %le GFlop/s: " ALGONUM_COLOR_GREEN "SUCCESS\n" ALGONUM_COLOR_RESET,
                        (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                        (transB == CblasNoTrans) ? "NoTrans" : "Trans", M, N, K,
                        alpha, +beta, gflops);
            }
            else {
                printf( "tA=%s tB=%s M= %4d N= %4d K= %4d alpha= %e beta= %e: %le GFlop/s: " ALGONUM_COLOR_RED "FAIL\n" ALGONUM_COLOR_RESET,
                        (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                        (transB == CblasNoTrans) ? "NoTrans" : "Trans", M, N, K,
                        alpha, +beta, gflops );
            }
            free( A );
            free( B );
            free( C );
            free( Cinit );
        }
    }
    else {
        if ( global_options.mpirank == 0 ) {
            printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: %le GFlop/s\n",
                    (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                    (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                    M, N, K, gflops );
        }
    }

    tileFree( Am, An, b, Atile );
    tileFree( Bm, Bn, b, Btile );
    tileFree( M,  N,  b, Ctile );

    return rc;
}

/**
 * @brief Function to test a series of tests on a gemm function.
 *
 * @param[in] dplrnt
 *          The function pointer to a dplrnt operation on tiled format to initialize the matrices.
 *
 * @param[in] dgemm
 *          The function pointer to a gemm operation on tiled format to test.
 *
 * @retval 0, on success
 * @retval The number of failures otherwise.
 *
 */
int
testall_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                     dgemm_tiled_fct_t  dgemm )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_K[] = { 0, 1, 8, 24, 47 };
    int all_b[] = { 1, 3, 16 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_K = sizeof( all_K ) / sizeof( int );
    int nb_b = sizeof( all_b ) / sizeof( int );

    int rc;
    int im, in, ik, ib, m, n, k, b;
    int nbfailed = 0;
    int nbnotimplemented = 0;
    int nbpassed = 0;
    int nbtests = 4 * nb_M * nb_N * nb_K * nb_b;
    CBLAS_TRANSPOSE tA, tB;

    for ( tA = CblasNoTrans; tA <= CblasTrans; tA ++ ) {
        for ( tB = CblasNoTrans; tB <= CblasTrans; tB ++ ) {
            for( im = 0; im < nb_M; im ++ ) {
                m = all_M[im];
                for( in = 0; in < nb_N; in ++ ) {
                    n = all_N[in];
                    for( ik = 0; ik < nb_K; ik ++ ) {
                        k = all_K[ik];
                        for( ib = 0; ib < nb_b; ib ++ ) {
                            b = all_b[ib];

                            rc = testone_dgemm_tiled( dplrnt, dgemm, tA, tB, m,
                                                      n, k, b, 1 );
                            if ( rc == ALGONUM_FAIL ) {
                                nbfailed++;
                            }
                            else if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
                                nbnotimplemented++;
                            }
                            nbpassed++;
                            if ( global_options.mpirank == 0 ) {
                                printf( "Test %4d / %4d\n", nbpassed, nbtests );
                            }
                        }
                    }
                }
            }
        }
    }

    if ( global_options.mpirank == 0 ) {
        if ( nbnotimplemented > 0 ) {
            printf( ALGONUM_COLOR_ORANGE
                    "Warning: %4d tests out of %d could not be assessed due to "
                    "non implemented variants\n" ALGONUM_COLOR_RESET,
                    nbnotimplemented, nbtests);
        }
        else {
            printf( "All tested variants were implemented\n" );
        }
        if ( nbfailed > 0 ) {
            printf( ALGONUM_COLOR_RED
                    "%4d tests failed out of %d\n" ALGONUM_COLOR_RESET,
                    nbfailed, nbtests - nbnotimplemented );
        }
        else {
            if ( nbtests != nbnotimplemented ) {
                printf( ALGONUM_COLOR_GREEN "Congratulations: all %4d tests "
                        "succeeded\n" ALGONUM_COLOR_RESET,
                        nbtests - nbnotimplemented);
            }
        }
    }
    return nbfailed;
}
