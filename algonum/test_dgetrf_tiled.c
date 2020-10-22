/**
 *
 * @file test_dgetrf_tiled.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgetrf variants in tile format.
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "algonum_int.h"

int
testone_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                      dgetrf_tiled_fct_t dgetrf,
                      int M, int N, int b, int check )
{
    int      rc = 0;
    double **Atile;
    int      lda;
    int      seedA = random();
    perf_t   start, stop;
    int minMN = ( M < N ) ? M : N;

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );

    /* Fill the matrices with random values */
    Atile = lapack2tile( M, N, b, NULL, lda );
    dplrnt( minMN, M, N, b, Atile, seedA );

    /* Calculate the product */
    perf( &start );
    rc = dgetrf( CblasColMajor, M, N, b, Atile );
    perf( &stop );

    if ( rc ) {
        fprintf( stderr,
                 "M= %4d N= %4d: Not Supported or not implemented\n",
                 M, N );
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
        double *A     = malloc( lda * N * sizeof(double) );
        double *Ainit = malloc( lda * N * sizeof(double) );
        CORE_dplrnt( minMN, M, N, Ainit, lda, M, 0, 0, seedA );

        tile2lapack( M, N, b, (const double **)Atile, A, lda );

        rc = check_dgetrf( M, N, A, Ainit, lda );

        free( A );
        free( Ainit );
    }
    else {
        printf( "M= %4d N= %4d : %le GFlop/s\n",
                M, N, gflops );
    }

    tileFree( M, N, b, Atile );

    return rc;
}

int
testall_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                      dgetrf_tiled_fct_t tested_dgetrf )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };
    int all_b[] = { 1, 3, 7, 16 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );
    int nb_b = sizeof( all_b ) / sizeof( int );

    int im, in, ib, m, n, b;
    int nbfailed = 0;
    int nbpassed = 0;
    int nbtests = nb_M * nb_N * nb_b;

    for( im = 0; im < nb_M; im ++ ) {
        m = all_M[im];
        for( in = 0; in < nb_N; in ++ ) {
            n = all_N[in];
            for( ib = 0; ib < nb_b; ib ++ ) {
                b = all_b[ib];

                nbfailed += testone_dgetrf_tiled( dplrnt, tested_dgetrf, m, n, b, 1 );
                nbpassed++;
                fprintf( stdout, "\r %4d / %4d", nbpassed, nbtests );
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
