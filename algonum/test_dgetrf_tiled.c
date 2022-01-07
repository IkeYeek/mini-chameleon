/**
 *
 * @file test_dgetrf_tiled.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgetrf variants in tile format.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
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
    int      seedA = 4562;
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

    if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
        if ( global_options.mpirank == 0 ) {
            printf( "M= %4d N= %4d: " ALGONUM_COLOR_ORANGE "not supported or not implemented\n" ALGONUM_COLOR_RESET,
                    M, N );
        }

        tileFree( M, N, b, Atile );
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
        double *A, *Ainit;

        /* Create the matrices for the test */
        if ( global_options.mpirank == 0 ) {
            A     = malloc( lda * N * sizeof(double) );
            Ainit = malloc( lda * N * sizeof(double) );
            CORE_dplrnt( minMN, M, N, Ainit, lda, M, 0, 0, seedA );
        }

        tile2lapack( M, N, b, (const double **)Atile, A, lda );

        if ( global_options.mpirank == 0 ) {
            rc = check_dgetrf( M, N, A, Ainit, lda );

            if ( rc == ALGONUM_SUCCESS ) {
                printf( "M= %4d N= %4d b=%3d: %le GFlop/s " ALGONUM_COLOR_GREEN
                        "SUCCESS\n" ALGONUM_COLOR_RESET,
                        M, N, b, gflops );
            } else {
                printf( "M= %4d N= %4d b=%3d: %le GFlop/s " ALGONUM_COLOR_RED
                        "FAIL\n" ALGONUM_COLOR_RESET,
                        M, N, b, gflops );
            }
            free( A );
            free( Ainit );
        }
    }
    else {
        if ( global_options.mpirank == 0 ) {
            printf( "M= %4d N= %4d b=%3d: %le GFlop/s\n",
                    M, N, b, gflops );
        }
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

    int rc;
    int im, in, ib, m, n, b;
    int nbfailed = 0;
    int nbnotimplemented = 0;
    int nbpassed = 0;
    int nbtests = nb_M * nb_N * nb_b;

    for( im = 0; im < nb_M; im ++ ) {
        m = all_M[im];
        for( in = 0; in < nb_N; in ++ ) {
            n = all_N[in];
            for( ib = 0; ib < nb_b; ib ++ ) {
                b = all_b[ib];

                rc = testone_dgetrf_tiled( dplrnt, tested_dgetrf, m, n, b, 1 );
                if ( rc == ALGONUM_FAIL ) {
                    nbfailed++;
                }
                else if (rc == ALGONUM_NOT_IMPLEMENTED) {
                    nbnotimplemented++;
                }
                nbpassed++;
                if ( global_options.mpirank == 0 ) {
                    printf( "Test %4d / %4d\n", nbpassed, nbtests );
                }
            }
        }
    }

    if ( global_options.mpirank == 0 ) {
        if ( nbnotimplemented > 0 ) {
            printf( ALGONUM_COLOR_ORANGE
                    "Warning: %4d tests out of %d could not be assessed due to "
                    "non implemented variants\n" ALGONUM_COLOR_RESET,
                    nbnotimplemented, nbtests );
        }
        else {
            printf( "All tested variants were implemented\n" );
        }
        if ( nbfailed > 0 ) {
            printf( ALGONUM_COLOR_RED
                    "%4d tests failed out of %d\n" ALGONUM_COLOR_RESET,
                    nbfailed, nbtests - nbnotimplemented);
        }
        else {
            if ( nbtests != nbnotimplemented ) {
                printf(ALGONUM_COLOR_GREEN "Congratulations: all %4d tests "
                       "succeeded\n" ALGONUM_COLOR_RESET,
                       nbtests - nbnotimplemented);
            }
        }
    }
    return nbfailed;
}
