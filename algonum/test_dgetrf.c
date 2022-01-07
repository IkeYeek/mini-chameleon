/**
 *
 * @file test_dgetrf.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgetrf variants on lapack format.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum_int.h"

int
testone_dgetrf( dgetrf_fct_t dgetrf,
                int M, int N, int check )
{
    int     rc = 0;
    double *A;
    int     lda;
    int     seedA = random();
    perf_t  start, stop;
    int minMN = ( M < N ) ? M : N;

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );
    A = malloc( lda * N * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( minMN, M, N, A, lda, M, 0, 0, seedA );

    /* Calculate the product */
    perf( &start );
    rc = dgetrf( CblasColMajor, M, N, A, lda );
    perf( &stop );

    if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
        printf( "M= %4d N= %4d:" ALGONUM_COLOR_ORANGE " not supported or not implemented\n" ALGONUM_COLOR_RESET,
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
        double *Ainit = malloc( lda * N  * sizeof(double) );
        CORE_dplrnt( minMN, M, N, Ainit, lda, M, 0, 0, seedA );

        rc = check_dgetrf( M, N, A, Ainit, lda );

        if ( rc == ALGONUM_SUCCESS) {
            printf( "M= %4d N= %4d: %le GFlop/s"
                    ALGONUM_COLOR_GREEN " SUCCESS\n" ALGONUM_COLOR_RESET, M, N,
                    gflops);
        } else {
            printf( "M= %4d N= %4d: %le GFlop/s"
                    ALGONUM_COLOR_RED " FAIL\n" ALGONUM_COLOR_RESET, M, N,
                    gflops);
        }

        free( Ainit );
    }
    else {
        printf( "M= %4d N= %4d: %le GFlop/s\n",
                M, N, gflops );
    }

    free( A );

    return rc;
}

int
testall_dgetrf( dgetrf_fct_t tested_dgetrf )
{
    int all_M[] = { 0, 5, 8, 17, 57 };
    int all_N[] = { 0, 3, 5, 17, 64 };

    int nb_M = sizeof( all_M ) / sizeof( int );
    int nb_N = sizeof( all_N ) / sizeof( int );

    int rc;
    int im, in, m, n;
    int nbfailed = 0;
    int nbnotimplemented = 0;
    int nbpassed = 0;
    int nbtests = nb_M * nb_N;

    for( im = 0; im < nb_M; im ++ ) {
        m = all_M[im];
        for( in = 0; in < nb_N; in ++ ) {
            n = all_N[in];

            rc = testone_dgetrf( tested_dgetrf, m, n, 1 );
            if ( rc == ALGONUM_FAIL ) {
                nbfailed++;
            } else if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
                nbnotimplemented++;
            }
            nbpassed++;
            printf("Test %4d / %4d\n", nbpassed, nbtests);
        }
    }

    if ( nbnotimplemented > 0 ) {
        printf( ALGONUM_COLOR_ORANGE
                " Warning: %4d tests out of %d could not be assessed due to "
                "non implemented variants\n" ALGONUM_COLOR_RESET,
                nbnotimplemented, nbtests );
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
        if (nbtests != nbnotimplemented) {
            printf( ALGONUM_COLOR_GREEN
                    "\n Congratulations all %4d tests succeeded\n" ALGONUM_COLOR_RESET,
                    nbtests - nbnotimplemented );
        }
    }
    return nbfailed;
}

int
testwarm_dgetrf( dgetrf_fct_t dgetrf,
                 int M, int N )
{
    int     rc = 0;
    double *A;
    int     lda, i;
    int     seedA = random();
    perf_t  start, stop;
    int64_t nbiter;
    int     minMN = my_imin( M, N );

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );
    A = malloc( lda * N * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( minMN, M, N, A, lda, M, 0, 0, seedA );

    /* Estimate the number of iteration to get a 2 seconds run @ 10GFLop/s */
    nbiter = ( 2. / flops ) * 1e10;
    nbiter = my_imin( 1000000, nbiter );
    nbiter = my_imax( 3, nbiter );

    /* Calculate the product */
    perf( &start );
    for( i=0; i<nbiter; i++ ) {
        rc += dgetrf( CblasColMajor, M, N, A, lda );
    }
    perf( &stop );

    if ( rc ) {
        fprintf( stderr,
                 "M= %4d N= %4d: Not Supported or not implemented\n",
                 M, N );
        return rc;
    }

    perf_diff( &start, &stop );
    if ( flops > 0. ) {
        gflops = perf_gflops( &stop, flops * nbiter );
    }
    else {
        gflops = 0.;
    }

    /* Check the solution */
    printf( "M= %4d N= %4d : %le GFlop/s (%7ld iterations)\n",
            M, N, gflops, nbiter );

    free( A );

    return rc;
}
