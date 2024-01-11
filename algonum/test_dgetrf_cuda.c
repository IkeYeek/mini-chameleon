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
#if !defined( ENABLE_CUDA )
#error "This file should not be compiled if CUDA is not enabled"
#endif

int
testone_dgetrf_cuda( dgetrf_fct_t dgetrf,
                     int M, int N, int check )
{
    int     rc = 0;
    double *cpuA;
    double *gpuA;
    int     lda;
    int     seedA = random();
    perf_t  start, stop;
    int minMN = ( M < N ) ? M : N;

    double gflops;
    double flops = flops_dgetrf( M, N );

    /* Create the matrices */
    lda = max( M, 1 );
    cpuA = malloc( lda * N * sizeof(double) );
    cudaMalloc( (void **)&gpuA, sizeof(double) * lda * N );

    /* Fill the matrices with random values */
    CORE_dplrnt( minMN, M, N, cpuA, lda, M, 0, 0, seedA );

    cudaMemcpy( gpuA, cpuA, sizeof(double) * lda * N, cudaMemcpyHostToDevice );

    /* Calculate the product */
    perf( &start );
    rc = dgetrf( CblasColMajor, M, N, gpuA, lda );
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

        cudaMemcpy( cpuA, gpuA, sizeof(double) * lda * N, cudaMemcpyDeviceToHost );

        rc = check_dgetrf( M, N, cpuA, Ainit, lda );

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

    free( cpuA );
    cudaFree( gpuA );

    return rc;
}
