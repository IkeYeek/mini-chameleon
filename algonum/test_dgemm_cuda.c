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
#if defined( ENABLE_CUDA )
#include <cublas_v2.h>
#else
#error "This file should not be compiled if CUDA is not enabled"
#endif

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
testone_dgemm_cuda( dgemm_fct_t     dgemm,
                    CBLAS_TRANSPOSE transA,
                    CBLAS_TRANSPOSE transB,
                    int M, int N, int K, int check )
{
    int     Am, An, Bm, Bn;
    int     rc = 0;
    double *cpuA, *cpuB, *cpuC;
    double *gpuA, *gpuB, *gpuC;
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
    cpuA = malloc( lda * An * sizeof(double) );
    cpuB = malloc( ldb * Bn * sizeof(double) );
    cpuC = malloc( ldc * N  * sizeof(double) );

    /* Fill the matrices with random values */
    CORE_dplrnt( 0, Am, An, cpuA, lda, Am, 0, 0, seedA );
    CORE_dplrnt( 0, Bm, Bn, cpuB, ldb, An, 0, 0, seedB );
    CORE_dplrnt( 0, M,  N,  cpuC, ldc, M,  0, 0, seedC );

    cudaMalloc( (void **)&gpuA, sizeof(double) * lda * An );
    cudaMalloc( (void **)&gpuB, sizeof(double) * ldb * Bn );
    cudaMalloc( (void **)&gpuC, sizeof(double) * ldc * N  );
    cudaMemcpy( gpuA, cpuA, sizeof(double) * lda * An, cudaMemcpyHostToDevice );
    cudaMemcpy( gpuB, cpuB, sizeof(double) * ldb * Bn, cudaMemcpyHostToDevice );
    cudaMemcpy( gpuC, cpuC, sizeof(double) * ldc * N,  cudaMemcpyHostToDevice );

    /* Calculate the product */
    perf( &start );
    rc = dgemm( CblasColMajor, transA, transB, M, N, K,
                alpha, gpuA, lda, gpuB, ldb,
                beta, gpuC, ldc );
    perf( &stop );

    if ( rc == ALGONUM_NOT_IMPLEMENTED ) {
        printf(  "tA=%s tB=%s M= %4d N= %4d K= %4d: " ALGONUM_COLOR_ORANGE "Not supported or not implemented\n" ALGONUM_COLOR_RESET,
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

    cudaFree( gpuA );
    cudaFree( gpuB );

    /* Check the solution */
    if ( check ) {
        double *Cinit = malloc( ldc * N  * sizeof(double) );
        CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, seedC );

        cudaMemcpy( cpuC, gpuC, sizeof(double) * ldc * N,  cudaMemcpyDeviceToHost );
        rc = check_dgemm( transA, transB, M, N, K,
                          alpha, cpuA, lda, cpuB, ldb,
                          beta, Cinit, cpuC, ldc );

        if ( rc == ALGONUM_SUCCESS) {
            printf( "tA=%s tB=%s M= %4d N= %4d K= %4d alpha= %e beta= %e: %14f GFlop/s: " ALGONUM_COLOR_GREEN "SUCCESS\n" ALGONUM_COLOR_RESET,
                    (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                    (transB == CblasNoTrans) ? "NoTrans" : "Trans", M, N, K, alpha,
                    beta, gflops);
        } else {
            printf( "tA=%s tB=%s M= %4d N= %4d K= %4d alpha= %e beta= %e: %14f GFlop/s: " ALGONUM_COLOR_RED
                    "FAIL\n" ALGONUM_COLOR_RESET,
                    (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                    (transB == CblasNoTrans) ? "NoTrans" : "Trans", M, N, K, alpha,
                    beta, gflops );
        }
        free( Cinit );
    }
    else {
        printf( "tA=%s tB=%s M= %4d N= %4d K= %4d: %14f GFlop/s\n",
                (transA == CblasNoTrans) ? "NoTrans" : "Trans",
                (transB == CblasNoTrans) ? "NoTrans" : "Trans",
                M, N, K, gflops );
    }

    free( cpuA );
    free( cpuB );
    free( cpuC );
    cudaFree( gpuC );

    return rc;
}
