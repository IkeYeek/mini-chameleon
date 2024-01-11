/**
 *
 * @file algonum_int.h
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Main header file of the library
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#ifndef _algonum_int_h_
#define _algonum_int_h_

#include <stdio.h>
#include <stdlib.h>
#include "algonum.h"

#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

static inline int
max( int M, int N )
{
    return ( M > N ) ? M : N;
}

extern dgemm_fct_t dgemm_seq, dgemm_omp;

void CORE_dplrnt( double bump, int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed );

#if defined( ENABLE_CUDA )
#include <cublas_v2.h>

static inline cublasSideMode_t
get_cublas_side( CBLAS_SIDE side )
{
    if ( side == CblasLeft ) {
        return CUBLAS_SIDE_LEFT;
    }
    else {
        return CUBLAS_SIDE_RIGHT;
    }
}

static inline cublasFillMode_t
get_cublas_uplo( CBLAS_UPLO uplo )
{
    if ( uplo == CblasUpper ) {
        return CUBLAS_FILL_MODE_UPPER;
    }
    else {
        return CUBLAS_FILL_MODE_LOWER;
    }
}

static inline cublasOperation_t
get_cublas_trans( CBLAS_TRANSPOSE trans )
{
    if ( trans == CblasNoTrans ) {
        return CUBLAS_OP_N;
    }
    else if ( trans == CblasTrans ) {
        return CUBLAS_OP_T;
    }
    else {
        return CUBLAS_OP_C;
    }
}

static inline cublasDiagType_t
get_cublas_diag( CBLAS_DIAG diag )
{
    if ( diag == CblasNonUnit ) {
        return CUBLAS_DIAG_NON_UNIT;
    }
    else {
        return CUBLAS_DIAG_UNIT;
    }
}
#endif /* defined(ENABLE_CUDA) */

#ifdef __cplusplus
}
#endif

#endif /* _algonum_int_h_ */
