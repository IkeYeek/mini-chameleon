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

#ifdef __cplusplus
}
#endif

#endif /* _algonum_int_h_ */
