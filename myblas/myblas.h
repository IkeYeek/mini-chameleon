/**
 *
 * @file myblas.h
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Header of all the blas implementations
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#ifndef _myblas_h_
#define _myblas_h_

#include "algonum.h"
#include <assert.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

/**
 * @brief Helper to read environment variable in order to change some
 * parameters in your code
 *
 * @param[in] string
 *        The name of the environment variable to read
 *
 * @param[in] default_value
 *        The default value to return is the variable is not set
 *
 * @return The value read in the environment if defined, the default
 * value otherwise.
 */
static inline int
myblas_getenv_value_int(char * string, int default_value) {
    long int ret;
    char *str = getenv( string );
    if (str == NULL) return default_value;

    if ( sscanf( str, "%ld", &ret ) != 1 ) {
        perror("sscanf");
        return default_value;
    }

    return (int)ret;
}

double ddot_seq( int N, const double *X, int incX,
                 const double *Y, int incY );

int dgemm_seq( CBLAS_LAYOUT layout,
               CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
               int M, int N, int K,
               double alpha, const double *A, int lda,
                             const double *B, int ldb,
               double beta,        double *C, int ldc );
int dgemm_vendor( CBLAS_LAYOUT layout,
                  CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K,
                  double alpha, const double *A, int lda,
                                const double *B, int ldb,
                  double beta,        double *C, int ldc );
int dgemm_omp( CBLAS_LAYOUT layout,
               CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
               int M, int N, int K,
               double alpha, const double *A, int lda,
                             const double *B, int ldb,
               double beta,        double *C, int ldc );


int dgemm_tiled_omp( CBLAS_LAYOUT layout,
                     CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int M, int N, int K, int b,
                     double alpha, const double **A,
                                   const double **B,
                     double beta,        double **C );
int dgemm_tiled_starpu( CBLAS_LAYOUT layout,
                        CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                        int M, int N, int K, int b,
                        double alpha, const double **A,
                                     const double **B,
                        double beta,        double **C );


int dgetrf_seq( CBLAS_LAYOUT layout,
                int m, int n, double *A, int lda );
int dgetrf_vendor( CBLAS_LAYOUT layout,
                int m, int n, double *A, int lda );
int dgetrf_omp( CBLAS_LAYOUT layout,
                int m, int n, double *A, int lda );

int dgetrf_tiled_omp( CBLAS_LAYOUT layout,
                      int M, int N, int b, double **A );
int dgetrf_tiled_starpu( CBLAS_LAYOUT layout,
                         int M, int N, int b, double **A );

#ifdef __cplusplus
}
#endif

#endif /* _myblas_h_ */
