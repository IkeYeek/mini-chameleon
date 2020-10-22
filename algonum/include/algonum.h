/**
 *
 * @file algonum.h
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Main header file of the library
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#ifndef _algonum_h_
#define _algonum_h_

#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include "flops.h"
#include "perf.h"

#define ALGO_GEMM  0
#define ALGO_GETRF 1

/**
 * Helper function to compute integer ceil
 */
static inline int
my_iceil( int a, int b )
{
    return ( a + b - 1 ) / b;
}

/**
 * Helper function to compute integer min
 */
static inline int
my_imin( int a, int b )
{
    return ( a < b ) ? a : b;
}

/**
 * Helpers for the matrix conversion from lapack layout to tile layout
 */
double ** lapack2tile( int M, int N, int b, const double *Alapack, int lda );
void      tile2lapack( int M, int N, int b, const double **Atile, double *A, int lda );
void      tileFree( int M, int N, int b, double **A );

/**
 * Helpers to generate random matrices in different format
 */
void CORE_dplrnt( double bump, int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed );
void dplrnt_tiled( double bump, int M, int N, int b,
                   double **A, unsigned long long int seed );
void dplrnt_tiled_starpu( double bump, int M, int N, int b,
                          double **A, unsigned long long int seed );

/**
 * Function prototypes
 */
typedef int (*dgemm_fct_t)( CBLAS_LAYOUT layout,
                            CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                            int M, int N, int K,
                            double alpha, const double *A, int lda,
			                  const double *B, int ldb,
			    double beta,        double *C, int ldc );

typedef int (*dgemm_tiled_fct_t)( CBLAS_LAYOUT layout,
                                  CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  int M, int N, int K, int b,
                                  double alpha, const double **A,
                                                const double **B,
                                  double beta,        double **C );

typedef int (*dgetrf_fct_t)( CBLAS_LAYOUT layout,
                             int m, int n, double *A, int lda );

typedef int (*dgetrf_tiled_fct_t)( CBLAS_LAYOUT layout,
                                   int m, int n, int b, double **A );

/**
 * Helper function and variable for the testings
 */
struct fct_list_s;
typedef struct fct_list_s fct_list_t;

/**
 * @brief Data structure to register an implementation of dgemm/dgetrf
 */
struct fct_list_s {
    int         tiled;  /**< If true, the function uses tile format */
    int         starpu; /**< If true, the function uses StarPU      */
    const char *name;   /**< Short name of the function             */
    const char *helper; /**< Long description of the implementation */
    void       *fctptr; /**< function pointer of the implementation */
    fct_list_t *next;   /**< Link to the next implementation        */
};

void register_fct( fct_list_t *fct, int algo );
fct_list_t *search_fct( const char *name, int algo );
void print_fct( int algo );

/**
 * @brief Data structure to read testing parameters
 */
typedef struct option_s {
    fct_list_t *fct;
    int         iter;
    int         M, N, K, b;
    CBLAS_TRANSPOSE transA;
    CBLAS_TRANSPOSE transB;
} option_t;

void print_usage( const char *name, int algo );
void parse_opts( int argc, char **argv, option_t *opts, int algo );

/**
 * Testing functions for the matrix-matrix product in LAPACK layout
 */
int check_dgemm( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                 int M, int N, int K,
                 double alpha, const double *A, int lda, const double *B, int ldb,
                 double beta, double *Cref, const double *C, int ldc );
int testone_dgemm( dgemm_fct_t dgemm,
                   CBLAS_TRANSPOSE transA,
                   CBLAS_TRANSPOSE transB,
                   int M, int N, int K, int check );
int testall_dgemm( dgemm_fct_t tested_dgemm );

/**
 * Testing functions for the LU factorization in LAPACK layout
 */
int check_dgetrf( int M, int N,
                  double *LU, double *A, int lda );
int testone_dgetrf( dgetrf_fct_t dgetrf,
                    int M, int N, int check );
int testall_dgetrf( dgetrf_fct_t tested_dgetrf );


/**
 * Testing functions for the tiled matrix-matrix product int tile layout
 */
typedef void (*dplrnt_tiled_fct_t)( double bump, int M, int N, int b,
                                    double **A, unsigned long long int seed );

int testone_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                         dgemm_tiled_fct_t dgemm,
                         CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB,
                         int M, int N, int K, int b, int check );
int testall_dgemm_tiled( dplrnt_tiled_fct_t dplrnt,
                         dgemm_tiled_fct_t tested_dgemm );

/**
 * Testing functions for the LU factorization in tile layout
 */
int testone_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                          dgetrf_tiled_fct_t dgetrf,
                          int M, int N, int b, int check );
int testall_dgetrf_tiled( dplrnt_tiled_fct_t dplrnt,
                          dgetrf_tiled_fct_t tested_dgetrf );

#if defined(ENABLE_STARPU)
#include "codelets.h"
#endif

#endif /* _algonum_h_ */
