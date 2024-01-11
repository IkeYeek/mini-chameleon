/**
 *
 * @file algonum.h
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
#ifndef _algonum_h_
#define _algonum_h_

#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include "config.h"
#include "flops.h"
#include "perf.h"

#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

#define ALGO_GEMM  0
#define ALGO_GETRF 1
#define ALGO_DDOT  2

#define ALGONUM_SUCCESS            0
#define ALGONUM_NOT_IMPLEMENTED -100
#define ALGONUM_FAIL            -101

#define ALGONUM_COLOR_RESET  "\033[0m"
#define ALGONUM_COLOR_GREEN  "\033[32m" /* Green */
#define ALGONUM_COLOR_ORANGE "\033[33m" /* Orange */
#define ALGONUM_COLOR_RED    "\033[31m" /* Red */

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
 * Helper function to compute integer max
 */
static inline int
my_imax( int a, int b )
{
    return ( a < b ) ? b : a;
}

/**
 * Helper function to compute the rank of the owner of the tile A[m, n]
 */
int get_rank_of( int M, int N );

/**
 * Helpers for the matrix conversion from lapack layout to tile layout
 */
double ** lapack2tile( int M, int N, int b, const double *Alapack, int lda );
void      tile2lapack( int M, int N, int b, const double **Atile, double *A, int lda );
void      tileFree( int M, int N, int b, double **A );

/**
 * Data and functions to use CUDA versions
 */
#if defined(ENABLE_CUDA)
extern cublasHandle_t my_cublas_handle;
#endif
void myCublasInit();
void myCublasDestroy();

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
typedef double (*ddot_fct_t)( const int N, const double *X, const int incX,
                                           const double *Y, const int incY );

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
    int         cuda;    /**< True if the function supports Cuda     */
    int         mpi;     /**< True if the function supports MPI      */
    int         openacc; /**< True if the function supports Cuda     */
    int         starpu;  /**< True if the function uses StarPU       */
    int         tiled;   /**< True if the function uses tile storage */
    const char *name;    /**< Short name of the function             */
    const char *helper;  /**< Long description of the implementation */
    void       *fctptr;  /**< function pointer of the implementation */
    fct_list_t *next;    /**< Link to the next implementation        */
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
    int         P, Q, M, N, K, b;
    int         mpirank, mpisize;
    CBLAS_TRANSPOSE transA;
    CBLAS_TRANSPOSE transB;
    int check;
} option_t;

extern option_t global_options;

void print_usage( const char *name, int algo );
void algonum_init( int argc, char **argv, option_t *opts, int algo );
void algonum_exit( option_t *opts, int algo );

/**
 * Testing functions for the scalar dot in LAPACK layout
 */
int check_ddot( const int N, const double *X, const int incX, const double *Y, const int incY, double *res_ref, const double *res );
int testone_ddot( ddot_fct_t ddot, int N, int check );
int testall_ddot( ddot_fct_t ddot );
int testwarm_ddot( ddot_fct_t ddot, int N );

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
int testwarm_dgemm( dgemm_fct_t dgemm,
                    CBLAS_TRANSPOSE transA,
                    CBLAS_TRANSPOSE transB,
                    int M, int N, int K );

/**
 * Testing functions for the LU factorization in LAPACK layout
 */
int check_dgetrf( int M, int N,
                  double *LU, double *A, int lda );
int testone_dgetrf( dgetrf_fct_t dgetrf,
                    int M, int N, int check );
int testall_dgetrf( dgetrf_fct_t tested_dgetrf );
int testwarm_dgetrf( dgetrf_fct_t dgetrf, int M, int N );


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

/**
 * Testing functions for the GPU implementations
 */
int testone_dgetrf_cuda( dgetrf_fct_t dgetrf,
                         int M, int N, int check );
int testone_dgemm_cuda( dgemm_fct_t dgemm,
                        CBLAS_TRANSPOSE transA,
                        CBLAS_TRANSPOSE transB,
                        int M, int N, int K, int check );

#ifdef __cplusplus
}
#endif

#if defined(ENABLE_STARPU)
#include "codelets.h"
#endif

#endif /* _algonum_h_ */
