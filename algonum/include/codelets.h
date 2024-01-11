/**
 *
 * @file codelets.h
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief StarPU codelets declaration header
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#ifndef _codelets_h_
#define _codelets_h_

#if defined(ENABLE_MPI)
#include <starpu_mpi.h>
#else
#include <starpu.h>
#endif

#include <cblas.h>

#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

/**
 * @brief Enable this if you want to serialize the task submission and execution
 */
#if defined(ENABLE_RUNTIME_SYNC)
#define TASK_SYNCHRONOUS , STARPU_TASK_SYNCHRONOUS, 1
#else
#define TASK_SYNCHRONOUS
#endif

/**
 * @brief Modify the insert call if MPI is used
 */
#if defined(ENABLE_MPI)
#define starpu_insert_task starpu_mpi_insert_task
#define starpu_mpi_codelet(_codelet_) MPI_COMM_WORLD, _codelet_ TASK_SYNCHRONOUS
#else
#define starpu_mpi_codelet(_codelet_) _codelet_ TASK_SYNCHRONOUS
#endif

/**
 * Include lapacke
 */
#include <lapacke.h>

/**
 * Control functions
 */
void my_starpu_init();
void my_starpu_exit();
starpu_data_handle_t get_starpu_handle( int id, starpu_data_handle_t *handles, double **A, int m, int n, int b, int MT );
starpu_data_handle_t get_starpu_handle_lap( int id, starpu_data_handle_t *handle,
                                            int i, int j, int m, int n, double *A, int lda, int mt );
void unregister_starpu_handle( int nb, starpu_data_handle_t *handlesA );

/**
 * Data access
 */
static inline double *
tile_interface_get( void *interface )
{
    /* STARPU_MATRIX_CHECK( interface ); */
    return (double *)(((struct starpu_matrix_interface *)(interface))->ptr);
}

int get_starpu_owner( int m, int n );

/**
 * Insert task functions
 */
void insert_dplrnt( double bump, int m, int n, starpu_data_handle_t A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed );
void insert_dlacpy( int m, int n, starpu_data_handle_t A, int lda, starpu_data_handle_t B, int ldb );
void insert_dgetrf( int m, int n, starpu_data_handle_t A, int lda );
void insert_dtrsm ( CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG diag,
                    int m, int n, double alpha, starpu_data_handle_t A, int lda, starpu_data_handle_t B, int ldb );
void insert_dgemm ( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k,
                    double alpha, starpu_data_handle_t A, int lda,
                    starpu_data_handle_t B, int ldb,
                    double beta, starpu_data_handle_t C, int ldc );

#ifdef __cplusplus
}
#endif

#endif /* _codelets_h_ */
