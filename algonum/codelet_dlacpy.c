/**
 *
 * @file codelet_dlacpy.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dlacpy StarPU codelet
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 * This file describes the StarPU codelet for the LAPACK dlacpy function that
 * copies a matrix A into a matrix B.
 *
 */
#include "codelets.h"

/**
 * @brief Structure to gather static parameters of the kernel
 */
typedef struct cl_dlacpy_arg_s {
    int m;
    int n;
    int lda;
    int ldb;
} cl_dlacpy_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dlacpy_cpu_func( void *descr[], void *cl_arg )
{
    cl_dlacpy_arg_t args;
    double *A;
    double *B;

    A = tile_interface_get(descr[0]);
    B = tile_interface_get(descr[1]);

    starpu_codelet_unpack_args( cl_arg, &args );

    LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A',
                         args.m, args.n, A, args.lda, B, args.ldb );
}

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dlacpy = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dlacpy_cpu_func,
    .cuda_flags = { 0 },
    .cuda_func  = NULL,
    .nbuffers   = 2,
    .name       = "lacpy"
};

/**
 * @brief Insert task function
 *
 * @param[in] m
 *       The number of rows of the matrices A and B.
 *
 * @param[in] n
 *       The number of columns of the matrices A and B.
 *
 * @param[in] A
 *       The StarPU data handle to a matrix A of size lda-by-n.
 *
 * @param[in] lda
 *       The leading dimension of the matrix A. lda >= max( m, 1 ).
 *
 * @param[in,out] B
 *       The StarPU data handle to a matrix B of size ldb-by-n.
 *
 * @param[in] ldb
 *       The leading dimension of the matrix B. ldb >= max( m, 1 ).
 */
void
insert_dlacpy( int                  m,
               int                  n,
               starpu_data_handle_t A,
               int                  lda,
               starpu_data_handle_t B,
               int                  ldb )
{
    cl_dlacpy_arg_t args = {
        .m      = m,
        .n      = n,
        .lda    = lda,
        .ldb    = ldb,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dlacpy),
        STARPU_VALUE, &args, sizeof(cl_dlacpy_arg_t),
        STARPU_R,      A,
        STARPU_W,      B,
        0);
}
