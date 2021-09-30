/**
 *
 * @file codelet_dplrnt.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief dplrnt StarPU codelet
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 * This file describes the StarPU codelet for the dplrnt function that
 * is an helper to generate a random matrix.
 *
 */
#include "algonum.h"
#include "codelets.h"

/**
 * @brief Structure to gather static parameters of the kernel
 */
typedef struct cl_dplrnt_arg_s {
    double bump;
    int m;
    int n;
    int lda;
    int bigM;
    int m0;
    int n0;
    unsigned long long int seed;
} cl_dplrnt_arg_t;

/**
 * @brief Codelet CPU function
 */
static void
cl_dplrnt_cpu_func( void *descr[], void *cl_arg )
{
    cl_dplrnt_arg_t args;
    double *A;

    A = tile_interface_get(descr[0]);

    starpu_codelet_unpack_args( cl_arg, &args );

    CORE_dplrnt( args.bump, args.m, args.n, A, args.lda,
                 args.bigM, args.m0, args.n0, args.seed );
}

/**
 * @brief Define the StarPU codelet structure
 */
struct starpu_codelet cl_dplrnt = {
    .where      = STARPU_CPU,
    .cpu_func   = cl_dplrnt_cpu_func,
    .cuda_flags = { 0 },
    .cuda_func  = NULL,
    .nbuffers   = 1,
    .name       = "plrnt"
};

/**
 * @brief Insert task function
 *
 * @param[in] bump
 *       Scalar value to eventually add to the diagonal of the entire generated
 *       matrix in order to make it diagonal dominant.
 *
 * @param[in] m
 *       The number of rows of the matrix A.
 *
 * @param[in] n
 *       The number of columns of the matrix A.
 *
 * @param[in,out] A
 *       The StarPU data handle to a matrix A of size lda-by-n.
 *
 * @param[in] lda
 *       The leading dimension of the matrix A. lda >= max( m, 1 ).
 *
 * @param[in] bigM
 *       The number of rows of the entire matrix that contains the submatrix A.
 *
 * @param[in] m0
 *       The row index of the first element of the submatrix A in the entire generated matrix.
 *
 * @param[in] n0
 *       The column index of the first element of the submatrix A in the entire generated matrix.
 *
 * @param[in] seed
 *       The seed used by random generator to generate the full matrix. Must be
 *       the same value for all the call with respect to the same generated
 *       matrix.
 */
void
insert_dplrnt( double               bump,
               int                  m,
               int                  n,
               starpu_data_handle_t A,
               int                  lda,
               int                  bigM,
               int                  m0,
               int                  n0,
               unsigned long long int seed )
{
    cl_dplrnt_arg_t args = {
        .bump = bump,
        .m    = m,
        .n    = n,
        .lda  = lda,
        .bigM = bigM,
        .m0   = m0,
        .n0   = n0,
        .seed = seed,
    };

    starpu_insert_task(
        starpu_mpi_codelet(&cl_dplrnt),
        STARPU_VALUE, &args, sizeof(cl_dplrnt_arg_t),
        STARPU_W,      A,
        0);
}
