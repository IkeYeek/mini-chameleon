/**
 *
 * @file starpu_dplrnt.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief StarPU dplrnt function
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 * This file contains the StarPU algorithm to generate a random tiled matrix.
 *
 */
#include "algonum.h"
#include "codelets.h"

/**
 * @brief Function to generate a random tiled matrix in parallel throug task
 *        insertion.
 *
 * @param[in] bump
 *       Scalar value to eventually add to the diagonal of the entire generated
 *       matrix in order to make it diagonal dominant.
 *
 * @param[in] M
 *       The number of rows of the matrix A.
 *
 * @param[in] N
 *       The number of columns of the matrix A.
 *
 * @param[in] b
 *       The size b-by-b of each tile in the matrix A.
 *
 * @param[in] A
 *       An array of pointer of size ((M + b - 1) / b) -by- ((N + b - 1) / b)
 *       to all the tiles of the matrix A of size M-by-N to generate.
 *
 * @param[in] seed
 *       The seed used by random generator to generate the full matrix. Must be
 *       the same value for all the call with respect to the same generated
 *       matrix.
 */
void
dplrnt_tiled_starpu( double bump, int M, int N, int b,
                     double **A, unsigned long long int seed )
{
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t hA;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int m, n, mm, nn;

    /* Allocate arraya of StarPU handles */
    handlesA = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    for( m=0; m<MT; m++ ) {
        mm = m == (MT-1) ? M - m * b : b;

        for( n=0; n<NT; n++ ) {
            nn = n == (NT-1) ? N - n * b : b;

            hA = get_starpu_handle( 0, handlesA, A, m, n, b, MT );

            /* Let's insert a task to perform the generation fo the tile */
            insert_dplrnt( bump, mm, nn, hA, b,
                           M, m * b, n * b, seed );
        }
    }

    /* Submit the unregistration of the handles */
    unregister_starpu_handle( MT * NT, handlesA );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesA );
}
