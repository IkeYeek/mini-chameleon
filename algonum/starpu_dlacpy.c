/**
 *
 * @file starpu_dlacpy.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief StarPU dlacpy function
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 * This file contains the StarPU algorithm to copy a tiled matrix into a lapack
 * layout matrix through StarPU codelets.
 *
 */
#include "algonum.h"
#include "codelets.h"

/**
 * @brief Function to copy a tiled matrix into a lapack layout matrix through
 *        task insertion.
 *
 * @param[in] M
 *       The number of rows of the matrices Atile and Alapack.
 *
 * @param[in] N
 *       The number of columns of the matrices Atile and Alapack.
 *
 * @param[in] b
 *       The size b-by-b of each tile in the matrix Atile.
 *
 * @param[in] Atile
 *       An array of pointer of size ((M + b - 1) / b) -by- ((N + b - 1) / b)
 *       to all the tiles of the matrix Atile of size M-by-N to copy.
 *
 * @param[in,out] Alapack
 *       The allocated matrix Alapack of size lda-by-N into which the matrix is
 *       copied.
 *
 * @param[in] lda
 *       The leading dimension of the matrix Alapack. lda >= max( 1, M ).
 */
void
tile2lapack_starpu( int M, int N, int b,
                    const double **Atile,
                    double *Alapack, int lda )
{
    starpu_data_handle_t *handlesAt;
    starpu_data_handle_t *handlesAl;
    starpu_data_handle_t hAt, hAl;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int m, n, mm, nn;

    /* Allocate arrays of StarPU handles */
    handlesAt = calloc( MT * NT, sizeof(starpu_data_handle_t) );
    handlesAl = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    assert( lda >= M );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        nn = n == (NT-1) ? N - n * b : b;

        for( m=0; m<MT; m++) {
            mm = m == (MT-1) ? M - m * b : b;

            hAt = get_starpu_handle( 0, handlesAt, (double**)Atile, m, n, b, MT );
            hAl = get_starpu_handle_lap( 1, handlesAl + n * MT + m,
                                         m, n, mm, nn,
                                         Alapack + n * lda + m, lda, MT );

            /* Let's insert a task to perform the copy */
            insert_dlacpy( mm, nn, hAt, b, hAl, lda );
        }
    }

    /* Submit the unregistration of the handles */
    unregister_starpu_handle( MT * NT, handlesAt );
    unregister_starpu_handle( MT * NT, handlesAl );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesAt );
    free( handlesAl );
}
