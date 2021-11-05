/**
 *
 * @file starpu.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief StarPU management module
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 * This file contains all the StarPU management functions.
 *
 */
#include "algonum.h"
#include "codelets.h"

void
my_starpu_init()
{
    int hres;
    hres = starpu_init(NULL);
#if defined(ENABLE_MPI)
    {
        int flag = 0;
        MPI_Initialized( &flag );

        hres = starpu_mpi_init( NULL, NULL, !flag );
    }
#endif

#if defined(ENABLE_CUDA)
    starpu_cublas_init();
#endif

    (void)hres;
}

void
my_starpu_exit()
{
#if defined(ENABLE_CUDA)
    starpu_cublas_init();
#endif

#if defined(ENABLE_MPI)
    starpu_mpi_shutdown();
#else
    starpu_shutdown();
#endif
}

static inline int
get_starpu_rank()
{
#if defined(ENABLE_MPI)
    int rank;
    starpu_mpi_comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
#else
    return 0;
#endif
}

starpu_data_handle_t
get_starpu_handle( int id, starpu_data_handle_t *handles, double **A, int m, int n, int b, int MT )
{
    starpu_data_handle_t *tile_handle = handles + n * MT + m;

    /* If the starpu_data_handle_t is NULL, we need to register the data */
    if ( *tile_handle == NULL ) {
        int home_node = -1;
        void *user_ptr = NULL;
        int myrank = get_starpu_rank();
        int owner  = get_rank_of( m, n );

        if ( myrank == owner ) {
            user_ptr = A[ MT * n + m ];
            if ( user_ptr != NULL ) {
                home_node = STARPU_MAIN_RAM;
            }
        }

        starpu_matrix_data_register( tile_handle, home_node, (uintptr_t) user_ptr,
                                     b, b, b, sizeof( double ) );

        starpu_data_set_coordinates( *tile_handle, 2, m, n );

#if defined(ENABLE_MPI)
        /**
         * We need to give a unique tag to each data
         * Be careful to take into account the multiple decriptors that can be used in parallel.
         */
        {
            int64_t tag = ((int64_t)id << 32) | ( n * MT + m );
            starpu_mpi_data_register( *tile_handle, tag, owner );
        }
#endif /* defined(CHAMELEON_USE_MPI) */
    }

    return *tile_handle;
}

starpu_data_handle_t
get_starpu_handle_lap( int id, starpu_data_handle_t *handle,
                       int i, int j, int m, int n, double *A, int lda, int MT )
{
    /* If the starpu_data_handle_t is NULL, we need to register the data */
    if ( *handle == NULL ) {
        int home_node = -1;
        void *user_ptr = NULL;
        int myrank = get_starpu_rank();
        int owner = 0;

        if ( myrank == owner ) {
            user_ptr = A;
            if ( user_ptr != NULL ) {
                home_node = STARPU_MAIN_RAM;
            }
        }

        starpu_matrix_data_register( handle, home_node, (uintptr_t) user_ptr,
                                     lda, m, n, sizeof( double ) );

        starpu_data_set_coordinates( *handle, 2, i, j );

#if defined(ENABLE_MPI)
        /**
         * We need to give a unique tag to each data
         * Be careful to take into account the multiple decriptors that can be used in parallel.
         */
        {
            int64_t tag = ((int64_t)id << 32) | ( j * MT + i );
            starpu_mpi_data_register( *handle, tag, owner );
        }
#endif /* defined(CHAMELEON_USE_MPI) */
    }

    return *handle;
}

void
unregister_starpu_handle( int nb, starpu_data_handle_t *handles )
{
    int i;
    for (i=0; i<nb; i++, handles++) {
        if (*handles != NULL) {
            starpu_data_unregister_submit( *handles );
        }
    }
}
