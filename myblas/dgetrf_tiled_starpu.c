/**
 *
 * @file dgetrf_tiled_starpu.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Prototype of a StarPU implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include "codelets.h"

int
dgetrf_tiled_starpu( CBLAS_LAYOUT layout,
                     int M, int N, int b, double **A )
{
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t hAkk, hAkn, hAmk, hAmn;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_imin( MT, NT );
    int m, n, k;

    /* Let's allocate data handlers for all pieces of data */
    handlesA = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    // ADD STARPU GETRF ALGORITHM RIGHT HERE

    /* Let's submit unregistration of all data handlers */
    unregister_starpu_handle( MT * NT, handlesA );

    /* Let's wait for the end of all the tasks */
    starpu_task_wait_for_all();
#if defined(ENABLE_MPI)
    starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

    free( handlesA );

    return ALGONUM_SUCCESS; /* Success */
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_starpu __attribute__ ((unused)) = dgetrf_tiled_starpu;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_starpu;

/**
 * @brief Registration function
 */
void dgetrf_tiled_starpu_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_starpu_init( void )
{
    fct_dgetrf_tiled_starpu.mpi    = 0;
    fct_dgetrf_tiled_starpu.tiled  = 1;
    fct_dgetrf_tiled_starpu.starpu = 1;
    fct_dgetrf_tiled_starpu.name   = "starpu";
    fct_dgetrf_tiled_starpu.helper = "StarPU implementation of the dgetrf";
    fct_dgetrf_tiled_starpu.fctptr = dgetrf_tiled_starpu;
    fct_dgetrf_tiled_starpu.next   = NULL;

    register_fct( &fct_dgetrf_tiled_starpu, ALGO_GETRF );
}
