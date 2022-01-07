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
    int K  = my_imin( M, N );
    int KT = my_imin( MT, NT );
    int m, n, k;

    /* Let's allocate data handlers for all pieces of data */
    handlesA = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    for( k=0; k<KT; k++) {
        int kk = k == (KT-1) ? K - k * b : b;

        hAkk = get_starpu_handle( 0, handlesA, A, k, k, b, MT );

        insert_dgetrf( kk, kk, hAkk, b );

        for( n=k+1; n<NT; n++) {
            int nn = n == (NT-1) ? N - n * b : b;

            hAkn = get_starpu_handle( 0, handlesA, A, k, n, b, MT );

            insert_dtrsm( CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                          kk, nn, 1., hAkk, b, hAkn, b );
        }

        for( m=k+1; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

            hAmk = get_starpu_handle( 0, handlesA, A, m, k, b, MT );

            insert_dtrsm( CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                          mm, kk, 1., hAkk, b, hAmk, b );

            for( n=k+1; n<NT; n++) {
                int nn = n == (NT-1) ? N - n * b : b;

                hAmn = get_starpu_handle( 0, handlesA, A, m, n, b, MT );

                insert_dgemm( CblasNoTrans, CblasNoTrans, mm, nn, kk,
                              -1., hAmk, b, hAkn, b, 1., hAmn, b );
            }
        }
    }

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
