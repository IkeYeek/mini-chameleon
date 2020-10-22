/**
 *
 * @file dgetrf_tiled_starpu.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Prototype of a StarPU implementation of the dgetrf.
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
 *
 */
#include "algonum.h"
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

    return 1; /* Not implemented */
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
    fct_dgetrf_tiled_starpu.tiled  = 1;
    fct_dgetrf_tiled_starpu.starpu = 1;
    fct_dgetrf_tiled_starpu.name   = "starpu";
    fct_dgetrf_tiled_starpu.helper = "StarPU implementation of the dgetrf";
    fct_dgetrf_tiled_starpu.fctptr = dgetrf_tiled_starpu;
    fct_dgetrf_tiled_starpu.next   = NULL;

    register_fct( &fct_dgetrf_tiled_starpu, ALGO_GETRF );
}
