/**
 *
 * @file dgetrf_tiled_openacc.c
 *
 * @copyright 2019-2023 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP tile-based implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @author Alycia Lisito
 * @date 2023-12-18
 *
 */
#include "myblas.h"

int
dgetrf_tiled_openacc( CBLAS_LAYOUT layout,
                      int M, int N, int b, double **A )
{
#if !defined(ENABLE_OPENACC)
    return ALGONUM_NOT_IMPLEMENTED;
#else
    double Akk[b*b], Amk[b*b], Akn[b*b];
    double *Akkptr, *Amkptr, *Aknptr;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_imin( MT, NT );
    int m, n, k;

    return ALGONUM_NOT_IMPLEMENTED;

    return ALGONUM_SUCCESS; /* Success */
#endif
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_openacc __attribute__ ((unused)) = dgetrf_tiled_openacc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_openacc;

/**
 * @brief Registration function
 */
void dgetrf_tiled_openacc_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_openacc_init( void )
{
    fct_dgetrf_tiled_openacc.openacc = 1;
    fct_dgetrf_tiled_openacc.tiled   = 1;
    fct_dgetrf_tiled_openacc.starpu  = 0;
    fct_dgetrf_tiled_openacc.name    = "openacc";
    fct_dgetrf_tiled_openacc.helper  = "openacc tile-based implementation of the dgetrf";
    fct_dgetrf_tiled_openacc.fctptr  = dgetrf_tiled_openacc;
    fct_dgetrf_tiled_openacc.next    = NULL;

    register_fct( &fct_dgetrf_tiled_openacc, ALGO_GETRF );
}
