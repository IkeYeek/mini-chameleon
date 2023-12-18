/**
 *
 * @file dgetrf_tiled_cuda.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP tile-based implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"

int
dgetrf_tiled_cuda( CBLAS_LAYOUT layout,
                   int M, int N, int b, double **A )
{
#if !defined(ENABLE_cuda)
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
static dgetrf_tiled_fct_t valid_dgetrf_tiled_cuda __attribute__ ((unused)) = dgetrf_tiled_cuda;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_cuda;

/**
 * @brief Registration function
 */
void dgetrf_tiled_cuda_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_cuda_init( void )
{
    fct_dgetrf_tiled_cuda.cuda   = 1;
    fct_dgetrf_tiled_cuda.tiled  = 1;
    fct_dgetrf_tiled_cuda.starpu = 0;
    fct_dgetrf_tiled_cuda.name   = "cuda";
    fct_dgetrf_tiled_cuda.helper = "cuda tile-based implementation of the dgetrf";
    fct_dgetrf_tiled_cuda.fctptr = dgetrf_tiled_cuda;
    fct_dgetrf_tiled_cuda.next   = NULL;

    register_fct( &fct_dgetrf_tiled_cuda, ALGO_GETRF );
}
