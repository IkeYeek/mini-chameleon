/**
 *
 * @file dgemm_tiled_starpu.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Prototype of a StarPU implementation of the dgemm.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"
#include "codelets.h"

int
dgemm_tiled_starpu( CBLAS_LAYOUT layout,
                    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                    int M, int N, int K, int b,
                    double alpha, const double **A,
                                  const double **B,
                    double beta,        double **C )
{
    starpu_data_handle_t *handlesA;
    starpu_data_handle_t *handlesB;
    starpu_data_handle_t *handlesC;
    starpu_data_handle_t hA, hB, hC;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );
    int m, n, k;

    /* Let's allocate data handlers for all pieces of data */
    handlesA = calloc( MT * KT, sizeof(starpu_data_handle_t) );
    handlesB = calloc( KT * NT, sizeof(starpu_data_handle_t) );
    handlesC = calloc( MT * NT, sizeof(starpu_data_handle_t) );

    for( n=0; n<NT; n++) {
        int nn = n == (NT-1) ? N - n * b : b;

        for( m=0; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

            hC = get_starpu_handle( 2, handlesC, C, m, n, b, MT );

            if ( transA == CblasNoTrans ) {
                if ( transB == CblasNoTrans ) {

                    /* A: CblasNoTrans / B: CblasNoTrans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, m, k, b, MT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, k, n, b, KT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
                else {

                    /* A: CblasNoTrans / B: Cblas[Conj]Trans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, m, k, b, MT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, n, k, b, NT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
            }
            else {
                if ( transB == CblasNoTrans ) {

                    /* A: Cblas[Conj]Trans / B: CblasNoTrans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, k, m, b, KT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, k, n, b, KT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
                else {

                    /* A: Cblas[Conj]Trans / B: Cblas[Conj]Trans */
                    for( k=0; k<KT; k++) {
                        int kk = k == (KT-1) ? K - k * b : b;
                        double lbeta = (k == 0) ? beta : 1.;

                        hA = get_starpu_handle( 0, handlesA, (double **)A, k, m, b, KT );
                        hB = get_starpu_handle( 1, handlesB, (double **)B, n, k, b, NT );

                        insert_dgemm( transA, transB, mm, nn, kk,
                                      alpha, hA, b, hB, b, lbeta, hC, b );
                    }
                }
            }
        }
    }

    /* Let's submit unregistration of all data handlers */
    unregister_starpu_handle( MT * KT, handlesA );
    unregister_starpu_handle( KT * NT, handlesB );
    unregister_starpu_handle( MT * NT, handlesC );

    /* Let's wait for the end of all the tasks */
#if defined(ENABLE_MPI)
    starpu_mpi_wait_for_all( MPI_COMM_WORLD );
    starpu_mpi_barrier( MPI_COMM_WORLD );
#else
    starpu_task_wait_for_all();
#endif

    free( handlesA );
    free( handlesB );
    free( handlesC );

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_tiled_fct_t valid_dgemm_tiled_starpu __attribute__ ((unused)) = dgemm_tiled_starpu;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_starpu;

/**
 * @brief Registration function
 */
void dgemm_tiled_starpu_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_starpu_init( void )
{
    fct_dgemm_tiled_starpu.mpi    = 1;
    fct_dgemm_tiled_starpu.tiled  = 1;
    fct_dgemm_tiled_starpu.starpu = 1;
    fct_dgemm_tiled_starpu.name   = "starpu";
    fct_dgemm_tiled_starpu.helper = "StarPU implementation of the dgemm";
    fct_dgemm_tiled_starpu.fctptr = dgemm_tiled_starpu;
    fct_dgemm_tiled_starpu.next   = NULL;

    register_fct( &fct_dgemm_tiled_starpu, ALGO_GEMM );
}
