/**
 *
 * @file dgemm_tiled_mpi.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the OpenMP version
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"

int dgemm_tiled_mpi( CBLAS_LAYOUT layout,
                     CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int M, int N, int K, int b,
                     double alpha, const double **A,
                                   const double **B,
                     double beta,        double **C )
{
#if !defined(ENABLE_MPI)
    return ALGONUM_NOT_IMPLEMENTED;
#else
    MPI_Status    status;
    const double *Aptr, *Bptr;
    double        wsA[b*b], wsB[b*b];
    double        lbeta;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );

    int m, n, k, mm, nn, kk;
    int ownerA, ownerB, ownerC;

    if ( transA == CblasNoTrans ) {
        if ( transB == CblasNoTrans ) {
            for( m=0; m<MT; m++ ) {
                mm = m == (MT-1) ? M - m * b : b;

                for( n=0; n<NT; n++ ) {
                    nn = n == (NT-1) ? N - n * b : b;
                    ownerC = get_rank_of( m, n );

                    if ( global_options.mpirank == ownerC ) {
                        for( k=0; k<KT; k++ ) {
                            kk = k == (KT-1) ? K - k * b : b;
                            lbeta = (k == 0) ? beta : 1.;

                            ownerA = get_rank_of( m, k );
                            ownerB = get_rank_of( k, n );

                            if ( global_options.mpirank == ownerA ) {
                                Aptr = A[MT * k + m];
                            }
                            else {
                                MPI_Recv( wsA, b*kk, MPI_DOUBLE,
                                          ownerA, MT * k + m, MPI_COMM_WORLD, &status );
                                Aptr = wsA;
                            }

                            if ( global_options.mpirank == ownerB ) {
                                Bptr = B[KT * n + k];
                            }
                            else {
                                MPI_Recv( wsB, b*nn, MPI_DOUBLE,
                                          ownerB, KT * n + k, MPI_COMM_WORLD, &status );
                                Bptr = wsB;
                            }

                            dgemm_seq( CblasColMajor, transA, transB,
                                       mm, nn, kk,
                                       alpha, Aptr, b, Bptr, b,
                                       lbeta, C[ MT * n + m ], b );
                        }
                    }
                    else {
                        for( k=0; k<KT; k++ ) {
                            kk = k == (KT-1) ? K - k * b : b;
                            lbeta = (k == 0) ? beta : 1.;

                            ownerA = get_rank_of( m, k );
                            ownerB = get_rank_of( k, n );

                            if ( global_options.mpirank == ownerA ) {
                                MPI_Send( A[MT * k + m], b * kk, MPI_DOUBLE,
                                          ownerC, MT * k + m, MPI_COMM_WORLD );
                            }

                            if ( global_options.mpirank == ownerB ) {
                                MPI_Send( B[KT * n + k], b * nn, MPI_DOUBLE,
                                          ownerC, KT * n + k, MPI_COMM_WORLD );
                            }
                        }
                    }
                }
            }
        }
        else {
            return ALGONUM_NOT_IMPLEMENTED;
        }
    }
    else {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    return ALGONUM_SUCCESS;
#endif
}

/* To make sure we use the right prototype */
static dgemm_tiled_fct_t valid_dgemm_tiled_mpi __attribute__ ((unused)) = dgemm_tiled_mpi;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_mpi;

/**
 * @brief Registration function
 */
void dgemm_tiled_mpi_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_mpi_init( void )
{
    fct_dgemm_tiled_mpi.mpi    = 1;
    fct_dgemm_tiled_mpi.tiled  = 1;
    fct_dgemm_tiled_mpi.starpu = 0;
    fct_dgemm_tiled_mpi.name   = "mpi";
    fct_dgemm_tiled_mpi.helper = "MPI tiled implementation of the dgemm";
    fct_dgemm_tiled_mpi.fctptr = dgemm_tiled_mpi;
    fct_dgemm_tiled_mpi.next   = NULL;

    register_fct( &fct_dgemm_tiled_mpi, ALGO_GEMM );
}
