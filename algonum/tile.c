/**
 *
 * @file tile.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Tile to/from lapack conversion subroutine
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include <assert.h>
#include "algonum_int.h"

int
get_rank_of( int M, int N ) {
    int p = M % global_options.P;
    int q = N % global_options.Q;

    return p * global_options.Q + q;
}

double **
lapack2tile( int M, int N, int b,
             const double *Alapack, int lda )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    /* Allocate the array of pointers to the tiles */
    double **Atile = malloc( MT * NT * sizeof(double*) );

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            double *tile;
            int mm, nn;

            if ( get_rank_of( m, n ) != global_options.mpirank ) {
                Atile[ MT * n + m ] = NULL;
                continue;
            }

            tile = malloc( b * b * sizeof(double) );
            mm   = m == (MT-1) ? M - m * b : b;
            nn   = n == (NT-1) ? N - n * b : b;

            /* Let's use LAPACKE to ease the copy */
            if ( Alapack != NULL ) {
                LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                     Alapack + lda * b * n + b * m, lda,
                                     tile, b );
            }
            Atile[ MT * n + m ] = tile;
        }
    }

    return Atile;
}

void
tile2lapack( int M, int N, int b,
             const double **Atile,
             double *Alapack, int lda )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    assert( lda >= M );

#if defined(ENABLE_MPI)
    MPI_Datatype *type;
    MPI_Datatype tiletypeA, tiletypeB, tiletypeC, tiletypeD;

    if ( global_options.mpirank != 0 ) {
        lda = b;
    }

    MPI_Type_vector( b,              b,              lda, MPI_DOUBLE, &tiletypeA );
    MPI_Type_vector( N - (NT-1) * b, b,              lda, MPI_DOUBLE, &tiletypeB );
    MPI_Type_vector( b,              M - (MT-1) * b, lda, MPI_DOUBLE, &tiletypeC );
    MPI_Type_vector( N - (NT-1) * b, M - (MT-1) * b, lda, MPI_DOUBLE, &tiletypeD );

    MPI_Type_commit( &tiletypeA );
    MPI_Type_commit( &tiletypeB );
    MPI_Type_commit( &tiletypeC );
    MPI_Type_commit( &tiletypeD );

    type = &tiletypeA;
#endif

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            const double *tile;
            int mm, nn;
            int owner = get_rank_of( m, n );

            if ( (global_options.mpirank != owner) &&
                 (global_options.mpirank != 0    ) )
            {
                assert( Atile[ MT * n + m ] == NULL );
                continue;
            }

            tile = Atile[ MT * n + m ];
            mm   = m == (MT-1) ? M - m * b : b;
            nn   = n == (NT-1) ? N - n * b : b;

            if ( owner == 0 ) {
                /* Let's use LAPACKE to ease the copy */
                LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                     tile, b,
                                     Alapack + lda * b * n + b * m, lda );
            }
#if defined(ENABLE_MPI)
            else {
                if ( m == (MT-1) ) {
                    if ( n == (NT-1) ) {
                        type = &tiletypeD;
                    }
                    else {
                        type = &tiletypeC;
                    }
                }
                else {
                    if ( n == (NT-1) ) {
                        type = &tiletypeB;
                    }
                    else {
                        type = &tiletypeA;
                    }
                }
                if ( global_options.mpirank == owner ) {
                    MPI_Send( tile, 1, *type,
                              0, MT * n + m, MPI_COMM_WORLD );
                }

                if ( global_options.mpirank == 0 ) {
                    MPI_Status status;
                    MPI_Recv( Alapack + lda * b * n + b * m, 1, *type,
                              owner, MT * n + m, MPI_COMM_WORLD, &status );
                }
            }
#endif
        }
    }


#if defined(ENABLE_MPI)
    MPI_Type_free( &tiletypeA );
    MPI_Type_free( &tiletypeB );
    MPI_Type_free( &tiletypeC );
    MPI_Type_free( &tiletypeD );
#endif
}

void
tileFree( int M, int N, int b, double **A )
{
    /* Let's compute the total number of tiles with a *ceil* */
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;

    /* Now, let's copy the tile one by one, in column major order */
    for( n=0; n<NT; n++) {
        for( m=0; m<MT; m++) {
            if ( A[ MT * n + m ] ) {
                free( A[ MT * n + m ] );
            }
        }
    }
    free( A );
}
