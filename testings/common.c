/**
 *
 * @file common.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm_tiled variants.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "algonum.h"
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

void
print_usage( const char *name, int algo )
{
    printf( "Options:\n"
            "  -h --help  Show this help\n"
            "  -v --v=xxx Select the version to test among:\n" );

    print_fct( algo );

    printf( "\n"
            "  -M x        Set the M value\n"
            "  -N x        Set the N value\n"
            "  -K x        Set the K value\n"
            "  -b --nb=x   Set the block size b value\n"
            "  -A          Switch transA to CblasTrans\n"
            "  -B          Switch transB to CblasTrans\n"
            "  -i --iter=x Set the number of iteration\n"
            "  -c --check  Enable checking of the result\n" );
#if defined(ENABLE_MPI)
    printf( "\n"
            "  -P x        Set the 2D bloc-cyclic parameter P such that P x Q = nbnodes\n" );
#endif

    return;
}

#define GETOPT_STRING "hv:M:N:K:b:ABi:P:"
static struct option long_options[] =
{
    {"help",          no_argument,       0,      'h'},
    {"v",             required_argument, 0,      'v'},
    {"P",             required_argument, 0,      'P'},
    // Matrix parameters
    {"M",             required_argument, 0,      'M'},
    {"N",             required_argument, 0,      'N'},
    {"K",             required_argument, 0,      'K'},
    {"nb",            required_argument, 0,      'b'},
    // Check/prints
    {"transA",        no_argument,       0,      'A'},
    {"transB",        no_argument,       0,      'B'},
    {"check",         no_argument,       0,      'c'},
    // Performance tests
    {"iter",          required_argument, 0,      'i'},
    {0, 0, 0, 0}
};

void
algonum_init( int argc, char **argv, option_t *options, int algo )
{
    int mpirank = 0;
    int mpisize = 1;
    int opt;

    /* Set defaults */
    options->fct    = NULL;
    options->P      = 1;
    options->N      = 100;
    options->M      = -'N';
    options->K      = -'N';
    options->b      = 320;
    options->iter   = 1;
    options->transA = CblasNoTrans;
    options->transB = CblasNoTrans;
    options->check  = 0;

#if defined(ENABLE_MPI)
    {
        int provided;
        MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );

        MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
        MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

        if ( mpirank == 0 ) {
            switch( provided ) {
            case MPI_THREAD_MULTIPLE:
                fprintf( stderr, "MPI_THREAD_LEVEL support: MPI_THREAD_MULTIPLE\n" );
                break;
            case MPI_THREAD_SERIALIZED:
                fprintf( stderr, "MPI_THREAD_LEVEL support: MPI_THREAD_SERIALIZED\n" );
                break;
            case MPI_THREAD_FUNNELED:
                fprintf( stderr, "MPI_THREAD_LEVEL support: MPI_THREAD_FUNNELED\n" );
                break;
            case MPI_THREAD_SINGLE:
                fprintf( stderr, "MPI_THREAD_LEVEL support: MPI_THREAD_SINGLE\n" );
                break;
            default:
                fprintf( stderr, "Error initializing MPI\n" );
                exit(1);
            }
        }
    }
#endif

    options->mpirank = mpirank;
    options->mpisize = mpisize;

    while ((opt = getopt_long(argc, argv, GETOPT_STRING, long_options, NULL)) != -1)
    {
        switch(opt) {
        case 'h':
            print_usage( argv[0], algo );
            exit(0);

        case 'M':
            options->M = atoi( optarg );
            break;
        case 'N':
            options->N = atoi( optarg );
            break;
        case 'K':
            options->K = atoi( optarg );
            break;
        case 'b':
            options->b = atoi( optarg );
            break;
        case 'i':
            options->iter = atoi( optarg );
            break;
        case 'A':
            options->transA = CblasTrans;
            break;
        case 'B':
            options->transB = CblasTrans;
            break;
        case 'c':
          options->check = 1;
          break;

        case 'P':
            options->P = atoi( optarg );
            break;

        case 'v':
            options->fct = search_fct( optarg, algo );

            if ( (mpisize > 1) && (!options->fct->mpi) ) {
                fprintf( stderr, "ERROR: Version %s does not support MPI\n",
                         options->fct->name );
                exit(1);
            }
            break;

        case '?': /* error from getopt[_long] */
            exit(1);
            break;

        default:
            print_usage( argv[0], algo );
            exit(1);
        }
    }

    if ( options->M == -'N' ) {
        options->M = options->N;
    }
    if ( options->K == -'N' ) {
        options->K = options->N;
    }

    if ( mpisize % options->P != 0 ) {
        fprintf( stderr, "Parameter P (%d) must divide the number of nodes (%d)\n",
                 options->P, mpisize );
        exit(1);
    }
    options->Q = mpisize / options->P;

    if ( options->fct == NULL ) {
        fprintf( stderr, "Need to define a version to test\n" );
        print_usage( argv[0], algo );
        exit(1);
    }
    else {
        if ( mpirank == 0 ) {
            printf( "Test: %s (%s)\n", options->fct->helper, options->fct->name );
        }
    }

#if defined(ENABLE_STARPU)
    if ( options->fct->starpu ) {
        my_starpu_init();
    }
#endif

    if ( options->fct->cuda ) {
        myCublasInit();
    }

    return;
}

void
algonum_exit( option_t *options, int algo )
{

#if defined(ENABLE_STARPU)
    if ( options->fct->starpu ) {
        my_starpu_exit();
    }
#endif

    if ( options->fct->cuda ) {
        myCublasDestroy();
    }

#if defined(ENABLE_MPI)
    MPI_Finalize();
#endif

    return;
}
