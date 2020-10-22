/**
 *
 * @file common.c
 *
 * @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm_tiled variants.
 *
 * @version 0.1.0
 * @author Mathieu Faverge
 * @date 2019-12-01
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
            "  -M x       Set the M value\n"
            "  -N x       Set the N value\n"
            "  -K x       Set the K value\n"
            "  -b --nb=x  Set the block size b value\n"
            "  -A         Switch transA to CblasTrans\n"
            "  -B         Switch transB to CblasTrans\n" );

    return;
}

#define GETOPT_STRING "hv:M:N:K:AB"
static struct option long_options[] =
{
    {"help",          no_argument,       0,      'h'},
    {"v",             required_argument, 0,      'v'},
    // Matrix parameters
    {"M",             required_argument, 0,      'M'},
    {"N",             required_argument, 0,      'N'},
    {"K",             required_argument, 0,      'K'},
    {"nb",            required_argument, 0,      'b'},
    // Check/prints
    {"transA",        no_argument,       0,      'A'},
    {"transB",        no_argument,       0,      'B'},
    // Performance tests
    {"iter",          no_argument,       0,      'i'},
    {0, 0, 0, 0}
};

void
parse_opts( int argc, char **argv, option_t *options, int algo )
{
    int opt;

    /* Set defaults */
    options->fct    = NULL;
    options->N      = 100;
    options->M      = -'N';
    options->K      = -'N';
    options->b      = 320;
    options->iter   = 1;
    options->transA = CblasNoTrans;
    options->transB = CblasNoTrans;

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

        case 'v':
            options->fct = search_fct( optarg, algo );
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
}
