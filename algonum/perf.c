/**
 *
 * @file perf.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Performance measurement subroutines
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @author Emmanuel Agullo
 * @date 2021-09-30
 *
 */
#include "algonum.h"
#include "perf.h"
#include <stdio.h>
#include <stdlib.h>
#if defined(ENABLE_MPI)
#include <mpi.h>
#endif

/* TODO : move that into common.c when common.c will be moved to algonum lib */
option_t global_options = {
    .fct     = NULL,
    .P       = 1,
    .Q       = 1,
    .N       = 100,
    .M       = -'N',
    .K       = -'N',
    .b       = 320,
    .iter    = 1,
    .mpirank = 0,
    .mpisize = 1,
    .transA  = CblasNoTrans,
    .transB  = CblasNoTrans,
    .check   = 0
};

void perf(perf_t *p)
{
#if defined(ENABLE_MPI)
    MPI_Barrier( MPI_COMM_WORLD );
#endif
    gettimeofday( p, NULL );
}

void perf_diff(const perf_t *begin, perf_t *end)
{
    end->tv_sec  = end->tv_sec  - begin->tv_sec;
    end->tv_usec = end->tv_usec - begin->tv_usec;
    if (end->tv_usec < 0)
    {
        (end->tv_sec)--;
        end->tv_usec += 1.e6;
    }

    if ( (end->tv_sec  == 0) &&
         (end->tv_usec == 0) )
    {
        end->tv_usec = 1;
    }
}

void perf_printh(const perf_t *p)
{
    long m = p->tv_sec / 60;
    long s = p->tv_sec - m * 60;
    long ms = p->tv_usec / 1.e3;
    long micros = p->tv_usec - ms * 1.e3;

    //  printf("%ld sec %ld usec\n", p->tv_sec, p->tv_usec);
    printf("%ld:%ld:%ld:%ld\n", m, s, ms, micros);
}

void perf_printmicro(const perf_t *p)
{
    printf("%le\n", p->tv_usec + (p->tv_sec * 1.e6));
}

double perf_mflops(const perf_t *p, const double nb_op)
{
    return nb_op / (p->tv_sec * 1.e6 + p->tv_usec);
}

double perf_gflops(const perf_t *p, const double nb_op)
{
    return (nb_op / (p->tv_sec * 1.e6 + p->tv_usec)) * 1.e-3;
}

#if defined(ENABLE_CUDA)

cublasHandle_t my_cublas_handle;

void myCublasInit()
{
    cublasCreate(&my_cublas_handle);
}

void myCublasDestroy()
{
    cublasDestroy(my_cublas_handle);
}

#else
void myCublasInit()
{
}

void myCublasDestroy()
{
}
#endif
