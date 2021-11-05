/**
 *
 * @file core_dplrnt.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon core_dplrnt CPU kernel
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @author Emmanuel Agullo
 * @date 2021-09-21
 *
 * Functiont to generate random matrices.
 */
#include "algonum_int.h"

/*
  Rnd64seed is a global variable but it doesn't spoil thread safety. All matrix
  generating threads only read Rnd64seed. It is safe to set Rnd64seed before
  and after any calls to create_tile(). The only problem can be caused if
  Rnd64seed is changed during the matrix generation time.
*/
//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static unsigned long long int
Rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
        if (n & 1)
            ran = a_k * ran + c_k;
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

/**
 * @brief Function to generate a random matrix.
 *
 * It can be called to generate an entire matrix at once with:
 *      m = bigM, m0 = n0 = 0
 * or it can be used to genrate in parallel portion of a global matrix as done
 * in dplrnt_tiled().
 *
 * @param[in] bump
 *       Scalar value to eventually add to the diagonal of the entire generated
 *       matrix in order to make it diagonal dominant.
 *
 * @param[in] m
 *       The number of rows of the matrix A.
 *
 * @param[in] n
 *       The number of columns of the matrix A.
 *
 * @param[in,out] A
 *       The matrix A of size lda-by-n to initialize.
 *
 * @param[in] lda
 *       The leading dimension of the matrix A. lda >= max( m, 1 ).
 *
 * @param[in] bigM
 *       The number of rows of the entire matrix that contains the submatrix A.
 *
 * @param[in] m0
 *       The row index of the first element of the submatrix A in the entire generated matrix.
 *
 * @param[in] n0
 *       The column index of the first element of the submatrix A in the entire generated matrix.
 *
 * @param[in] seed
 *       The seed used by random generator to generate the full matrix. Must be
 *       the same value for all the call with respect to the same generated
 *       matrix.
 */
void CORE_dplrnt( double bump, int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed )
{
    double *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)bigM;

    for (j=0; j<n; ++j ) {
        ran = Rnd64_jump( jump, seed );
        for (i = 0; i < m; ++i) {
            *tmp = 0.5f - ran * RndF_Mul;
            ran  = Rnd64_A * ran + Rnd64_C;

            if ( (i + m0) == (j + n0) ) {
                *tmp += bump;
            }

            tmp++;
        }
        tmp  += lda-i;
        jump += bigM;
    }
}

/**
 * @brief Function to generate a random matrix in the tiled format.
 *
 * @param[in] bump
 *       Scalar value to eventually add to the diagonal of the entire generated
 *       matrix in order to make it diagonal dominant.
 *
 * @param[in] M
 *       The number of rows of the matrix A.
 *
 * @param[in] N
 *       The number of columns of the matrix A.
 *
 * @param[in] b
 *       The size b-by-b of each tile in the matrix A.
 *
 * @param[in,out] A
 *       An array of pointer of size ((M + b - 1) / b) -by- ((N + b - 1) / b)
 *       to all the tiles of the matrix A of size M-by-N to initialize.
 *
 * @param[in] seed
 *       The seed used by random generator to generate the full matrix. Must be
 *       the same value for all the call with respect to the same generated
 *       matrix.
 */
void dplrnt_tiled( double bump, int M, int N, int b,
                   double **A, unsigned long long int seed )
{
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int m, n;
    int mm, nn;

#pragma omp parallel for collapse(2) private (m, n, mm, nn)
    for( m=0; m<MT; m++ ) {
        for( n=0; n<NT; n++ ) {

            mm = m == (MT-1) ? M - m * b : b;
            nn = n == (NT-1) ? N - n * b : b;

            if ( A[ MT * n + m ] == NULL ) {
                continue;
            }

            CORE_dplrnt( bump, mm, nn, A[ MT * n + m ], b,
                         M, m * b, n * b, seed );
        }
    }
}
