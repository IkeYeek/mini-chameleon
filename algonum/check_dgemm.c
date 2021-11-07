/**
 *
 * @file check_dgemm.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Function to numerically validate a gemm function
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum_int.h"

/**
 *  @brief Function to check one single case of dgemm.
 *
 *  Performs one of the following check
 *
 *  Compute Cref with validated library such that
 *    \f[ Cref = \alpha [op( A )\times op( B )] + \beta Cref, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m-by-k matrix, op( B ) a k-by-n matrix and C an m-by-n matrix.
 *
 * And finally valide the operation by checking that: || Cref - C ||  < eps
 *
 * @param[in] transA
 *          - CblasNoTrans:   A is not transposed,
 *          - CblasTrans:     A is transposed,
 *
 * @param[in] transB
 *          - CblasNoTrans:   B is not transposed,
 *          - CblasTrans:     B is transposed,
 *
 * @param[in] M
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] N
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] K
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          An lda-by-ka matrix, where ka is k when transa = ChamNoTrans,
 *          and is m otherwise.
 *
 * @param[in] LDA
 *          The leading dimension of the array A.
 *          When transa = ChamNoTrans, lda >= max(1,m),
 *          otherwise, lda >= max(1,k).
 *
 * @param[in] B
 *          An ldb-by-kb matrix, where kb is n when transb = ChamNoTrans,
 *          and is k otherwise.
 *
 * @param[in] LDB
 *          The leading dimension of the array B.
 *          When transb = ChamNoTrans, ldb >= max(1,k),
 *          otherwise, ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          An ldc-by-n matrix. On exit, the array is overwritten by the m-by-n
 *          matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * @param[in] LDC
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 * @retval 0, on success
 * @retval 1, on failure
 *
 */
int
check_dgemm( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
             int M, int N, int K,
             double alpha, const double *A, int lda, const double *B, int ldb,
             double beta, double *Cref, const double *C, int ldc )
{
    int info_solution;
    double Anorm, Bnorm, Crefnorm, Rnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    double *work = malloc( max(M, max(N, K) ) * sizeof(double) );

    /* Calculates the norms of A and B according to the transposition */
    if ( transA == CblasNoTrans ) {
        Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'I', M, K, A, lda, work );
    } else {
        Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', K, M, A, lda, work );
    }
    if ( transB == CblasNoTrans ) {
        Bnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', K, N, B, ldb, work );
    } else {
        Bnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'I', N, K, B, ldb, work );
    }

    /* Compute the original norm */
    Crefnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, Cref, ldc, NULL );

    /* Makes the multiplication with the validated function */
    cblas_dgemm( CblasColMajor, transA, transB, M, N, K,
                 alpha, A, lda, B, ldb, beta, Cref, ldc );
    cblas_daxpy( ldc * N, -1., C, 1, Cref, 1 );

    /* Compute the norm of the residual */
    Rnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', M, N, Cref, ldc, NULL );

    result = ((fabs(alpha) * max(Anorm, Bnorm) + fabs(beta) * Crefnorm) * K * eps);
    if ( result > 0. ) {
        result = Rnorm / result;
    }

    /* Verify if the result is inside the threshold threshold */
    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = ALGONUM_FAIL;
    }
    else {
        info_solution = ALGONUM_SUCCESS;
    }

    free(work);

    return info_solution;
}
