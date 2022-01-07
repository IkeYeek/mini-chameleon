/**
 *
 * @file check_ddot.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Function to numerically validate a gemm function
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-23
 *
 */
#include "algonum_int.h"

int
check_ddot( const int N, const double *X, const int incX, const double *Y, const int incY, double *res_ref, const double *res )
{
    int info_solution;
    double Anorm, Bnorm, res_refnorm, Rnorm, result;
    double eps = LAPACKE_dlamch_work('e');
    double *work = malloc( max(1, max(1, N) ) * sizeof(double) );

    /* Calculates the norms of X and Y according to the transposition */
    Anorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'I', 1, N, X, incX, work );
    Bnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'O', N, 1, Y, incY, work );

    /* Compute the original norm */
    res_refnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', 1, 1, res_ref, 1, NULL );

    /* Makes the multiplication with the validated function */
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, 1, 1, N,
                 1.0, X, incX, Y, incY, 0.0, res_ref, 1 );
    cblas_daxpy( 1 * 1, -1., res, 1, res_ref, 1 );

    /* Compute the norm of the residual */
    Rnorm = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'M', 1, 1, res_ref, 1, NULL );

    result = ((fabs(1.0) * max(Anorm, Bnorm) + fabs(0.0) * res_refnorm) * N * eps);
    if ( result > 0. ) {
        result = Rnorm / result;
    }

    /* Verify if the result is inside the threshold threshold */
    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        if (result > 10.0){
          fprintf(stdout, "\nresult : %le\n", result);
        }
        else{
          fprintf(stdout, "\nERREUR BIZAAAAAAAAAAAAAAAAAAAAAAAAAARE\n");
        }
        info_solution = ALGONUM_FAIL;
    }
    else {
        info_solution = ALGONUM_SUCCESS;
    }

    free(work);

    return info_solution;
}
