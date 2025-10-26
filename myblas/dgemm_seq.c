/**
 *
 * @file dgemm_seq.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Sequential version of the Matrix-Matrix multiply operation
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "algonum.h"
#include "myblas.h"
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

// Exemple of ways to add additionnal parameters to your kernel
// See the registration function to change its value
static int dgemm_seq_block_size = -1;

int dgemm_scalaire(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                   CBLAS_TRANSPOSE transB, const int M, const int N,
                   const int K, const double alpha, const double *A,
                   const int lda, const double *B, const int ldb,
                   const double beta, double *C, const int ldc) {
  int m, n, k;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {
      for (n = 0; n < N; n++) {
        if (beta != 1.) {
          for (m = 0; m < M; m++) {
            C[ldc * n + m] = beta * C[ldc * n + m];
          }
        }
        for (k = 0; k < K; k++) {
          for (m = 0; m < M; m++) {
            C[ldc * n + m] += alpha * A[lda * k + m] * B[ldb * n + k];
          }
        }
      }
    } else {
      for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
          if (beta != 1.) {
            C[ldc * n + m] = beta * C[ldc * n + m];
          }
          for (k = 0; k < K; k++) {
            C[ldc * n + m] += alpha * A[lda * k + m] * B[ldb * k + n];
          }
        }
      }
    }
  } else {
    if (transB == CblasNoTrans) {
      for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
          if (beta != 1.) {
            C[ldc * n + m] = beta * C[ldc * n + m];
          }
          for (k = 0; k < K; k++) {
            C[ldc * n + m] += alpha * A[lda * m + k] * B[ldb * n + k];
          }
        }
      }
    } else {
      for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
          if (beta != 1.) {
            C[ldc * n + m] = beta * C[ldc * n + m];
          }
          for (k = 0; k < K; k++) {
            C[ldc * n + m] += alpha * A[lda * m + k] * B[ldb * k + n];
          }
        }
      }
    }
  }

  return ALGONUM_SUCCESS;
}

#define VEC_BLOCK_SIZE 4
static inline void scale_vec(const int M, const int N, const int rem,
                             const double beta, double *C, const int ldc) {
  __m256d C_mn_subvec;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M - rem; m += VEC_BLOCK_SIZE) {
      C_mn_subvec = _mm256_loadu_pd(&C[ldc * n + m]);
      C_mn_subvec = _mm256_mul_pd(C_mn_subvec, _mm256_set1_pd(beta));
      _mm256_storeu_pd(&C[ldc * n + m], C_mn_subvec);
    }
    for (int m = M - rem; m < M; m++) {
      C[ldc * n + m] *= beta;
    }
  }
}

int dgemm_avx2(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
               const double alpha, const double *A, const int lda,
               const double *B, const int ldb, const double beta, double *C,
               const int ldc) {

  // TODO: is it possible to know when we could use store/load instead of
  // storeu/loadu without introducing conditional branchment in the loops?
  // TODO: is treating sequentially first and not least an issue ?
  // TODO: AVX512???
  // TODO: compare with and without fmadd
  int m, n, k;
  // the remainder of M % VEC_BLOCK_SIZE that we'll treat sequentially
  int rem = M % VEC_BLOCK_SIZE;
  __m256d C_mn_subvec, B_nk_vec, A_mk_subvec;
  for (n = 0; n < N; n++) {
    // doing the beta*C computations
    if (beta != 1.0) {
      for (int m = 0; m < rem; m++) {
        C[ldc * n + m] *= beta;
      }
      for (m = rem; m < M; m += VEC_BLOCK_SIZE) {
        C_mn_subvec = _mm256_loadu_pd(&C[ldc * n + m]);
        C_mn_subvec = _mm256_mul_pd(C_mn_subvec, _mm256_set1_pd(beta));
        _mm256_storeu_pd(&C[ldc * n + m], C_mn_subvec);
      }
    }

    // actual alpha*A*B+beta*C
    for (k = 0; k < K; k++) {
      B_nk_vec = _mm256_set1_pd(B[ldb * n + k]);
      for (m = 0; m < rem; m++) {
        C[ldc * n + m] += alpha * A[lda * k + m] * B[ldb * n + k];
      }
      for (m = rem; m < M; m += VEC_BLOCK_SIZE) {
        A_mk_subvec = _mm256_loadu_pd(&A[lda * k + m]);
        C_mn_subvec = _mm256_loadu_pd(&C[ldc * n + m]);

        A_mk_subvec = _mm256_mul_pd(A_mk_subvec, _mm256_set1_pd(alpha));
        C_mn_subvec = _mm256_fmadd_pd(A_mk_subvec, B_nk_vec, C_mn_subvec);
        // C_mn_subvec =
        //     _mm256_add_pd(C_mn_subvec, _mm256_mul_pd(A_mk_subvec, B_nk_vec));

        _mm256_storeu_pd(&C[m + ldc * n], C_mn_subvec);
      }
    }
  }

  return ALGONUM_SUCCESS;
}

int dgemm_avx2_microkernel_4x4(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                               CBLAS_TRANSPOSE transB, const int M, const int N,
                               const int K, const double alpha, const double *A,
                               const int lda, const double *B, const int ldb,
                               const double beta, double *C, const int ldc) {
  int k;
  __m256d C_vec_0, C_vec_1, C_vec_2, C_vec_3;

  C_vec_0 = _mm256_loadu_pd(C + 0 * 0);
  C_vec_1 = _mm256_loadu_pd(C + ldc);
  C_vec_2 = _mm256_loadu_pd(C + ldc * 2);
  C_vec_3 = _mm256_loadu_pd(C + ldc * 3);

  for (k = 0; k < K / 4; k += 1) {
    int k4 = k * 4;
    __m256d A_vec_1 = _mm256_loadu_pd(A + (k4 * lda));
    __m256d A_vec_2 = _mm256_loadu_pd(A + (k4 * lda) + lda);
    __m256d A_vec_3 = _mm256_loadu_pd(A + (k4 * lda) + 2 * lda);
    __m256d A_vec_4 = _mm256_loadu_pd(A + (k4 * lda) + 3 * lda);

    {
      __m256d B_b0 = _mm256_set1_pd(alpha * B[0 * ldb + k4 + 0]);
      __m256d B_b1 = _mm256_set1_pd(alpha * B[0 * ldb + k4 + 1]);
      __m256d B_b2 = _mm256_set1_pd(alpha * B[0 * ldb + k4 + 2]);
      __m256d B_b3 = _mm256_set1_pd(alpha * B[0 * ldb + k4 + 3]);
      C_vec_0 = _mm256_fmadd_pd(A_vec_1, B_b0, C_vec_0);
      C_vec_0 = _mm256_fmadd_pd(A_vec_2, B_b1, C_vec_0);
      C_vec_0 = _mm256_fmadd_pd(A_vec_3, B_b2, C_vec_0);
      C_vec_0 = _mm256_fmadd_pd(A_vec_4, B_b3, C_vec_0);
    }

    {
      __m256d B_b0 = _mm256_set1_pd(alpha * B[1 * ldb + k4 + 0]);
      __m256d B_b1 = _mm256_set1_pd(alpha * B[1 * ldb + k4 + 1]);
      __m256d B_b2 = _mm256_set1_pd(alpha * B[1 * ldb + k4 + 2]);
      __m256d B_b3 = _mm256_set1_pd(alpha * B[1 * ldb + k4 + 3]);
      C_vec_1 = _mm256_fmadd_pd(A_vec_1, B_b0, C_vec_1);
      C_vec_1 = _mm256_fmadd_pd(A_vec_2, B_b1, C_vec_1);
      C_vec_1 = _mm256_fmadd_pd(A_vec_3, B_b2, C_vec_1);
      C_vec_1 = _mm256_fmadd_pd(A_vec_4, B_b3, C_vec_1);
    }

    {
      __m256d B_b0 = _mm256_set1_pd(alpha * B[2 * ldb + k4 + 0]);
      __m256d B_b1 = _mm256_set1_pd(alpha * B[2 * ldb + k4 + 1]);
      __m256d B_b2 = _mm256_set1_pd(alpha * B[2 * ldb + k4 + 2]);
      __m256d B_b3 = _mm256_set1_pd(alpha * B[2 * ldb + k4 + 3]);
      C_vec_2 = _mm256_fmadd_pd(A_vec_1, B_b0, C_vec_2);
      C_vec_2 = _mm256_fmadd_pd(A_vec_2, B_b1, C_vec_2);
      C_vec_2 = _mm256_fmadd_pd(A_vec_3, B_b2, C_vec_2);
      C_vec_2 = _mm256_fmadd_pd(A_vec_4, B_b3, C_vec_2);
    }

    {
      __m256d B_b0 = _mm256_set1_pd(alpha * B[3 * ldb + k4 + 0]);
      __m256d B_b1 = _mm256_set1_pd(alpha * B[3 * ldb + k4 + 1]);
      __m256d B_b2 = _mm256_set1_pd(alpha * B[3 * ldb + k4 + 2]);
      __m256d B_b3 = _mm256_set1_pd(alpha * B[3 * ldb + k4 + 3]);
      C_vec_3 = _mm256_fmadd_pd(A_vec_1, B_b0, C_vec_3);
      C_vec_3 = _mm256_fmadd_pd(A_vec_2, B_b1, C_vec_3);
      C_vec_3 = _mm256_fmadd_pd(A_vec_3, B_b2, C_vec_3);
      C_vec_3 = _mm256_fmadd_pd(A_vec_4, B_b3, C_vec_3);
    }
  }

  _mm256_storeu_pd(C + 0 * 0, C_vec_0);
  _mm256_storeu_pd(C + ldc, C_vec_1);
  _mm256_storeu_pd(C + ldc * 2, C_vec_2);
  _mm256_storeu_pd(C + ldc * 3, C_vec_3);
  return ALGONUM_SUCCESS;
}

int dgemm_avx2_bloc(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                    CBLAS_TRANSPOSE transB, const int M, const int N,
                    const int K, const double alpha, const double *A,
                    const int lda, const double *B, const int ldb,
                    const double beta, double *C, const int ldc) {
  const int MB = 4;
  const int NB = 4;
  int m0, n0, k;

  if (beta != 1.0) {
    scale_vec(M, N, M % MB, beta, C, ldc);
  }

  // we compute the parts that can be handled by microkernel
  const int M4 = (M / MB) * MB;
  const int N4 = (N / NB) * NB;
  const int K4 = (K / 4) * 4;

  if (K4 == 0 || M4 == 0 || N4 == 0) {
    return dgemm_avx2(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                      1.0, C, ldc);
  }

  for (n0 = 0; n0 < N4; n0 += NB) {
    const double *B_block = B + n0 * ldb;
    double *C_col = C + n0 * ldc;

    for (m0 = 0; m0 < M4; m0 += MB) {
      const double *A_block = A + m0;
      double *C_block = C_col + m0;
      dgemm_avx2_microkernel_4x4(layout, transA, transB, MB, NB, K4, alpha,
                                 A_block, lda, B_block, ldb, 1.0, C_block, ldc);

      for (k = K4; k < K; k++) {
        for (int nn = 0; nn < NB; nn++) {
          const double b = alpha * B_block[ldb * nn + k];
          const double *A_col = A_block + k * lda;
          __m256d A_vec = _mm256_loadu_pd(A_col);
          __m256d C_vec = _mm256_loadu_pd(C_block + nn * ldc);
          __m256d B_broadcase = _mm256_set1_pd(b);
          C_vec = _mm256_fmadd_pd(A_vec, B_broadcase, C_vec);
          _mm256_storeu_pd(C_block + nn * ldc, C_vec);
        }
      }
    }
  }

  if (N4 < N && M4 > 0) {
    const int N_right = N - N4;
    dgemm_avx2(layout, transA, transB, M4, N_right, K, alpha, A, lda,
               B + N4 * ldb, ldb, 1.0, C + N4 * ldc, ldc);
  }

  if (M4 < M) {
    const int M_bottom = M - M4;
    dgemm_avx2(layout, transA, transB, M_bottom, N, K, alpha, A + M4, lda, B,
               ldb, 1.0, C + M4, ldc);
  }

  return ALGONUM_SUCCESS;
}

int dgemm_custom(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                 CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta, double *C,
                 const int ldc) {

  int m, n, k;

  if (beta != 1.0) {
    for (n = 0; n < N; n++) {
      for (m = 0; m < M; m++) {
        C[ldc * n + m] = beta * C[ldc * n + m];
      }
    }
  }
  for (n = 0; n < N; n++) {
    for (k = 0; k < K; k++) {
      for (m = 0; m < M; m++) {
        C[ldc * n + m] += alpha * A[lda * k + m] * B[ldb * n + k];
      }
    }
  }

  return ALGONUM_SUCCESS;
}

void scale_block(const int M, const int N, const double beta, double *C,
                 const int ldc) {
  for (int bn = 0; bn < dgemm_seq_block_size; bn++) {
    for (int bm = 0; bm < dgemm_seq_block_size; bm++) {
      C[ldc * bn + bm] = beta * C[ldc * bn + bm];
    }
  }
}

int dgemm_bloc(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
               const double alpha, const double *A, const int lda,
               const double *B, const int ldb, const double beta, double *C,
               const int ldc) {
  int m, n, k;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {
      for (n = 0; n < N; n += dgemm_seq_block_size) {
        if (beta != 1.) {
          for (m = 0; m < M; m += dgemm_seq_block_size) {
            scale_block(dgemm_seq_block_size, dgemm_seq_block_size, beta,
                        C + ldc * n + m, ldc);
          }
        }
        for (k = 0; k < K; k += dgemm_seq_block_size) {
          for (m = 0; m < M; m += dgemm_seq_block_size) {
            // C[ldc * n + m] += alpha * A[lda * k + m] * B[ldb * n + k];
            dgemm_scalaire(layout, transA, transB, dgemm_seq_block_size,
                           dgemm_seq_block_size, dgemm_seq_block_size, alpha,
                           A + lda * k + m, lda, B + ldb * n + k, ldb, 1.0,
                           C + ldc * n + m, ldc);
          }
        }
      }
    } else {
      for (m = 0; m < M; m += dgemm_seq_block_size) {
        for (n = 0; n < N; n += dgemm_seq_block_size) {
          if (beta != 1.) {
            scale_block(dgemm_seq_block_size, dgemm_seq_block_size, beta,
                        C + ldc * n + m, ldc);
          }
          for (k = 0; k < K; k += dgemm_seq_block_size) {
            dgemm_scalaire(layout, transA, transB, dgemm_seq_block_size,
                           dgemm_seq_block_size, dgemm_seq_block_size, alpha,
                           A + lda * k + m, lda, B + ldb * k + n, ldb, 1.0,
                           C + ldc * n + m, ldc);
          }
        }
      }
    }
  } else {
    if (transB == CblasNoTrans) {
      for (m = 0; m < M; m += dgemm_seq_block_size) {
        for (n = 0; n < N; n += dgemm_seq_block_size) {
          // if beta != 0, scale the C block before accumulating into it
          if (beta != 1.) {
            scale_block(dgemm_seq_block_size, dgemm_seq_block_size, beta,
                        C + ldc * n + m, ldc);
          }
          for (k = 0; k < K; k += dgemm_seq_block_size) {
            dgemm_scalaire(layout, transA, transB, dgemm_seq_block_size,
                           dgemm_seq_block_size, dgemm_seq_block_size, alpha,
                           A + lda * m + k, lda, B + ldb * n + k, ldb, 1.0,
                           C + ldc * n + m, ldc);
          }
        }
      }
    } else {
      for (m = 0; m < M; m += dgemm_seq_block_size) {
        for (n = 0; n < N; n += dgemm_seq_block_size) {
          // if beta != 0, scale the C block before accumulating into it
          if (beta != 1.) {
            scale_block(dgemm_seq_block_size, dgemm_seq_block_size, beta,
                        C + ldc * n + m, ldc);
          }
          for (k = 0; k < K; k += dgemm_seq_block_size) {
            dgemm_scalaire(layout, transA, transB, dgemm_seq_block_size,
                           dgemm_seq_block_size, dgemm_seq_block_size, alpha,
                           A + lda * m + k, lda, B + ldb * k + n, ldb, 1.0,
                           C + ldc * n + m, ldc);
          }
        }
      }
    }
  }

  return ALGONUM_SUCCESS;
}

int dgemm_seq(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
              CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
              const double alpha, const double *A, const int lda,
              const double *B, const int ldb, const double beta, double *C,
              const int ldc) {
  if (dgemm_seq_block_size != 1) {
    dgemm_bloc(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
  } else {
    dgemm_scalaire(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                   C, ldc);
  }

  return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_seq __attribute__((unused)) = dgemm_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_seq;

/**
 * @brief Registration function
 */
void dgemm_seq_init(void) __attribute__((constructor));
void dgemm_seq_init(void) {
  int ver_idx = 0;

  char *versions[] = {"scalaire", "custom", "bloc", "avx2", "avx2_bloc"};
  void *fcptrs[] = {dgemm_scalaire, dgemm_custom, dgemm_bloc, dgemm_avx2,
                    dgemm_avx2_bloc};
  char *env_version = getenv("SEQ_VER");

  if (env_version != NULL) {
    bool found = false;
    int env_version_env_len = strlen(env_version);
    for (int i = 0; i < sizeof(versions) / sizeof(versions[0]); i++) {
      char *cur_version = versions[i];
      void *cur_ptr = fcptrs[i];
      if (env_version_env_len == strlen(cur_version) &&
          strncmp(env_version, cur_version, env_version_env_len) == 0) {
        found = true;
        ver_idx = i;
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "Couldn't find seq version matching %s. Aborting.\n",
              env_version);
      exit(1);
    }
  }

  fct_dgemm_seq.mpi = 0;
  fct_dgemm_seq.tiled = 0;
  fct_dgemm_seq.starpu = 0;
  fct_dgemm_seq.name = "seq";
  char *helper = calloc(255,
                        sizeof(char)); // TODO: find where in the app
                                       // lifecycle I could free this one
  snprintf(helper, 255, "Sequential version of DGEMM, %s.", versions[ver_idx]);
  fct_dgemm_seq.helper = helper;
  fct_dgemm_seq.fctptr = fcptrs[ver_idx];
  fct_dgemm_seq.next = NULL;

  register_fct(&fct_dgemm_seq, ALGO_GEMM);

  /* Read the value of dgemm_block_size */
  dgemm_seq_block_size = myblas_getenv_value_int("BLOCKSIZE", 1);
}
