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
#include "myblas.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
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

// micro-kernel is MRxNR
#define MR (4)
#define NR (4)
// panels of A are MCxKC, panels of B are KCxN, panels of C are MCxN
#define KC (192)
#define MC (128)
#if (KC % MR != 0 || KC % NR != 0 || MC % MR != 0 || MC % NR != 0)
#error "KC or MC is not a multiple of MKR"
#endif

static inline void dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                              CBLAS_TRANSPOSE transB, const int M, const int N,
                              const int K, const double alpha, const double *A,
                              const int lda, const double *B, const int ldb,
                              const double beta, double *C, const int ldc);

static inline void dgepp(const int M, const int N, const int K,
                         const double alpha, const double *A_panel,
                         const int lda, const double *B, const int ldb,
                         double *B_packed, double *C);

static inline void dgebp(const int M, const int N, const int K,
                         const double alpha, const double *A_block,
                         const int lda, const double *B_panel, const int ldb,
                         double *C);

static inline void dgemm_kernel(const int M, const int N, const int K,
                                const double alpha, const double *A,
                                const int lda, const double *B, const int ldb,
                                double *C);

static inline void dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                              CBLAS_TRANSPOSE transB, const int M, const int N,
                              const int K, const double alpha, const double *A,
                              const int lda, const double *B, const int ldb,
                              const double beta, double *C, const int ldc) {
  int k, n, m;
  double *B_packed = aligned_alloc(32, sizeof(double) * N * KC);

  // first we scale the whole matrix. TODO: optimize this?
  int rem = M % VEC_BLOCK_SIZE;
  __m256d C_mn_subvec;
  if (beta != 1.0) {
    for (n = 0; n < N; n++) {
      for (m = 0; m < M - rem; m += VEC_BLOCK_SIZE) {
        C_mn_subvec = _mm256_loadu_pd(&C[ldc * n + m]);
        C_mn_subvec = _mm256_mul_pd(C_mn_subvec, _mm256_set1_pd(beta));
        _mm256_storeu_pd(&C[ldc * n + m], C_mn_subvec);
      }
      for (int m = M - rem; m < M; m++) {
        C[ldc * n + m] *= beta;
      }
    }
  }

  // and now we go through each block KC on the K dimension to construct C
  for (k = 0; k < K; k += KC) {
    dgepp(M, N, KC, alpha, &A[lda * k], lda, &B[k], ldb, B_packed, C);
  }

  // TODO: handle remainder
  free(B_packed);
}

static inline void dgepp(const int M, const int N, const int K,
                         const double alpha, const double *A_panel,
                         const int lda, const double *B_panel, const int ldb,
                         double *B_packed, double *C) {
  // we pack B into a contiguous array. b is still stored column major
  // (problem?)
  int m, n, k;
  for (n = 0; n < N; n++) {
    for (k = 0; k < K; k++) {
      B_packed[n * K + k] = B_panel[n * ldb + k];
    }
  }
  // we go line by line on each panel
  for (m = 0; m < M; m += MC) {
    dgebp(MC, N, K, alpha, A_panel + m, lda, B_packed, K, &C[m]);
  }
}

static inline void dgebp(const int M, const int N, const int K,
                         const double alpha, const double *A_block,
                         const int lda, const double *B_panel, const int ldb,
                         double *C) {
  // TODO: pack A into \hat{b}
  double *A_packed = aligned_alloc(32, K * M * sizeof(double));
  double *C_aux = aligned_alloc(32, MR * NR * sizeof(float));
  int m, mm, k, n;
  for (m = 0; m < M; m += MR) {
    for (k = 0; k < K; k++) {
      for (mm = 0; mm < 4; mm++) {
        A_packed[mm + 4 * k + K * m] = A_block[m + mm + lda * k];
      }
    }
  }
  for (n = 0; n < N; n += NR) {
    for (m = 0; m < M; m += MR) {
      dgemm_kernel(MR, NR, K, alpha, &A_packed[m * 4], lda, B_panel, ldb,
                   C_aux);
    }
  }
  for (n = 0; n < NR; n++) {
    for (m = 0; m < NR; n++) {
    }
  }
}

static inline void dgemm_kernel(const int M, const int N, const int K,
                                const double alpha, const double *A,
                                const int lda, const double *B, const int ldb,
                                double *C) {
  int m, n, k;

  for (n = 0; n < N; n++) {
    for (k = 0; k < K; k++) {
      for (m = 0; m < M; m++) {
        C[MR * n + m] += alpha * A[lda * k + m] * B[ldb * n + k];
      }
    }
  }
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

  char *versions[] = {"scalaire", "goto", "bloc", "avx2"};
  void *fcptrs[] = {dgemm_scalaire, dgemm_goto, dgemm_bloc, dgemm_avx2};
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
