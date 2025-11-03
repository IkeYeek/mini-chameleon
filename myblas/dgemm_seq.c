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

void affiche(int M, int N, double *A) {
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      printf("%lf ", A[n * M + m]);
    }
    printf("\n");
  }
  printf("\n");
}

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
#define NR (6)
// panels of A are MCxKC, panels of B are KCxN, panels of C are MCxN
/*
 * Paper states that KC*NR*sizeof(double) < L1/2, with NR=6 it leaves us with
 * KC*48<16KB, so let's say KC=320
 */
#define KC (320)
/*
 * It then states that MC*KC*sizeof(double) < L2/2, with KC=320 it leaves us
 * with MC=32
 */
#define MC (32)

static inline int dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                             CBLAS_TRANSPOSE transB, const int M, const int N,
                             const int K, const double alpha, const double *A,
                             const int lda, const double *B, const int ldb,
                             const double beta, double *C, const int ldc);

static inline void dgepp(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, const int M, const int N,
                         const int K, const double alpha, const double *A_panel,
                         const int lda, const double *B_panel, const int ldb,
                         double *B_packed, const double beta, double *C,
                         const int ldc, double *A_packed, double *C_aux);

static inline void dgebp(const int M, const int N, const int K,
                         const double alpha, const double *A_block,
                         const int lda, const double *B_panel, const int ldb,
                         double *C, const int ldc, double *A_packed,
                         double *C_aux);

static inline void dgemm_kernel_4x6(const int M, const int N, const int K,
                                    const double alpha, const double *A_packed,
                                    const double *B_panel, double *C_aux);

static inline void scale_C(const int M, const int N, const double beta,
                           double *C, const int ldc);

#define MIN(a, b) ((a < b) ? a : b)

/**
 * (poorly) based on the latest version of this paper:
 * https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf
 *
 * The goal is to implement GEMM using GEPP and GEBP for respectively Panel x
 * Panel and Block x Panel multiplication and computing into a small microkernel
 * optimized for the architecture
 *
 * Even though no code was used from it, this repo
 * https://github.com/ytsutano/dgemm-goto-in-c/ (and especially its drawings)
 * helped me understanding better the packing of A and the use of \hat{A} with
 * C_{aux} (and how it differs from the packing of B)
 */
static inline int dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                             CBLAS_TRANSPOSE transB, const int M, const int N,
                             const int K, const double alpha, const double *A,
                             const int lda, const double *B, const int ldb,
                             const double beta, double *C, const int ldc) {
  /* TODO:
   * - [x] alloc only once!
   * - [x] AVX microkernel (gotta take the one I already wrote in bloc_vec)
   * - [x] review packing of B, pretty sure I'm doing it wrong
   * - [x] make sure there are no cases where aligned_alloc might fail. Perhaps
   * add some assertions
   * - [x] handle remainder because for now it only works on matrix that are
   * multiple of MC/KC
   * - [ ] profiling the code
   * - [x] find a better way to do scaling as I'm pretty sure it's a bottleneck
   * right now
   * - [x] optimize parameters for the architecture
   * - [x] check the actual impact of aligned vs unaligned load and stores on
   * haswell.
   */

  // Will hold panels of B re-packed into a contiguous array in order to make it
  // easily fit inside of cache lines. It is aligned on 32 bytes to make sure we
  // can use aligned load/stores as it seems to have an impact on haswell
  int ldbp = N + ((NR - (N % NR)) % NR); // TODO: check this
  double *B_packed = aligned_alloc(32, sizeof(double) * ldbp * KC);

  // Allocating them here so we only do it once
  double *A_packed = aligned_alloc(32, M * K * sizeof(double));
  double *C_aux = aligned_alloc(32, MR * NR * sizeof(double));

  if (!B_packed || !A_packed || !C_aux) {
    perror("Aligned Alloc Error");
    exit(1);
  }

  // first we scale the whole matrix.
  scale_C(M, N, beta, C, ldc);

  // and now we go through each block KC on the K dimension to construct C
  int k;
  for (k = 0; k < K; k += KC) {
    int Kb = MIN(KC, K - k);
    const double *A_panel = &A[lda * k];
    const double *B_panel = &B[k];
    dgepp(layout, transA, transB, M, N, Kb, alpha, A_panel, lda, B_panel, ldb,
          B_packed, beta, C, ldc, A_packed, C_aux);
  }

  free(B_packed);
  free(A_packed);
  free(C_aux);

  return ALGONUM_SUCCESS;
}

static inline void scale_C(const int M, const int N, const double beta,
                           double *C, const int ldc) {
  if (beta == 1.0)
    return;
  int rem = M % VEC_BLOCK_SIZE;

  __m256d C_mn_subvec;
  for (int n = 0; n < N; n++) {
    int m;
    for (m = 0; m < M - rem; m += VEC_BLOCK_SIZE) {
      C_mn_subvec = _mm256_loadu_pd(&C[ldc * n + m]);
      C_mn_subvec = _mm256_mul_pd(C_mn_subvec, _mm256_set1_pd(beta));
      _mm256_storeu_pd(&C[ldc * n + m], C_mn_subvec);
    }
    for (; m < M; m++) {
      C[ldc * n + m] *= beta;
    }
  }
}

static inline void dgepp(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, const int M, const int N,
                         const int K, const double alpha, const double *A_panel,
                         const int lda, const double *B_panel, const int ldb,
                         double *B_packed, const double beta, double *C,
                         const int ldc, double *A_packed, double *C_aux) {
  /*
   * Say I have this 6x5 matrix, and panels are 2x5.
   * We want to go from this:
   * |-------------|
   * |1 7  13 19 25|
   * |2 8  14 20 26|
   * |-------------|
   *  3 9  15 21 27
   *  4 10 16 22 28
   *  5 11 17 23 29
   *  6 12 18 24 30

   *  that would be stored as [1, 2, 3, 4, 5, 6, 7, 8, 9...30].
   *  We want to keep only the panel so it is then stored as
   * [1, 7, 13, 19, 2, 8, 14, 20, 25(...)]
   */
  for (int n = 0; n < N; n += NR) {
    int Nb = MIN(NR, N - n);
    int n_block = n / NR;
    for (int k = 0; k < K; k++) {
      for (int nn = 0; nn < NR; nn++) {
        if (n + nn < N) {
          B_packed[n_block * NR * K + k * NR + nn] =
              B_panel[k + (n + nn) * ldb];
        } else {
          B_packed[n_block * NR * K + k * NR + nn] = 0.0;
        }
      }
    }
  }

  int m;
  for (m = 0; m + MC < M; m += MC) {
    int Mb = MIN(MC, M - m);
    dgebp(Mb, N, K, alpha, A_panel + m, lda, B_packed, K, C + m, ldc, A_packed,
          C_aux);
  }
  if (m < M) {
    int Mb = M - m;
    dgemm_scalaire(layout, transA, transB, Mb, N, K, alpha, A_panel + m, lda,
                   B_panel, ldb, 1.0, C + m, ldc);
  }
}

static inline void dgebp(const int M, const int N, const int K,
                         const double alpha, const double *A_block,
                         const int lda, const double *B_panel, const int ldb,
                         double *C, const int ldc, double *A_packed,
                         double *C_aux) {
  /*
   * Say we have this bloc
   * 1 5 9  13
   * 2 6 10 14
   * 3 7 11 15
   * 4 8 12 16

   * which is stored as [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

   * Now say we work with a 2x2 kernel.
   * We're reducing on K, so I use 1,2 - 5,6 - 9,10 (...).
   * It would be beneficial if they were contiguously stored (kind of as a work
   array):
   * [1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16]
   * That is what A_packed is for (and how it differs from the packing of B)
  */
  for (int m = 0; m < M; m += MR) {
    int Mb = MIN(MR, M - m);
    int m_block = m / MR;
    int base = m_block * MR * K;
    for (int k = 0; k < K; k++) {
      for (int mm = 0; mm < MR; mm++) {
        if (m + mm < M) {
          A_packed[base + k * MR + mm] = A_block[m + mm + lda * k];
        } else {
          A_packed[base + k * MR + mm] = 0.0;
        }
      }
    }
  }

  for (int n = 0; n < N; n += NR) {
    int Nb = MIN(NR, N - n);
    int n_block = n / NR;
    for (int m = 0; m < M; m += MR) {
      int Mb = MIN(MR, M - m);
      int m_block = m / MR;

      memset(C_aux, 0, MR * NR * sizeof(double));

      dgemm_kernel_4x6(Mb, Nb, K, alpha, A_packed + m_block * MR * K,
                       B_panel + n_block * NR * K, C_aux);

      // unpacking C_aux into C
      for (int nn = 0; nn < Nb; nn++) {
        for (int mm = 0; mm < Mb; mm++) {
          C[(n + nn) * ldc + (m + mm)] += C_aux[nn * MR + mm];
        }
      }
    }
  }
}

static inline void dgemm_kernel_naive(const int M, const int N, const int K,
                                      const double alpha,
                                      const double *A_packed,
                                      const double *B_panel, double *C_aux) {
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += A_packed[k * MR + m] * B_panel[k * NR + n];
      }
      C_aux[n * MR + m] += alpha * sum;
    }
  }
}

// the use of preproc is totally inspired by the code there:
// "github.com/ytsutano/dgemm-goto-in-c/blob/master/dgemm-goto.c"
#define KERNEL_ITER(k)                                                         \
  do {                                                                         \
    A_vec0 = _mm256_load_pd(&A_packed[(k) * MR]);                              \
                                                                               \
    B_bcast0 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 0]);                  \
    B_bcast1 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 1]);                  \
    B_bcast2 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 2]);                  \
    B_bcast3 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 3]);                  \
    B_bcast4 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 4]);                  \
    B_bcast5 = _mm256_set1_pd(alpha * B_panel[(k) * NR + 5]);                  \
                                                                               \
    C_vec0 = _mm256_fmadd_pd(A_vec0, B_bcast0, C_vec0);                        \
    C_vec1 = _mm256_fmadd_pd(A_vec0, B_bcast1, C_vec1);                        \
    C_vec2 = _mm256_fmadd_pd(A_vec0, B_bcast2, C_vec2);                        \
    C_vec3 = _mm256_fmadd_pd(A_vec0, B_bcast3, C_vec3);                        \
    C_vec4 = _mm256_fmadd_pd(A_vec0, B_bcast4, C_vec4);                        \
    C_vec5 = _mm256_fmadd_pd(A_vec0, B_bcast5, C_vec5);                        \
  } while (0);

static inline void dgemm_kernel_4x6(const int M, const int N, const int K,
                                    const double alpha, const double *A_packed,
                                    const double *B_panel, double *C_aux) {
  __m256d alpha_vec = _mm256_set1_pd(alpha);
  __m256d A_vec0;
  __m256d B_bcast0, B_bcast1, B_bcast2, B_bcast3, B_bcast4, B_bcast5;
  __m256d C_vec0, C_vec1, C_vec2, C_vec3, C_vec4, C_vec5;

  C_vec0 = _mm256_load_pd(C_aux + 0 * MR);
  C_vec1 = _mm256_load_pd(C_aux + 1 * MR);
  C_vec2 = _mm256_load_pd(C_aux + 2 * MR);
  C_vec3 = _mm256_load_pd(C_aux + 3 * MR);
  C_vec4 = _mm256_load_pd(C_aux + 4 * MR);
  C_vec5 = _mm256_load_pd(C_aux + 5 * MR);

  int k;
#define KR 8
  for (k = 0; k + KR < K + 1; k += KR) {
    KERNEL_ITER(k);
    KERNEL_ITER(k + 1);
    KERNEL_ITER(k + 2);
    KERNEL_ITER(k + 3);
    KERNEL_ITER(k + 4);
    KERNEL_ITER(k + 5);
    KERNEL_ITER(k + 6);
    KERNEL_ITER(k + 7);
  }
  if (k < K) {
    for (; k < K; k += 1) {
      KERNEL_ITER(k);
    }
  }

  _mm256_store_pd(C_aux + 0 * MR, C_vec0);
  _mm256_store_pd(C_aux + 1 * MR, C_vec1);
  _mm256_store_pd(C_aux + 2 * MR, C_vec2);
  _mm256_store_pd(C_aux + 3 * MR, C_vec3);
  _mm256_store_pd(C_aux + 4 * MR, C_vec4);
  _mm256_store_pd(C_aux + 5 * MR, C_vec5);
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
