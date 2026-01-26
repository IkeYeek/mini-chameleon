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
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Exemple of ways to add additionnal parameters to your kernel
// See the registration function to change its value
static int dgemm_seq_block_size = -1;

#define MIN(a, b) ((a < b) ? a : b)
#define VEC_BLOCK_SIZE 4
// added this to be able to force disabling inlining because it makes profiling
// easier
#define STATIC_INLINE (true)
// micro-kernel is MRxNR
#define MR (4)
#define NR (8)
// panels of A are MCxKC, panels of B are KCxN, panels of C are MCxN
// KC and MC where obtaind through empirical testing
#define KC (256)
#define MC (2048)

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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    int
    dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
               const double alpha, const double *A, const int lda,
               const double *B, const int ldb, const double beta, double *C,
               const int ldc);

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgepp(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
          const int M, const int N, const int K, const double alpha,
          const double *A_panel, const int lda, const double *B_panel,
          const int ldb, double *B_packed, const double beta, double *C,
          const int ldc, double *A_packed, double *C_aux);

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgebp(const int M, const int N, const int K, const double alpha,
          const double *A_block, const int lda, const double *B_panel,
          const int ldb, double *C, const int ldc, double *A_packed,
          double *C_aux);

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgemm_kernel_4x8(const int M, const int N, const int K, const double alpha,
                     const double *A_packed, const double *B_panel,
                     double *C_aux);

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    scale_C(const int M, const int N, const double beta, double *C,
            const int ldc);

/**
 * (poorly) based on the latest version of this paper:
 * https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf
 *
 * The goal is to implement GEMM using GEPP and GEBP
 * for respectively Panel x Panel and Block x Panel
 * multiplication and computing into a small
 * microkernel optimized for the architecture
 *
 * Even though no code was used from it, this repo
 * https://github.com/ytsutano/dgemm-goto-in-c/ (and
 * especially its drawings) helped me understanding
 * better the packing of A and the use of \hat{A} with
 * C_{aux} (and how it differs from the packing of B)
 */
#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    int
    dgemm_goto(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
               const double alpha, const double *A, const int lda,
               const double *B, const int ldb, const double beta, double *C,
               const int ldc) {
  // Will hold panels of B re-packed into a contiguous
  // array in order to make it easily fit inside of
  // cache lines. It is aligned on 32 bytes to make
  // sure we can use aligned load/stores as it seems to
  // have an impact on haswell
  if (transA != CblasNoTrans || transB != CblasNoTrans) {
    return dgemm_scalaire(layout,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
  }
  int ldbp = N + ((NR - (N % NR)) % NR);
  double *B_packed = aligned_alloc(32, sizeof(double) * ldbp * KC);

  // Allocating them here so we only do it once
  double *A_packed = aligned_alloc(32, MC * KC * sizeof(double));
  double *C_aux = aligned_alloc(32, MR * NR * sizeof(double));

  if (!B_packed || !A_packed || !C_aux) {
    perror("Aligned Alloc Error");
    exit(1);
  }

  // first we scale the whole matrix.
  scale_C(M, N, beta, C, ldc);

  // and now we go through each block KC on the K
  // dimension to construct C
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    scale_C(const int M, const int N, const double beta, double *C,
            const int ldc) {
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgepp(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
          const int M, const int N, const int K, const double alpha,
          const double *A_panel, const int lda, const double *B_panel,
          const int ldb, double *B_packed, const double beta, double *C,
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

   *  that would be stored as [1, 2, 3, 4, 5, 6, 7,
   8, 9...30].
   *  We want to keep only the panel so it is then
   stored as
   * [1, 7, 13, 19, 2, 8, 14, 20, 25(...)]
   */
  for (int n = 0; n < N; n += NR) {
    int Nb = MIN(NR, N - n);
    int n_block = n / NR;
    for (int k = 0; k < K; k++) {
      for (int nn = 0; nn < NR; nn++) {
        if (n + nn < N) {
          B_packed[n_block * NR * K + k * NR + nn] =
              alpha * B_panel[k + (n + nn) * ldb];
        } else {
          B_packed[n_block * NR * K + k * NR + nn] = 0.0;
        }
      }
    }
  }

  __builtin_prefetch(B_packed); // noticed small improvements when doing that
  for (int m = 0; m < M; m += MC) {
    int Mb = MIN(MC, M - m);
    dgebp(Mb, N, K, alpha, A_panel + m, lda, B_packed, K, C + m, ldc, A_packed,
          C_aux);
  }
}

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgebp(const int M, const int N, const int K, const double alpha,
          const double *A_block, const int lda, const double *B_panel,
          const int ldb, double *C, const int ldc, double *A_packed,
          double *C_aux) {
  /*
   * Say we have this bloc
   * 1 5 9  13
   * 2 6 10 14
   * 3 7 11 15
   * 4 8 12 16

   * which is stored as [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
   11, 12, 13, 14, 15, 16]

   * Now say we work with a 2x2 kernel.
   * We're reducing on K, so I use 1,2 - 5,6 - 9,10
   (...).
   * It would be beneficial if they were contiguously
   stored (kind of as a work array):
   * [1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12,
   15, 16]
   * That is what A_packed is for (and how it differs
   from the packing of B)
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

  __builtin_prefetch(A_packed);
  for (int n = 0; n < N; n += NR) {
    int Nb = MIN(NR, N - n);
    int n_block = n / NR;
    for (int m = 0; m < M; m += MR) {
      int Mb = MIN(MR, M - m);
      int m_block = m / MR;

      memset(C_aux, 0, MR * NR * sizeof(double));

      dgemm_kernel_4x8(Mb, Nb, K, alpha, A_packed + m_block * MR * K,
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

// this one was for testing purposes only
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    dgemm_kernel_4x8(const int M, const int N, const int K, const double alpha,
                     const double *A_packed, const double *B_panel,
                     double *C_aux) {
  __m256d C_vec[NR];
#pragma GCC unroll 8
  for (int i = 0; i < NR; i++) {
    C_vec[i] = _mm256_load_pd(C_aux + i * MR);
  }

#pragma GCC unroll 8
  for (int k = 0; k < K; k++) {
    __m256d A_vec = _mm256_load_pd(&A_packed[k * MR]);

#pragma GCC unroll 8
    for (int i = 0; i < NR; i++) {
      __m256d B_bcast = _mm256_set1_pd(B_panel[k * NR + i]);
      C_vec[i] = _mm256_fmadd_pd(A_vec, B_bcast, C_vec[i]);
    }
  }

#pragma GCC unroll 8
  for (int i = 0; i < NR; i++) {
    _mm256_store_pd(C_aux + i * MR, C_vec[i]);
  }
}


#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    void
    scale_vec(const int M, const int N, const int rem, const double beta,
              double *C, const int ldc) {
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    int
    dgemm_avx2(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
               const double alpha, const double *A, const int lda,
               const double *B, const int ldb, const double beta, double *C,
               const int ldc) {

  if (transA != CblasNoTrans || transB != CblasNoTrans) {
    ALGONUM_NOT_IMPLEMENTED;
  }
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    int
    dgemm_avx2_microkernel_4x4(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
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

#if STATIC_INLINE
static inline
#else
__attribute__((noinline))
#endif
    int
    dgemm_avx2_bloc(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
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

  // we compute the parts that cant be handled by microkernel
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
            // C[ldc * n + m] += alpha * A[lda * k + m]
            // * B[ldb * n + k];
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
          // if beta != 0, scale the C block before
          // accumulating into it
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
          // if beta != 0, scale the C block before
          // accumulating into it
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
  dgemm_goto(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
             ldc);

  return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_seq __attribute__((unused)) = dgemm_seq;

/* Declare the variable that will store the information
 * about this version */
fct_list_t fct_dgemm_seq;

/**
 * @brief Registration function
 */
void dgemm_seq_init(void) __attribute__((constructor));
void dgemm_seq_init(void) {
  int ver_idx = 0;

  char *versions[] = {"goto", "scalaire", "bloc", "avx2",
                      "avx2_bloc"}; // Goto as default if SEQ_VER not
                                    // defined
  void *fcptrs[] = {dgemm_goto, dgemm_scalaire, dgemm_bloc, dgemm_avx2,
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
      fprintf(stderr,
              "Couldn't find seq version matching %s. "
              "Aborting.\n",
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
