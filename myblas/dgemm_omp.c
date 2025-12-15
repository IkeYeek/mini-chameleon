/**
 *
 * @file dgemm_omp.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Template to develop the OpenMP version
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-30
 *
 */
#include "myblas.h"
#include <immintrin.h>
#include <stdbool.h>
#include <string.h>

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
#define MC (256)

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
    return ALGONUM_NOT_IMPLEMENTED;
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for private(A_packed, C_aux) schedule(static)             \
    shared(B_packed, C)
  for (int m = 0; m < M; m += MC) {
    A_packed = aligned_alloc(32, MC * KC * sizeof(double));
    C_aux = aligned_alloc(32, MR * NR * sizeof(double));
    int Mb = MIN(MC, M - m);
    dgebp(Mb, N, K, alpha, A_panel + m, lda, B_packed, K, C + m, ldc, A_packed,
          C_aux);
    free(A_packed);
    free(C_aux);
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

int dgemm_omp(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
              CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
              const double alpha, const double *A, const int lda,
              const double *B, const int ldb, const double beta, double *C,
              const int ldc) {
  return dgemm_goto(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_omp __attribute__((unused)) = dgemm_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_omp;

/**
 * @brief Registration function
 */
void dgemm_omp_init(void) __attribute__((constructor));
void dgemm_omp_init(void) {
  fct_dgemm_omp.mpi = 0;
  fct_dgemm_omp.tiled = 0;
  fct_dgemm_omp.starpu = 0;
  fct_dgemm_omp.name = "omp";
  fct_dgemm_omp.helper = "Scalar OpenMP implementation of the dgemm";
  fct_dgemm_omp.fctptr = dgemm_omp;
  fct_dgemm_omp.next = NULL;

  register_fct(&fct_dgemm_omp, ALGO_GEMM);
}
