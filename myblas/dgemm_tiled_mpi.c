/**
 *
 * @file dgemm_tiled_mpi.c
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
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

int dgemm_tiled_mpi(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                    CBLAS_TRANSPOSE transB, int M, int N, int K, int b,
                    double alpha, const double **A, const double **B,
                    double beta, double **C) {
#if !defined(ENABLE_MPI)
  return ALGONUM_NOT_IMPLEMENTED;
#else
  MPI_Status status;
  const double *Aptr, *Bptr;
  double wsA[b * b], wsB[b * b];
  double lbeta;

  /* Let's compute the total number of tiles with a *ceil* */
  int MT = my_iceil(M, b);
  int NT = my_iceil(N, b);
  int KT = my_iceil(K, b);

  int m, n, k, mm, nn, kk;
  int ownerA, ownerB, ownerC;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {
      for (m = 0; m < MT; m++) {
        mm = m == (MT - 1) ? M - m * b : b;

        for (n = 0; n < NT; n++) {
          nn = n == (NT - 1) ? N - n * b : b;
          ownerC = get_rank_of(m, n);

          if (global_options.mpirank == ownerC) {
            for (k = 0; k < KT; k++) {
              kk = k == (KT - 1) ? K - k * b : b;
              lbeta = (k == 0) ? beta : 1.;

              ownerA = get_rank_of(m, k);
              ownerB = get_rank_of(k, n);

              if (global_options.mpirank == ownerA) {
                Aptr = A[MT * k + m];
              } else {
                MPI_Recv(wsA, b * kk, MPI_DOUBLE, ownerA, MT * k + m,
                         MPI_COMM_WORLD, &status);
                Aptr = wsA;
              }

              if (global_options.mpirank == ownerB) {
                Bptr = B[KT * n + k];
              } else {
                MPI_Recv(wsB, b * nn, MPI_DOUBLE, ownerB, KT * n + k,
                         MPI_COMM_WORLD, &status);
                Bptr = wsB;
              }

              dgemm_seq(CblasColMajor, transA, transB, mm, nn, kk, alpha, Aptr,
                        b, Bptr, b, lbeta, C[MT * n + m], b);
            }
          } else {
            for (k = 0; k < KT; k++) {
              kk = k == (KT - 1) ? K - k * b : b;
              lbeta = (k == 0) ? beta : 1.;

              ownerA = get_rank_of(m, k);
              ownerB = get_rank_of(k, n);

              if (global_options.mpirank == ownerA) {
                MPI_Send(A[MT * k + m], b * kk, MPI_DOUBLE, ownerC, MT * k + m,
                         MPI_COMM_WORLD);
              }

              if (global_options.mpirank == ownerB) {
                MPI_Send(B[KT * n + k], b * nn, MPI_DOUBLE, ownerC, KT * n + k,
                         MPI_COMM_WORLD);
              }
            }
          }
        }
      }
    } else {
      return ALGONUM_NOT_IMPLEMENTED;
    }
  } else {
    return ALGONUM_NOT_IMPLEMENTED;
  }

  return ALGONUM_SUCCESS;
#endif
}

int dgemm_tiled_mpi_summa(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                          CBLAS_TRANSPOSE transB, int M, int N, int K, int b,
                          double alpha, const double **A, const double **B,
                          double beta, double **C) {
#if !defined(ENABLE_MPI)
  return ALGONUM_NOT_IMPLEMENTED;
#else
  int rank, size;
  MPI_Status status;
  MPI_Comm row_comm, col_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int MT = my_iceil(M, b);
  int NT = my_iceil(N, b);
  int KT = my_iceil(K, b);

  int my_p = rank / global_options.Q;
  int my_q = rank % global_options.Q;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {

      MPI_Comm_split(MPI_COMM_WORLD, my_p, my_q, &row_comm);
      MPI_Comm_split(MPI_COMM_WORLD, my_q, my_p, &col_comm);

      int max_loc_n = (NT + global_options.Q - 1) / global_options.Q;
      double *wsB = malloc(max_loc_n * b * b * sizeof(double));
      double wsA[b * b];

      int m, n, k, mm, nn, kk;
      double lbeta;

      for (k = 0; k < KT; k++) {
        kk = k == (KT - 1) ? K - k * b : b;
        lbeta = (k == 0) ? beta : 1.;

        int owner_row_B = k % global_options.P;
        int owner_col_A = k % global_options.Q;

        int idx_n = 0;
        for (n = my_q; n < NT; n += global_options.Q) {
          double *dest = wsB + idx_n * b * b;

          if (my_p == owner_row_B) {
            memcpy(dest, B[KT * n + k], b * b * sizeof(double));
          }

          MPI_Bcast(dest, b * b, MPI_DOUBLE, owner_row_B, col_comm);
          idx_n++;
        }

        for (m = my_p; m < MT; m += global_options.P) {
          mm = m == (MT - 1) ? M - m * b : b;

          double *currA = wsA;
          if (my_q == owner_col_A) {
            currA = (double *)A[MT * k + m];
          }
          MPI_Bcast(currA, b * b, MPI_DOUBLE, owner_col_A, row_comm);

          idx_n = 0;
          for (n = my_q; n < NT; n += global_options.Q) {
            nn = n == (NT - 1) ? N - n * b : b;
            double *currB = wsB + idx_n * b * b;

            /*dgemm_seq(layout, transA, transB,
                      mm, nn, kk,
                      alpha, currA, b, currB, b,
                      lbeta, C[MT * n + m], b);*/

            dgemm_omp(layout, transA, transB, mm, nn, kk, alpha, currA, b,
                      currB, b, lbeta, C[MT * n + m], b);
            idx_n++;
          }
        }
      }

      free(wsB);
      MPI_Comm_free(&row_comm);
      MPI_Comm_free(&col_comm);
    } else {
      return ALGONUM_NOT_IMPLEMENTED;
    }
  } else {
    return ALGONUM_NOT_IMPLEMENTED;
  }

  return ALGONUM_SUCCESS;
#endif
}

int dgemm_tiled_mpi_summa_opt(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                              CBLAS_TRANSPOSE transB, int M, int N, int K,
                              int b, double alpha, const double **A,
                              const double **B, double beta, double **C) {
#if !defined(ENABLE_MPI)
  return ALGONUM_NOT_IMPLEMENTED;
#else
  int rank, size;
  MPI_Status status;
  MPI_Comm row_comm, col_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int MT = my_iceil(M, b);
  int NT = my_iceil(N, b);
  int KT = my_iceil(K, b);

  int my_p = rank / global_options.Q;
  int my_q = rank % global_options.Q;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {

      MPI_Comm_split(MPI_COMM_WORLD, my_p, my_q, &row_comm);
      MPI_Comm_split(MPI_COMM_WORLD, my_q, my_p, &col_comm);

      int max_loc_n = (NT + global_options.Q - 1) / global_options.Q;
      double *wsB = malloc(max_loc_n * b * b * sizeof(double));
      double wsA[b * b];

      int bs_opt = (b < 128) ? b : 128;
      int max_tiles = (b + bs_opt - 1) / bs_opt;
      int tile_sz = bs_opt * bs_opt;
      double *packedA =
          malloc(max_tiles * max_tiles * tile_sz * sizeof(double));
      double *packedB =
          malloc(max_tiles * max_tiles * tile_sz * sizeof(double));
      double *packedC =
          malloc(max_tiles * max_tiles * tile_sz * sizeof(double));
      const double **ptileA = malloc(max_tiles * max_tiles * sizeof(double *));
      const double **ptileB = malloc(max_tiles * max_tiles * sizeof(double *));
      double **ptileC = malloc(max_tiles * max_tiles * sizeof(double *));

      int m, n, k, mm, nn, kk;
      double lbeta;

      for (k = 0; k < KT; k++) {
        kk = k == (KT - 1) ? K - k * b : b;
        lbeta = (k == 0) ? beta : 1.;

        int owner_row_B = k % global_options.P;
        int owner_col_A = k % global_options.Q;

        int idx_n = 0;
        for (n = my_q; n < NT; n += global_options.Q) {
          double *dest = wsB + idx_n * b * b;

          if (my_p == owner_row_B) {
            memcpy(dest, B[KT * n + k], b * b * sizeof(double));
          }

          MPI_Bcast(dest, b * b, MPI_DOUBLE, owner_row_B, col_comm);
          idx_n++;
        }

        for (m = my_p; m < MT; m += global_options.P) {
          mm = m == (MT - 1) ? M - m * b : b;

          double *currA = wsA;
          if (my_q == owner_col_A) {
            currA = (double *)A[MT * k + m];
          }
          MPI_Bcast(currA, b * b, MPI_DOUBLE, owner_col_A, row_comm);

          idx_n = 0;
          for (n = my_q; n < NT; n += global_options.Q) {
            nn = n == (NT - 1) ? N - n * b : b;
            double *currB = wsB + idx_n * b * b;

            /*dgemm_seq(layout, transA, transB,
                      mm, nn, kk,
                      alpha, currA, b, currB, b,
                      lbeta, C[MT * n + m], b);*/

            /* Packing A */
            int MT_km = (mm + bs_opt - 1) / bs_opt;
            int KT_km = (kk + bs_opt - 1) / bs_opt;
            for (int k_loc = 0; k_loc < KT_km; k_loc++) {
              int kk_loc = (k_loc == KT_km - 1) ? kk - k_loc * bs_opt : bs_opt;
              for (int m_loc = 0; m_loc < MT_km; m_loc++) {
                int mm_loc =
                    (m_loc == MT_km - 1) ? mm - m_loc * bs_opt : bs_opt;
                double *dest = packedA + (k_loc * MT_km + m_loc) * tile_sz;
                ptileA[k_loc * MT_km + m_loc] = dest;
                for (int j = 0; j < kk_loc; j++) {
                  memcpy(dest + j * bs_opt,
                         currA + (k_loc * bs_opt + j) * b + m_loc * bs_opt,
                         mm_loc * sizeof(double));
                }
              }
            }

            /* Packing B */
            int NT_kn = (nn + bs_opt - 1) / bs_opt;
            for (int n_loc = 0; n_loc < NT_kn; n_loc++) {
              int nn_loc = (n_loc == NT_kn - 1) ? nn - n_loc * bs_opt : bs_opt;
              for (int k_loc = 0; k_loc < KT_km; k_loc++) {
                int kk_loc =
                    (k_loc == KT_km - 1) ? kk - k_loc * bs_opt : bs_opt;
                double *dest = packedB + (n_loc * KT_km + k_loc) * tile_sz;
                ptileB[n_loc * KT_km + k_loc] = dest;
                for (int j = 0; j < nn_loc; j++) {
                  memcpy(dest + j * bs_opt,
                         currB + (n_loc * bs_opt + j) * b + k_loc * bs_opt,
                         kk_loc * sizeof(double));
                }
              }
            }

            /* Packing C */
            for (int n_loc = 0; n_loc < NT_kn; n_loc++) {
              int nn_loc = (n_loc == NT_kn - 1) ? nn - n_loc * bs_opt : bs_opt;
              for (int m_loc = 0; m_loc < MT_km; m_loc++) {
                int mm_loc =
                    (m_loc == MT_km - 1) ? mm - m_loc * bs_opt : bs_opt;
                double *dest = packedC + (n_loc * MT_km + m_loc) * tile_sz;
                ptileC[n_loc * MT_km + m_loc] = dest;
                double *srcC = C[MT * n + m];

                for (int j = 0; j < nn_loc; j++) {
                  memcpy(dest + j * bs_opt,
                         srcC + (n_loc * bs_opt + j) * b + m_loc * bs_opt,
                         mm_loc * sizeof(double));
                }
              }
            }

            dgemm_tiled_omp(layout, transA, transB, mm, nn, kk, bs_opt, alpha,
                            ptileA, ptileB, lbeta, ptileC);

            /* Unpacking C */
            for (int n_loc = 0; n_loc < NT_kn; n_loc++) {
              int nn_loc = (n_loc == NT_kn - 1) ? nn - n_loc * bs_opt : bs_opt;
              for (int m_loc = 0; m_loc < MT_km; m_loc++) {
                int mm_loc =
                    (m_loc == MT_km - 1) ? mm - m_loc * bs_opt : bs_opt;
                double *src = packedC + (n_loc * MT_km + m_loc) * tile_sz;
                double *destC = C[MT * n + m];

                for (int j = 0; j < nn_loc; j++) {
                  memcpy(destC + (n_loc * bs_opt + j) * b + m_loc * bs_opt,
                         src + j * bs_opt, mm_loc * sizeof(double));
                }
              }
            }
            idx_n++;
          }
        }
      }

      free(packedA);
      free(packedB);
      free(packedC);
      free(ptileA);
      free(ptileB);
      free(ptileC);
      free(wsB);
      MPI_Comm_free(&row_comm);
      MPI_Comm_free(&col_comm);
    } else {
      return ALGONUM_NOT_IMPLEMENTED;
    }
  } else {
    return ALGONUM_NOT_IMPLEMENTED;
  }

  return ALGONUM_SUCCESS;
#endif
}

int dgemm_tiled_mpi_summa_vendor(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                          CBLAS_TRANSPOSE transB, int M, int N, int K, int b,
                          double alpha, const double **A, const double **B,
                          double beta, double **C) {
#if !defined(ENABLE_MPI)
  return ALGONUM_NOT_IMPLEMENTED;
#else
  int rank, size;
  MPI_Status status;
  MPI_Comm row_comm, col_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int MT = my_iceil(M, b);
  int NT = my_iceil(N, b);
  int KT = my_iceil(K, b);

  int my_p = rank / global_options.Q;
  int my_q = rank % global_options.Q;

  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {

      MPI_Comm_split(MPI_COMM_WORLD, my_p, my_q, &row_comm);
      MPI_Comm_split(MPI_COMM_WORLD, my_q, my_p, &col_comm);

      int max_loc_n = (NT + global_options.Q - 1) / global_options.Q;
      double *wsB = malloc(max_loc_n * b * b * sizeof(double));
      double wsA[b * b];

      int m, n, k, mm, nn, kk;
      double lbeta;

      for (k = 0; k < KT; k++) {
        kk = k == (KT - 1) ? K - k * b : b;
        lbeta = (k == 0) ? beta : 1.;

        int owner_row_B = k % global_options.P;
        int owner_col_A = k % global_options.Q;

        int idx_n = 0;
        for (n = my_q; n < NT; n += global_options.Q) {
          double *dest = wsB + idx_n * b * b;

          if (my_p == owner_row_B) {
            memcpy(dest, B[KT * n + k], b * b * sizeof(double));
          }

          MPI_Bcast(dest, b * b, MPI_DOUBLE, owner_row_B, col_comm);
          idx_n++;
        }

        for (m = my_p; m < MT; m += global_options.P) {
          mm = m == (MT - 1) ? M - m * b : b;

          double *currA = wsA;
          if (my_q == owner_col_A) {
            currA = (double *)A[MT * k + m];
          }
          MPI_Bcast(currA, b * b, MPI_DOUBLE, owner_col_A, row_comm);

          idx_n = 0;
          for (n = my_q; n < NT; n += global_options.Q) {
            nn = n == (NT - 1) ? N - n * b : b;
            double *currB = wsB + idx_n * b * b;

            dgemm_vendor(layout, transA, transB, mm, nn, kk, alpha, currA, b,
                      currB, b, lbeta, C[MT * n + m], b);
            idx_n++;
          }
        }
      }

      free(wsB);
      MPI_Comm_free(&row_comm);
      MPI_Comm_free(&col_comm);
    } else {
      return ALGONUM_NOT_IMPLEMENTED;
    }
  } else {
    return ALGONUM_NOT_IMPLEMENTED;
  }

  return ALGONUM_SUCCESS;
#endif
}

/* To make sure we use the right prototype */
static dgemm_tiled_fct_t valid_dgemm_tiled_mpi __attribute__((unused)) =
    dgemm_tiled_mpi;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled_mpi;

/**
 * @brief Registration function
 */
void dgemm_tiled_mpi_init(void) __attribute__((constructor));
void dgemm_tiled_mpi_init(void) {

  int ver_idx = 0;

  char *versions[] = {"base", "summa", "summa-opt", "summa-vendor"};
  void *fcptrs[] = {dgemm_tiled_mpi, dgemm_tiled_mpi_summa,
                    dgemm_tiled_mpi_summa_opt, dgemm_tiled_mpi_summa_vendor};
  char *env_version = getenv("MPI_VER");

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

  fct_dgemm_tiled_mpi.mpi = 1;
  fct_dgemm_tiled_mpi.tiled = 1;
  fct_dgemm_tiled_mpi.starpu = 0;
  fct_dgemm_tiled_mpi.name = "mpi";
  char *helper = calloc(255,
                        sizeof(char)); // TODO: find where in the app
                                       // lifecycle I could free this one
  snprintf(helper, 255, "MPI version of DGEMM, %s.", versions[ver_idx]);
  fct_dgemm_tiled_mpi.helper = helper;
  fct_dgemm_tiled_mpi.fctptr = fcptrs[ver_idx];
  fct_dgemm_tiled_mpi.next = NULL;

  register_fct(&fct_dgemm_tiled_mpi, ALGO_GEMM);
}
