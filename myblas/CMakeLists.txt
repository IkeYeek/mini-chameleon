###
#
#  @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                       Univ. Bordeaux. All rights reserved.
#
#  @author Mathieu Faverge
#  @version 0.2.0
#  @date 2021-09-30
#
###

# Define here the list of DDOT implementation
# -------------------------------------------
set( DDOT_srcs
  ddot_seq.c
  # ADD HERE: your additional ddot implementation
  #  ddot_xxx.c
)

# Define here the list of DGEMM implementation
# --------------------------------------------
set( DGEMM_srcs
  dgemm_seq.c
  dgemm_mkl.c
  dgemm_omp.c
  dgemm_tiled_omp.c
  dgemm_tiled_mpi.c
  # ADD HERE: your additional dgemm implementation
  #  dgemm_xxx.c
)

# Define here the list of DGETRF implementation
# ---------------------------------------------
set( DGETRF_srcs
  dgetrf_seq.c
  dgetrf_mkl.c
  dgetrf_omp.c
  dgetrf_tiled_omp.c
  dgetrf_tiled_mpi.c
  # ADD HERE: your additional dgetrf implementation
  #  dgetrf_xxx.c
)

#------------------------------------------
# DO NOT CHANGE ANYTHING BEYOND THIS POINT
#------------------------------------------

# Define the list of source files for the myblas library
# ------------------------------------------------------
set( myblas_srcs
  fctlist.c
  ${DDOT_srcs}
  ${DGEMM_srcs}
  ${DGETRF_srcs}
)

#
# Add the OpenMP flag to some of the files
#
if (OpenMP_C_FOUND)
  set_source_files_properties(
    dgemm_omp.c
    dgemm_tiled_omp.c
    # ADD HERE: your OpenMP version
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS} )
endif()

if( ENABLE_STARPU )
  list( APPEND myblas_srcs
    dgemm_tiled_starpu.c
    dgetrf_tiled_starpu.c
    )
endif()

# Add the files of your own myblas library here
add_library( myblas
  ${myblas_srcs}
  )

target_link_libraries( myblas PUBLIC algonum )
target_link_libraries( myblas PUBLIC OpenMP::OpenMP_C )

if(ENABLE_MPI)
  target_link_libraries( myblas PUBLIC MPI::MPI_C )
  target_compile_definitions( myblas PUBLIC ENABLE_MPI )
endif()

if(ENABLE_STARPU)
  target_link_libraries( myblas PRIVATE MORSE::STARPU )
  target_compile_definitions( myblas PUBLIC ENABLE_STARPU )
endif()

target_link_libraries( myblas PUBLIC MORSE::LAPACKE )
target_link_libraries( myblas PUBLIC MORSE::CBLAS   )
target_link_libraries( myblas PUBLIC MORSE::M       )

install(TARGETS myblas
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

