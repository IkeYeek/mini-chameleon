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
set( algonum_srcs
  core_dplrnt.c
  tile.c
  perf.c
  check_ddot.c
  test_ddot.c
  check_dgemm.c
  test_dgemm.c
  test_dgemm_tiled.c
  check_dgetrf.c
  test_dgetrf.c
  test_dgetrf_tiled.c
  )

if( ENABLE_STARPU )
  list( APPEND algonum_srcs
    codelet_dplrnt.c
    codelet_dlacpy.c
    codelet_dgemm.c
    codelet_dgetrf.c
    codelet_dtrsm.c
    starpu.c
    starpu_dplrnt.c
    starpu_dlacpy.c
    )
endif()

if( ENABLE_CUDA )
  list( APPEND algonum_srcs
    test_dgetrf_cuda.c
    test_dgemm_cuda.c
    )
endif()

add_library( algonum
  ${algonum_srcs}
  )

target_include_directories( algonum PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/include )

target_link_libraries( algonum PUBLIC MORSE::LAPACKE )
target_link_libraries( algonum PUBLIC MORSE::CBLAS   )
target_link_libraries( algonum PUBLIC MORSE::M       )

if(ENABLE_MPI)
  target_link_libraries( algonum PUBLIC MPI::MPI_C )
endif()

if(ENABLE_STARPU)
  target_link_libraries( algonum PUBLIC MORSE::STARPU )
endif()

if(ENABLE_CUDA)
  target_link_libraries( algonum PUBLIC CUDA::CUDA )
  target_link_libraries( algonum PUBLIC CUDA::CUBLAS )
endif()

install(TARGETS algonum
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES
  include/algonum.h
  include/algonum_int.h
  include/cblas.h
  include/codelets.h
  include/flops.h
  include/lapacke.h
  include/perf.h
  DESTINATION include )

