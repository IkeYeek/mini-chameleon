###
#
#  @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                       Univ. Bordeaux. All rights reserved.
#
#  @author Mathieu Faverge
#  @version 0.2.0
#  @date 2021-09-23
#
###
set(TESTINGS_SRCS
  common.c
  )

set(TESTINGS
  check_ddot.c
  check_dgemm.c
  check_dgetrf.c
  perf_ddot.c
  perf_dgemm.c
  perf_dgetrf.c
  warm_ddot.c
  warm_dgemm.c
  warm_dgetrf.c
  )

foreach( _test ${TESTINGS} )

  get_filename_component(_name_we ${_test} NAME_WE)
  add_executable(${_name_we} ${_test} ${TESTINGS_SRCS})

  target_link_libraries( ${_name_we} PUBLIC myblas )
  target_link_libraries( ${_name_we} PUBLIC algonum )
  if (ENABLE_MPI)
    target_link_libraries( ${_name_we} PUBLIC MPI::MPI_C )
  endif (ENABLE_MPI)
  if (ENABLE_STARPU)
    target_link_libraries(${_name_we} PUBLIC MORSE::STARPU)
  endif (ENABLE_STARPU)

  install(TARGETS ${_name_we} RUNTIME DESTINATION bin)

endforeach()

