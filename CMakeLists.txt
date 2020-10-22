###
#
#  @copyright 2019-2020 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                       Univ. Bordeaux. All rights reserved.
#
#  @author Mathieu Faverge
#  @date 2019-11-13
#
# This is the main CmakeLists.txt of the project.
# You shoudl not have to modify it.
#
###
cmake_minimum_required (VERSION 2.8)
project(myblas)

include_directories( ${CMAKE_SOURCE_DIR}/algonum/include )

#
# Add option to detect MPI (A partir de la séance 2)
#
option( ENABLE_MPI "Define if MPI support should be enabled" OFF )
if (ENABLE_MPI)
  add_definitions(-DENABLE_MPI)
endif()

#
# Add option to detect the StarPU package (A partir de la séance 3)
#
option( ENABLE_STARPU "Define if StarPU support is enabled or not" OFF )
if (ENABLE_STARPU)
  add_definitions(-DENABLE_STARPU)
endif()

#
# Add option to detect the CUDA (Bonus not yet available)
#
option( ENABLE_CUDA "Define if CUDA support is enabled or not" OFF )
if (ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()

#
# Find external packages
#
if (ENABLE_MPI)
  find_package(MPI)
endif()

if( ENABLE_STARPU )
  include(FindPkgConfig)
  find_package(PkgConfig QUIET)

  set (starpu_version "1.3")
  if( ENABLE_MPI )
    pkg_search_module(STARPU starpumpi-${starpu_version})
  else()
    pkg_search_module(STARPU starpu-${starpu_version})
  endif()

  if (STARPU_FOUND AND STARPU_LIBRARIES)
    message(STATUS "Looking for STARPU - found using PkgConfig")
  else()
    message(ERROR "${Magenta}Looking for STARPU - not found using PkgConfig."
      "\n   Perhaps you should add the directory containing libstarpu.pc"
      "\n   to the PKG_CONFIG_PATH environment variable.${ColourReset}")
  endif()

  include_directories(${STARPU_INCLUDE_DIRS})
  link_directories(${STARPU_LIBRARY_DIRS})
endif()

#
# Find OpenMP if available
#
find_package(OpenMP)

#
# Add libraries
# -------------

# Internal one
add_subdirectory( algonum )

# Your BLAS library
add_subdirectory( myblas )

# Testings and performance evaluation binaries
add_subdirectory( testings )
