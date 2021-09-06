#include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set(FORTRAN_FLAGS ${CMAKE_Fortran_FLAGS})

#if(CMAKE_Fortran_COMPILER_ID STREQUAL GNU AND ${CMAKE_Fortran_COMPILER_VERSION} VERSION_GREATER_EQUAL 10.0.0)
    #set(FORTRAN_FLAGS ${FORTRAN_FLAGS} "-fPIC -w -fallow-argument-mismatch")
#endif()

ExternalProject_Add(mumps
    GIT_REPOSITORY https://github.com/scivision/mumps.git
    GIT_PROGRESS TRUE
    GIT_TAG ${MUMPS_VERS_STR} #"v5.4.0.7"
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/mumps
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/mumps/build
    PATCH_COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/contrib/mumps/build
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DMPI_C_COMPILER=${MPI_C_COMPILER} -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER} -DMPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} ${BLA_OUTPUT} -DMPI_C_FOUND=${MPI_C_FOUND} -DMPI_C_INCLUDE_PATH=${MPI_C_INCLUDE_PATH} -DMPI_C_LIBRARIES=${MPI_C_LIBRARIES} -DMPI_C_COMPILE_FLAGS=${MPI_C_COMPILE_FLAGS} -DMPI_CXX_FOUND=${MPI_CXX_FOUND} -DMPI_CXX_INCLUDE_PATH=${MPI_CXX_INCLUDE_PATH} -DMPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES} -DMPI_CXX_COMPILE_FLAGS=${MPI_CXX_COMPILE_FLAGS} -DMPI_Fortran_FOUND=${MPI_Fortran_FOUND} -DMPI_Fortran_INCLUDE_PATH=${MPI_Fortran_INCLUDE_PATH} MPI_Fortran_LIBRARIES=${MPI_Fortaran_LIBRARIES} -DMPI_Fortran_COMPILE_FLAGS=${MPI_Fortran_COMPILE_FLAGS} -DCMAKE_Fortran_FLAGS=${FORTRAN_FLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/contrib/mumps -Dparallel=1 -Wno-dev ..
    BUILD_COMMAND make
    INSTALL_COMMAND make install
  )

add_custom_target(mumps-distclean COMMAND ${CMAKE_COMMAND} -E remove_directory "contrib/mumps"
                                 COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/mumps-prefix" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})