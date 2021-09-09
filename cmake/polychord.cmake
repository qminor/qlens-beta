include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

#set(BLA_OUTPUT "")
#if(DEFINED BLA_VENDOR)
    #set(BLA_OUTPUT "-DBLA_VENDOR=${BLA_VENDOR}")
#endif(DEFINED BLA_VENDOR)

## allow a single extra target if needed
#set(EXTRA_TARGET "")
#if(DEFINED MULTINEST_USE_MPI)
    #set(EXTRA_TARGET "multinest_mpi_static")
#endif(DEFINED MULTINEST_USE_MPI)

set(FORTRAN_FLAGS ${CMAKE_Fortran_FLAGS})

if(CMAKE_Fortran_COMPILER_ID STREQUAL GNU AND ${CMAKE_Fortran_COMPILER_VERSION} VERSION_GREATER_EQUAL 10.0.0)
    set(FORTRAN_FLAGS ${FORTRAN_FLAGS} "-fPIC -w -fallow-argument-mismatch")
endif()

if(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)
    set(MPI_TAG "-DMPI=ON")
else(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)
    set(MPI_TAG "-DMPI=OFF")
endif(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)

ExternalProject_Add(polychord
    GIT_REPOSITORY https://github.com/stenczelt/PolyChordLite.git
    GIT_PROGRESS TRUE
    GIT_TAG cmake-dev
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/PolyChordLite
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/PolyChordLite/build
    PATCH_COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/contrib/PolyChordLite/build
    COMMAND patch -N ${CMAKE_CURRENT_SOURCE_DIR}/contrib/PolyChordLite/CMakeLists.txt ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/polychord_cmake.patch
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DMPI_C_COMPILER=${MPI_C_COMPILER} -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER} -DMPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} ${BLA_OUTPUT} -DMPI_C_FOUND=${MPI_C_FOUND} -DMPI_C_INCLUDE_PATH=${MPI_C_INCLUDE_PATH} -DMPI_C_LIBRARIES=${MPI_C_LIBRARIES} -DMPI_C_COMPILE_FLAGS=${MPI_C_COMPILE_FLAGS} -DMPI_CXX_FOUND=${MPI_CXX_FOUND} -DMPI_CXX_INCLUDE_PATH=${MPI_CXX_INCLUDE_PATH} -DMPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES} -DMPI_CXX_COMPILE_FLAGS=${MPI_CXX_COMPILE_FLAGS} -DMPI_Fortran_FOUND=${MPI_Fortran_FOUND} -DMPI_Fortran_INCLUDE_PATH=${MPI_Fortran_INCLUDE_PATH} -DMPI_Fortran_LIBRARIES=${MPI_Fortaran_LIBRARIES} -DMPI_Fortran_COMPILE_FLAGS=${MPI_Fortran_COMPILE_FLAGS} -DCMAKE_Fortran_FLAGS=${FORTRAN_FLAGS} -Dpython=OFF ${MPI_TAG}  -Wno-dev ..
    # must build multinest_static to populate modules directory
    BUILD_COMMAND make chord
    INSTALL_COMMAND ""
  )

add_custom_target(polychord-distclean COMMAND ${CMAKE_COMMAND} -E remove_directory "contrib/PolyChordLite"
                                 COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/polychord-prefix" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
