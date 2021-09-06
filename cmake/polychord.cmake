include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set(name "polychord")
set(ver "1.17.1")
set(lib "libchord")
set(md5 "c47a4b58e1ce5b98eec3b7d79ec7284f")
set(dl "https://github.com/PolyChord/PolyChordLite/archive/${ver}.tar.gz")
set(dir "${PROJECT_SOURCE_DIR}/contrib/${name}/${ver}")
set(pcSO_LINK "${CMAKE_Fortran_COMPILER} ${OpenMP_Fortran_FLAGS} ${CMAKE_Fortran_MPI_SO_LINK_FLAGS} ${CMAKE_CXX_MPI_SO_LINK_FLAGS}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  string(REGEX REPLACE "(-lstdc\\+\\+)" "-lc++" pcSO_LINK "${pcSO_LINK}")
  string(REGEX MATCH "(-lc\\+\\+)" LINKED_OK "${pcSO_LINK}")
  if (NOT LINKED_OK)
  set(pcSO_LINK "${pcSO_LINK} -lc++")
  endif()
endif()
if(MPI_Fortran_FOUND)
  set(pcFFLAGS "${CMAKE_Fortran_FLAGS} ${MPI_Frotran_COMPILE_FLAGS} -fPIC -O3")
else()
  set(pcFFLAGS "${CMAKE_Fortran_FLAGS} -fPIC -O3")
endif()
if(MPI_CXX_FOUND)
  set(pcCXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -fPIC -O3 -std=c++11")
else()
  set(pcCXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS} -fPIC -O3 -std=c++11")
endif()
if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
  set(pcFFLAGS "${pcFFLAGS} -heap-arrays -assume noold_maxminloc ")
  set(pcCOMPILER_TYPE "intel")
else()
  set(pcFFLAGS "${pcFFLAGS} -fno-stack-arrays -ffree-line-length-none")
  set(pcCOMPILER_TYPE "gnu")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(pcCXXFLAGS "pcCXXFLAGS -isysroot ${CMAKE_OSX_SYSROOT}")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

ExternalProject_Add(${name}#_${ver}
    #DOWNLOAD_COMMAND ${DL_SCANNER} ${dl} ${md5} ${dir} ${name} ${ver}
    URL ${dl}
    URL_MD5 ${md5}
    SOURCE_DIR ${dir}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make ${lib}.a FC=${CMAKE_Fortran_COMPILER} FFLAGS=${pcFFLAGS} CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=${pcCXXFLAGS} LD=${pcSO_LINK} COMPILER_TYPE=${pcCOMPILER_TYPE}
    INSTALL_COMMAND ""
)

add_custom_target(polychord-distclean COMMAND ${CMAKE_COMMAND} -E remove_directory "contrib/polychord"
                                 COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/polychord-prefix" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
