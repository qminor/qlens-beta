cmake_minimum_required(VERSION 3.13.0)

project(qlen VERSION 1.0.0 LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0074 NEW)

include(cmake/utils.cmake)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20.0")
execute_process(
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/modules 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMUMPS_v5411.cmake FindMUMPS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindSCALAPACK_v5411.cmake FindSCALAPACK.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMETIS_v5411.cmake FindMETIS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindScotch_v5411.cmake FindScotch.cmake 
    RESULT_VARIABLE result)
    set(MUMPS_OK TRUE)
    set(MUMPS_VERS_STR "v5.4.1.1")
    set(SCALAPACK_VERS_STR "v2.1.0.27")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19.0")
execute_process(
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/modules 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMUMPS_v5407.cmake FindMUMPS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindSCALAPACK_v5407.cmake FindSCALAPACK.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMETIS_v5407.cmake FindMETIS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindScotch_v5407.cmake FindScotch.cmake 
    RESULT_VARIABLE result)
    set(MUMPS_OK TRUE)
    set(MUMPS_VERS_STR "v5.4.0.7")
    set(SCALAPACK_VERS_STR "v2.1.0.25")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.13.0")
execute_process(
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/modules 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMUMPS_v5400.cmake FindMUMPS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindSCALAPACK_v5400.cmake FindSCALAPACK.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindMETIS_v5400.cmake FindMETIS.cmake 
    COMMAND ${CMAKE_COMMAND} -E create_symlink FindScotch_v5400.cmake FindScotch.cmake 
    RESULT_VARIABLE result)
    set(MUMPS_OK FALSE)
    set(MUMPS_VERS_STR "v5.4.0.0")
    set(SCALAPACK_VERS_STR "v2.1.0.19")
    message(STATUS "${Red}For cmake versions < 3.19 MUMPS will be excluded from 'make deps'.${ColourReset}")
else()
    message(STATUS "MUMPS needs cmake version greater or equal to 3.13")
endif()

if (NOT ${result} STREQUAL "0")
    message(SEND_ERROR "Cannot make symlink, mesg=${result}")
endif()

set(QLENS_COMPILE_FLAGS
    "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS} -Wno-write-strings -std=c++11"
)

set(QLENS_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

set(QLENS_COMPILE_DEFINITIONS 
)

set(QLENS_INCLUDE_DIRS
    ${CMAKE_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/include
)
    
#set(QLENS_LIBRARIES )

set(QLENS_HEADERS
)

set(QLENS_CORE_SOURCES 
    ${PROJECT_SOURCE_DIR}/src/profile.cpp 
    ${PROJECT_SOURCE_DIR}/src/sbprofile.cpp 
    ${PROJECT_SOURCE_DIR}/src/models.cpp 
    ${PROJECT_SOURCE_DIR}/src/lens.cpp 
    ${PROJECT_SOURCE_DIR}/src/imgsrch.cpp 
    ${PROJECT_SOURCE_DIR}/src/pixelgrid.cpp
    ${PROJECT_SOURCE_DIR}/src/cg.cpp 
    ${PROJECT_SOURCE_DIR}/src/mcmchdr.cpp 
    ${PROJECT_SOURCE_DIR}/src/errors.cpp 
    ${PROJECT_SOURCE_DIR}/src/brent.cpp 
    ${PROJECT_SOURCE_DIR}/src/sort.cpp 
    ${PROJECT_SOURCE_DIR}/src/gauss.cpp 
    ${PROJECT_SOURCE_DIR}/src/romberg.cpp 
    ${PROJECT_SOURCE_DIR}/src/spline.cpp
    ${PROJECT_SOURCE_DIR}/src/trirectangle.cpp 
    ${PROJECT_SOURCE_DIR}/src/GregsMathHdr.cpp 
    ${PROJECT_SOURCE_DIR}/src/hyp_2F1.cpp 
    ${PROJECT_SOURCE_DIR}/src/cosmo.cpp
    ${PROJECT_SOURCE_DIR}/src/simplex.cpp 
    ${PROJECT_SOURCE_DIR}/src/powell.cpp 
    ${PROJECT_SOURCE_DIR}/src/mcmceval.cpp
    #${PROJECT_SOURCE_DIR}/cmake/dummy.f
)

set(QLENS_SOURCES 
    ${QLENS_CORE_SOURCES}
    ${PROJECT_SOURCE_DIR}/src/qlens.cpp 
    #${PROJECT_SOURCE_DIR}/src/commands.cpp
    $<TARGET_OBJECTS:commands_cpp>
)

set(MKDIST_SOURCES
    ${PROJECT_SOURCE_DIR}/src/mkdist.cpp
    ${PROJECT_SOURCE_DIR}/src/GregsMathHdr.cpp 
    ${PROJECT_SOURCE_DIR}/src/errors.cpp 
    ${PROJECT_SOURCE_DIR}/src/mcmceval.cpp
)

set(COSMOCALC_SOURCES
    ${PROJECT_SOURCE_DIR}/src/cosmocalc.cpp
    ${PROJECT_SOURCE_DIR}/src/errors.cpp 
    ${PROJECT_SOURCE_DIR}/src/spline.cpp 
    ${PROJECT_SOURCE_DIR}/src/romberg.cpp 
    ${PROJECT_SOURCE_DIR}/src/cosmo.cpp 
    ${PROJECT_SOURCE_DIR}/src/brent.cpp
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

message("")
message("*** System ***")
message(STATUS "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}")
message(STATUS "CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}")

message("")
MESSAGE("** OpenMP ***")
find_package(OpenMP)
if(OPENMP_FOUND)
    message(STATUS "OPENMP_FOUND = ${OPENMP_FOUND}")
    message(STATUS "OpenMP_C_FLAGS = ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
    message(STATUS "OpenMP_CXX_INCLUDE_DIRS = ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP_C_LIBRARIES = ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP_CXX_LIBRARIES = ${OpenMP_CXX_LIBRARIES}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_OPENMP")
    set(QLENS_COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} ${OpenMP_C_FLAGS}")
    set(QLENS_COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${OpenMP_C_INCLUDE_DIRS})
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${OpenMP_C_LIBRARIES})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
    set (QLENS_LINKER_FLAGS "${QLENS_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    message(STATUS "${Red}OpenMP not found${ColourReset}")
endif(OPENMP_FOUND)

MESSAGE("")
MESSAGE("*** Python ***")
if(${CMAKE_VERSION} VERSION_GREATER "3.10.2" AND NOT DEFINED PYTHON_EXECUTABLE)
	find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
	set(PYTHONLIBS_VERSION_STRING ${Python_VERSION})
else()
    # Force rediscovery of python every time
    unset(PYTHON_INCLUDE_DIRS CACHE)
    unset(PYTHON_LIBRARIES CACHE)
    find_package(PythonInterp)
    find_package(PythonLibs)
    find_package(NumPy)
    set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
    set(Python_LIBRARIES ${PYTHON_LIBRARIES})
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(Python_VERSION ${PYTHON_VERSION_STRING})
    set(Python_NumPy_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})
    set(Python_NumPy_VERSION ${NUMPY_VERSION_STRING})
endif()

MESSAGE( STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}" )
MESSAGE( STATUS "Python_VERSION = ${Python_VERSION}" )
MESSAGE( STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}" )
MESSAGE( STATUS "Python_LIBRARIES = ${Python_LIBRARIES}" )
MESSAGE( STATUS "PYTHONLIBS_VERSION_STRING = ${PYTHONLIBS_VERSION_STRING}" )
MESSAGE( STATUS "Python_NumPy_INCLUDE_DIRS = ${Python_NumPy_INCLUDE_DIRS}")
MESSAGE( STATUS "Python_NumPy_VERSION = ${Python_NumPy_VERSION}")

MESSAGE("")
message("*** MPI ***")
if(NOT MPI_DISABLE)
    find_package(MPI)
    if(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)
        message(STATUS "MPI_C_FOUND = ${MPI_C_FOUND}" )
        message(STATUS "MPI_C_INCLUDE_PATH = ${MPI_C_INCLUDE_PATH}" )
        message(STATUS "MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}" )
        message(STATUS "MPI_C_COMPILE_FLAGS = ${MPI_C_COMPILE_FLAGS}" )
        message(STATUS "MPI_CXX_FOUND = ${MPI_CXX_FOUND}" )
        message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}" )
        message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}" )
        message(STATUS "MPI_CXX_COMPILE_FLAGS = ${MPI_CXX_COMPILE_FLAGS}" )
        message(STATUS "MPI_Fortran_FOUND = ${MPI_Fortran_FOUND}" )
        message(STATUS "MPI_Fortran_INCLUDE_PATH = ${MPI_Fortran_INCLUDE_PATH}" )
        message(STATUS "MPI_Fortran_LIBRARIES = ${MPI_Fortran_LIBRARIES}" )
        message(STATUS "MPI_Fortran_COMPILE_FLAGS = ${MPI_Fortran_COMPILE_FLAGS}" )
        # compile with MPI if found
        set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_MPI")
        if(MPI_C_FOUND)
            set(QLENS_COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS}")
            set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
            set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${MPI_C_LIBRARIES})
        endif(MPI_C_FOUND)
        if(MPI_CXX_FOUND)
            set(QLENS_COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
            set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
            set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${MPI_CXX_LIBRARIES})
        endif(MPI_CXX_FOUND)
        if(MPI_Fortran_FOUND)
            set(MULTINEST_USE_MPI TRUE)
            #set(QLENS_COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} ${MPI_Frotran_COMPILE_FLAGS}")
            #set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${MPI_Fortran_INCLUDE_PATH})
            set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${MPI_Fortran_LIBRARIES})
        endif(MPI_Fortran_FOUND)
    else(MPI_C_FOUND OR MPI_CXX_FOUND)
        message(STATUS "${Red}MPI not found${ColourReset}")
    endif(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)
endif(NOT MPI_DISABLE)

message("")
message("*** TCMalloc ***")
find_library(TCMALLOC_LIBRARY
    NAMES tcmalloc
    HINTS ${TCMALLOC_ROOT_DIR}
    DOC "TCMalloc library"
)
if(NOT TCMALLOC_LIBRARY STREQUAL "TCMALLOC_LIBRARY-NOTFOUND")
    message(STATUS "TCMALLOC_LIBRARY = ${TCMALLOC_LIBRARY}")
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${TCMALLOC_LIBRARY})
else(NOT TCMALLOC_LIBRARY STREQUAL "TCMALLOC_LIBRARY-NOTFOUND")
    message(STATUS "${Red}TCMALLOC not found${ColourReset}")
endif(NOT TCMALLOC_LIBRARY STREQUAL "TCMALLOC_LIBRARY-NOTFOUND")

message("")
message("*** CFitsio ***")
find_package(CFITSIO)
if(CFITSIO_FOUND)
    message(STATUS "CFITSIO_FOUND = ${CFITSIO_FOUND}")
    message(STATUS "CFITSIO_INCLUDE_DIR = ${CFITSIO_INCLUDE_DIR}")
    message(STATUS "CFITSIO_LIBRARY = ${CFITSIO_LIBRARY}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_FITS")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${CFITSIO_LIBRARY})
else(CFITSIO_FOUND)
    message(STATUS "${Red}CFitsio not found${ColourReset}")
endif(CFITSIO_FOUND)

message("")
message("*** Readline ***")
find_package("Readline")
if(READLINE_FOUND)
    message(STATUS "READLINE_FOUND = ${READLINE_FOUND}")
    message(STATUS "Readline_INCLUDE_DIR = ${Readline_INCLUDE_DIR}")
    message(STATUS "Readline_LIBRARY = ${Readline_LIBRARY}")

    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_READLINE")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${Readline_INCLUDE_DIR})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${Readline_LIBRARY})
else(READLINE_FOUND)
    message(STATUS "${Red}Readline not found${ColourReset}")
endif(READLINE_FOUND)

message("")
message("*** Umfpack ***")
find_package(UMFPACK)
if(UMFPACK_FOUND)
    message(STATUS "UMFPACK_FOUND = ${UMFPACK_FOUND}")
    message(STATUS "UMFPACK_INCLUDES = ${UMFPACK_INCLUDES}")
    message(STATUS "UMFPACK_LIBRARIES = ${UMFPACK_LIBRARIES}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_UMFPACK")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${UMFPACK_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${UMFPACK_LIBRARIES})
else(UMFPACK_FOUND)
    message(STATUS "${Red}Umfpack not found${ColourReset}")
endif(UMFPACK_FOUND)

message("")
message("*** LAPACK ***")
find_package(LAPACK)
if(LAPACK_FOUND)
    message(STATUS "LAPACK_FOUND = ${LAPACK_FOUND}")
    message(STATUS "LAPACK95_FOUND = ${LAPACK_FOUND}")
    message(STATUS "LAPACK_LINKER_FLAGS = ${LAPACK_INCLUDES}")
    message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
    message(STATUS "LAPACK95_LIBRARIES = ${LAPACK_LIBRARIES}")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
    set(QLENS_LINKER_FLAGS ${QLENS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACK95_LIBRARIES})
else(LAPACK_FOUND)
    message(STATUS "${Red}LAPACK not found${ColourReset}")
endif(LAPACK_FOUND)

message("")
message("*** MUMPS ***")
#unset(MUMPS_ROOT CACHE)
set(MUMPS_ROOT ${MUMPS_ROOT} ${PROJECT_SOURCE_DIR}/contrib/mumps)
unset(MUMPS_FOUND CACHE)
unset(MUMPS_INCLUDE_DIRS CACHE)
unset(UMPS_LIBRARIES CACHE)
unset(MUMPS_INCLUDE_DIR CACHE)
unset(MUMPS_LIBRARY CACHE)
find_package(MUMPS)
if(MUMPS_FOUND)
    message(STATUS "MUMPS_FOUND = ${MUMPS_FOUND}")
    message(STATUS "MUMPS_INCLUDE_DIRS = ${MUMPS_INCLUDE_DIRS}")
    message(STATUS "MUMPS_LIBRARIES = ${MUMPS_LIBRARIES}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_MUMPS")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${MUMPS_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${MUMPS_LIBRARIES})
else(MUMPS_FOUND)
    message(STATUS "${Red}MUMPS not found${ColourReset}")
endif(MUMPS_FOUND)

message("")
message("*** SCALAPACK ***")
#unset(SCALAPACK_ROOT CACHE)
set(SCALAPACK_ROOT ${SCALAPACK_ROOT} ${PROJECT_SOURCE_DIR}/contrib/scalapack)
unset(SCALAPACK_FOUND CACHE)
unset(SCALAPACK_INCLUDE_DIRS CACHE)
unset(UMPS_LIBRARIES CACHE)
unset(SCALAPACK_INCLUDE_DIR CACHE)
unset(SCALAPACK_LIBRARY CACHE)
find_package(SCALAPACK)
if(SCALAPACK_FOUND)
    message(STATUS "SCALAPACK_FOUND = ${SCALAPACK_FOUND}")
    message(STATUS "SCALAPACK_INCLUDE_DIRS = ${SCALAPACK_INCLUDE_DIRS}")
    message(STATUS "SCALAPACK_LIBRARIES = ${SCALAPACK_LIBRARIES}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_SCALAPACK")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${SCALAPACK_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${SCALAPACK_LIBRARIES})
else(SCALAPACK_FOUND)
    message(STATUS "${Red}SCALAPACK not found${ColourReset}")
endif(SCALAPACK_FOUND)

message("")
message("*** BLACS ***")
find_library(BLACS_LIBRARY
    NAMES blacs
    HINTS ${BLACS_ROOT_DIR} ${PROJECT_SOURCE_DIR}/contrib/scalapack/lib
    DOC "BLACS library"
)
if(NOT BLACS_LIBRARY STREQUAL "BLACS_LIBRARY-NOTFOUND")
    message(STATUS "BLACS_LIBRARY = ${BLACS_LIBRARY}")
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${BLACS_LIBRARY})
else(NOT BLACS_LIBRARY STREQUAL "BLACS_LIBRARY-NOTFOUND")
    message(STATUS "${Red}BLACS not found${ColourReset}")
endif(NOT BLACS_LIBRARY STREQUAL "BLACS_LIBRARY-NOTFOUND")

message("")
message("*** Polychord ***")
unset(Polychord_FOUND CACHE)
unset(Polychord_INCLUDE_DIRS CACHE)
unset(Polychord_INCLUDE_DIR CACHE)
unset(Polychord_LIBRARIES CACHE)
unset(Polychord_LIBRARY CACHE)
#set(Polychord_HINT_DIR ${Polychord_HINT_DIR} ${PROJECT_SOURCE_DIR}/contrib/polychord/1.17.1)
set(Polychord_HINT_DIR ${Polychord_HINT_DIR} ${PROJECT_SOURCE_DIR}/contrib/PolyChordLite)
find_package(Polychord)
if(Polychord_FOUND)
    message (STATUS "Polychord_FOUND = ${Polychord_FOUND}")
    message (STATUS "Polychord_INCLUDE_DIRS = ${Polychord_INCLUDE_DIRS}")
    message (STATUS "Polychord_LIBRARIES = ${Polychord_LIBRARIES}")
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_POLYCHORD")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${Polychord_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${Polychord_LIBRARIES})
else(Polychord_FOUND)
    message(STATUS "${Red}Polychord not found${ColourReset}")
endif(Polychord_FOUND)

message("")
message("*** MultiNest ***")
set(MULTINEST_HINT ${MULTINEST_HINT} ${PROJECT_SOURCE_DIR}/contrib/MultiNest/MultiNest_v3.12_CMake/multinest)
unset(_MULTINEST_INCLUDE_DIRS CACHE)
unset(MULTINEST_FOUND CACHE)
unset(MULTINEST_INCLUDE_DIRS CACHE)
unset(MULTINEST_LIBRARIES CACHE)
unset(MULTINEST_MODULE_DIRS CACHE)
unset(MULTINEST_INCLUDES CACHE)
unset(MULTINEST_ROOT_DIR CACHE)
find_package(MULTINEST)
if(MULTINEST_FOUND)
    message(STATUS "MULTINEST_INCLUDE_DIRS = ${MULTINEST_INCLUDE_DIRS}" )
    message(STATUS "MULTINEST_LIBRARIES = ${MULTINEST_LIBRARIES}" )
    message(STATUS "MULTINEST_MODULES = ${MULTINEST_MODULE_DIRS}" )
    if(LAPACK_FOUND OR LAPACK95_FOUND)
        message(STATUS "BLA_VENDOR = ${BLA_VENDOR}")
        message(STATUS "LAPACK_LINKER_FLAGS = ${LAPACK_LINKER_FLAGS}")
    endif(LAPACK_FOUND OR LAPACK95_FOUND)
    message(STATUS "LAPACK_FOUND = ${LAPACK_FOUND}")
    if(LAPACK_FOUND)
        message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
    endif(LAPACK_FOUND)
    message(STATUS "LAPACK95_FOUND = ${LAPACK95_FOUND}")
    if(LAPACK95_FOUND)
        message(STATUS "LAPACK_95LIBRARIES = ${LAPACK_95LIBRARIES}")
    endif(LAPACK95_FOUND)
    set(QLENS_COMPILE_DEFINITIONS ${QLENS_COMPILE_DEFINITIONS} "USE_MULTINEST")
    set(QLENS_INCLUDE_DIRS ${QLENS_INCLUDE_DIRS} ${MULTINEST_INCLUDE_DIRS})
    set(QLENS_LIBRARIES ${QLENS_LIBRARIES} ${MULTINEST_LIBRARIES})
else(MULTINEST_FOUND)
    message(STATUS "${Red}MultiNest not found${ColourReset}")
endif(MULTINEST_FOUND)
message("")

include(cmake/polychord.cmake)
include(cmake/multinest.cmake)
include(cmake/scalapack.cmake)
include(cmake/mumps.cmake)
include(cmake/pybind11.cmake)
include(cmake/python.cmake)
include(cmake/executables.cmake)

if(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)
    add_dependencies(mumps scalapack)
    add_dependencies(mumps-distclean scalapack-distclean)
endif(MPI_C_FOUND OR MPI_CXX_FOUND OR MPI_Fortran_FOUND)

add_custom_target(qlens-all)
add_dependencies(qlens-all qlens mkdist cosmocalc python)

add_custom_target(deps)
add_dependencies(deps pybind11 multinest polychord)

add_custom_target(deps-distclean)
add_dependencies(deps-distclean pybind11-distclean multinest-distclean polychord-distclean)

if(MUMPS_OK)
    add_dependencies(deps mumps)
    add_dependencies(deps-distclean mumps-distclean)
endif(MUMPS_OK)
