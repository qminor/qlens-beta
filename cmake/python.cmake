SET(QLENS_PYTHON_DEFINITIONS)

FOREACH(item ${QLENS_COMPILE_DEFINITIONS})
    IF (NOT ${item} STREQUAL "USE_MPI" AND NOT ${item} STREQUAL "USE_MUMPS")
        SET(QLENS_PYTHON_DEFINITIONS ${QLENS_PYTHON_DEFINITIONS} ${item})
    ENDIF (NOT ${item} STREQUAL "USE_MPI" AND NOT ${item} STREQUAL "USE_MUMPS")
ENDFOREACH(item ${QLENS_DEFINITIONS}})

SET(QLENS_PYTHON_SOURCE_FILES   
    ${QLENS_CORE_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qlens_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qlens_export.cpp
)

SET(QLENS_PYTHON_HEADER_FILES
)

SET(QLENS_PYTHON_INCLUDE_DIRS
    ${QLENS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/pybind11/include
    ${Python_INCLUDE_DIRS}
)

SET(QLENS_PYTHON_LIBRARIES
    ${QLENS_LIBRARIES}
    #${Python_LIBRARIES}
)

IF(EBUG)
    message("QLENS_PYTHON_SOURCE_FILES: ${QLENS_PYTHON_SOURCE_FILES}")
    message("QLENS_PYTHON_INCLUDE_DIRS: ${QLENS_PYTHON_INCLUDE_DIRS}")
    message("QLENS_PYTHON_LIBRARIES: ${QLENS_PYTHON_LIBRARIES}\n")
ENDIF(EBUG)

ADD_LIBRARY(python SHARED ${QLENS_PYTHON_SOURCE_FILES})

TARGET_LINK_LIBRARIES( python ${QLENS_PYTHON_LIBRARIES})

IF(APPLE)
    SET_TARGET_PROPERTIES( python
                            PROPERTIES
                            COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} -O3"
                            INCLUDE_DIRECTORIES "${QLENS_PYTHON_INCLUDE_DIRS}"
                            COMPILE_DEFINITIONS "${QLENS_PYTHON_DEFINITIONS}"
                            LINK_FLAGS "${QLENS_LINKER_FLAGS} -undefined dynamic_lookup"
                            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/qlens"
                            LIBRARY_OUTPUT_NAME "qlens"
                            LINKER_LANGUAGE Fortran
                            PREFIX ""
                            SUFFIX ".so")
ELSE(APPLE)
    SET_TARGET_PROPERTIES( python
                            PROPERTIES
                            COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} -O3"
                            INCLUDE_DIRECTORIES "${QLENS_PYTHON_INCLUDE_DIRS}"
                            COMPILE_DEFINITIONS "${QLENS_PYTHON_DEFINITIONS}"
                            LINK_FLAGS "${QLENS_LINKER_FLAGS}"
                            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/qlens"
                            LIBRARY_OUTPUT_NAME "qlens"
                            LINKER_LANGUAGE Fortran
                            PREFIX ""
                            SUFFIX ".so")
ENDIF(APPLE)

ADD_DEPENDENCIES(python cmake_cmd)
