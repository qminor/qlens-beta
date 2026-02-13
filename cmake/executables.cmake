add_library(commands_cpp OBJECT ${PROJECT_SOURCE_DIR}/src/commands.cpp)

add_library(fitpack OBJECT ${PROJECT_SOURCE_DIR}/contrib/fitpack/curfit.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpback.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpbspl.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpchec.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpcurf.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpdisc.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpgivs.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fpknot.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fprati.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/fprota.f  ${PROJECT_SOURCE_DIR}/contrib/fitpack/splev.f)


set_target_properties(
    commands_cpp
    PROPERTIES
    INCLUDE_DIRECTORIES "${QLENS_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS "${QLENS_COMPILE_DEFINITIONS}"
    COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} -O0"
    #LINK_LIBRARIES ${GDP_LIBRARIES}
    LINK_FLAGS "${QLENS_LINKER_FLAGS}"
    LINKER_LANGUAGE CXX
)

set_target_properties(
    fitpack
    PROPERTIES
    #INCLUDE_DIRECTORIES "${QLENS_INCLUDE_DIRS}"
    #COMPILE_DEFINITIONS "${QLENS_COMPILE_DEFINITIONS}"
    COMPILE_FLAGS "${CMAKE_Fortran_FLAGS}"
    #LINK_LIBRARIES ${GDP_LIBRARIES}
    LINK_FLAGS "${QLENS_LINKER_FLAGS}"
    LINKER_LANGUAGE Fortran
)

add_executable(qlens
    ${QLENS_SOURCES}
)

set_target_properties(
    qlens 
    PROPERTIES
    INCLUDE_DIRECTORIES "${QLENS_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS "${QLENS_COMPILE_DEFINITIONS}"
    COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} -O3"
    #LINK_LIBRARIES ${GDP_LIBRARIES}
    LINK_FLAGS "${QLENS_LINKER_FLAGS}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    LINKER_LANGUAGE Fortran
    OUTPUT_NAME "qlens"
)

add_dependencies(qlens commands_cpp cmake_cmd)

target_link_libraries(qlens ${QLENS_LIBRARIES})

add_executable(mkdist
    ${MKDIST_SOURCES}
)

set_target_properties(
    mkdist 
    PROPERTIES
    INCLUDE_DIRECTORIES "${QLENS_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS "${QLENS_COMPILE_DEFINITIONS}"
    COMPILE_FLAGS "${QLENS_COMPILE_FLAGS} -O3"
    #LINK_LIBRARIES ${GDP_LIBRARIES}
    LINK_FLAGS "${QLENS_LINKER_FLAGS}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    LINKER_LANGUAGE CXX
    OUTPUT_NAME "mkdist"
)

target_link_libraries(mkdist ${QLENS_LIBRARIES})

