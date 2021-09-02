include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

ExternalProject_Add(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_PROGRESS TRUE
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/contrib/pybind11"
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )

add_custom_target(pybind11-distclean 
                                COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/contrib/pybind11" 
                                COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/pybind11-prefix" 
                                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
