# defining some colors
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")

# Simple function to find specific Python modules
macro(find_python_module module)
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import ${module}" RESULT_VARIABLE return_value ERROR_QUIET)
if (NOT return_value)
    set(Python_${module}_FOUND TRUE)
    #message(STATUS "Found Python module ${module}.")
  else()
    if(${ARGC} GREATER 1 AND ${ARGV1} STREQUAL "REQUIRED")
      message(FATAL_ERROR "FAILED to find Python module ${module}.")
    else()
      message(STATUS "FAILED to find Python module ${module}.") #-- FAILED ...
    endif()
  endif()
endmacro()

# Crash function for failed execute_processes
function(check_result result command)
  if(NOT ${result} STREQUAL "0")
    message(FATAL_ERROR "${BoldRed}cmake failed because ${command} did not return 0.  Culprit: ${command}${ColourReset}")
  endif()
endfunction()

add_custom_target(cmake_cmd
    COMMAND ${CMAKE_COMMAND} ${PROJECT_SOURCE_DIR}
)
