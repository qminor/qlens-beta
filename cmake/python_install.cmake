
set(PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
set(PYTHON_RUN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}") # Final package installation path
#set(LIB_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/qlens") # Temporary build path (unused?)

#--- Post-build copying
# Ok the build seems to be working, but now I need to copy all the scanner stuff
# into the final python package install directory.
# There might be a "cleaner" CMake way to do this... I will do it this "dumb"
# way for now.

add_custom_target(python_install)

# Copies stuff after scan_python target is built
set(copy_target python_install)
function(copy_file in out)
   message("Generating post-build instructions to copy ${in} to ${out}")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${in} to ${out}...")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out})
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${in} ${out})
endfunction(copy_file)

function(copy_dir in out)
   message("Generating post-build instructions to copy ${in} to ${out}")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${in} to ${out}...")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out})
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${in} ${out})
endfunction(copy_dir)

# Copy the ScannerBit 'python' directory that will contain the main 'ScannerBit.so' python-usable library
if (DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    copy_dir(${PYTHON_DIR}/qlens
             ${PYTHON_RUN_DIR})
endif (DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
