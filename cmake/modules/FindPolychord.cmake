# - Try to find Polychord.
# Once executed, this module will define:
# Variables defined by this module:
#  Polychord_FOUND        - system has Polychord
#  Polychord_INCLUDE_DIR  - the Polychord include directory (cached)
#  Polychord_INCLUDE_DIRS - the Polychord include directories
#                         (identical to Polychord_INCLUDE_DIR)
#  Polychord_LIBRARY      - the Polychord library (cached)
#  Polychord_LIBRARIES    - the Polychord libraries
#                         (identical to Polychord_LIBRARY)
# 
# This module will use the following enviornmental variable
# when searching for Polychord:
#  Polychord_ROOT_DIR     - Polychord root directory
#

# 
#  Copyright (c) 2012 Brian Kloppenborg
# 
#  This file is part of the C++ OIFITS Library (CCOIFITS).
#  
#  CCOIFITS is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License 
#  as published by the Free Software Foundation, either version 3 
#  of the License, or (at your option) any later version.
#  
#  CCOIFITS is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#  
#  You should have received a copy of the GNU Lesser General Public 
#  License along with CCOIFITS.  If not, see <http://www.gnu.org/licenses/>.
# 

if(NOT Polychord_FOUND)

    find_path(Polychord_INCLUDE_DIR interfaces.h
        HINTS ${Polychard_HINT_DIR}
        PATH_SUFFIXES polychord src/polychord)
    find_library(Polychord_LIBRARY chord
        HINTS ${Polychard_HINT_DIR}
        PATH_SUFFIXES lib)
  
    mark_as_advanced(Polychord_INCLUDE_DIR Polychord_LIBRARY)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Polychord DEFAULT_MSG
        Polychord_LIBRARY Polychord_INCLUDE_DIR)

    set(Polychord_INCLUDE_DIRS ${Polychord_INCLUDE_DIR})
    set(Polychord_LIBRARIES ${Polychord_LIBRARY})

endif(NOT Polychord_FOUND)
