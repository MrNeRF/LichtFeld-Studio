if(NOT TARGET GMSH::GMSH)

set(GMSH_DIR $ENV{GMSH_DIR} CACHE PATH "GMSH directory (containing include/gmsh.h).")

find_path(GMSH_INCLUDE_DIR
           NAMES gmsh.h
           HINTS ${GMSH_DIR}
                 ${GMSH_DIR}/include
           PATHS /usr/local/include
                 /usr/include
          )

find_library(GMSH_LIBRARY
           NAMES gmsh
           HINTS ${GMSH_DIR}
                 ${GMSH_DIR}/lib
           PATHS /usr/local/lib
                 /usr/lib
          )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMSH DEFAULT_MSG GMSH_INCLUDE_DIR GMSH_LIBRARY)

if(GMSH_FOUND AND NOT TARGET GMSH::GMSH)
  add_library(GMSH::GMSH INTERFACE IMPORTED)
  target_include_directories(GMSH::GMSH INTERFACE ${GMSH_INCLUDE_DIR})
  target_link_libraries(GMSH::GMSH INTERFACE ${GMSH_LIBRARY})
endif()

mark_as_advanced(GMSH_INCLUDE_DIR)
mark_as_advanced(GMSH_LIBRARY)


endif(NOT TARGET GMSH::GMSH)
