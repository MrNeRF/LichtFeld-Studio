# Look for lp_solve: https://lpsolve.sourceforge.net/5.5/index.htm
if(NOT TARGET lpsolve::lpsolve)
set(LPSOLVE_DIR $ENV{LPSOLVE_DIR} CACHE PATH "lpsolve directory (contain include/lp_lib.h).")

find_path(LPSOLVE_INCLUDE_DIR
          NAMES lp_lib.h
          PATHS ${LPSOLVE_DIR}
                ${LPSOLVE_DIR}/include
                /usr/local/include
                /usr/include
                ${VS_SEARCH_PATH}/lpsolve-5.5.2.11/include
                ~/sw/lpsolve-5.5.2.11/include
          )

find_library(LPSOLVE_LIBRARY
             NAMES lpsolve55
             HINTS ${LPSOLVE_DIR}
                   ${LPSOLVE_DIR}/lib
                   ${VS_SEARCH_PATH}/lpsolve-5.5.2.11/lib
                   ~/sw/lpsolve-5.5.2.11/lib
             PATHS /usr/local/lib
                   /usr/lib
            )

get_filename_component(LPSOLVE_LIBRARY_DIR ${LPSOLVE_LIBRARY} DIRECTORY CACHE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LPSolve DEFAULT_MSG LPSOLVE_INCLUDE_DIR LPSOLVE_LIBRARY LPSOLVE_LIBRARY_DIR)

if(LPSOLVE_FOUND AND NOT TARGET lpsolve::lpsolve)
    add_library(lpsolve::lpsolve INTERFACE IMPORTED)
    target_include_directories(lpsolve::lpsolve INTERFACE ${LPSOLVE_INCLUDE_DIR})
    target_link_directories(lpsolve::lpsolve INTERFACE ${LPSOLVE_LIBRARY_DIR})
    target_link_libraries(lpsolve::lpsolve INTERFACE ${LPSOLVE_LIBRARY})
endif()

mark_as_advanced(LPSOLVE_INCLUDE_DIR)
mark_as_advanced(LPSOLVE_LIBRARY)

endif(NOT TARGET lpsolve::lpsolve)
