if (NOT TARGET MPFR::MPFR)
find_path(MPFR_INCLUDE_DIR
    NAMES mpfr.h
    PATHS /usr/local/include
    /usr/include
    )

find_library(MPFR_LIBRARY
    NAMES mpfr
    PATHS /usr/local/lib
    /usr/lib
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR
    REQUIRED_VARS MPFR_INCLUDE_DIR MPFR_LIBRARY)

if(MPFR_FOUND AND NOT TARGET MPFR::MPFR)
    add_library(MPFR::MPFR INTERFACE IMPORTED)
    target_include_directories(MPFR::MPFR INTERFACE ${MPFR_INCLUDE_DIR})
    target_link_libraries(MPFR::MPFR INTERFACE ${MPFR_LIBRARY})
endif()

mark_as_advanced(MPFR_INCLUDE_DIR)
mark_as_advanced(MPFR_LIBRARY)

endif (NOT TARGET MPFR::MPFR)
