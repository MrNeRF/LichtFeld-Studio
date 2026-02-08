if (NOT TARGET MPFI::MPFI)

find_path(MPFI_INCLUDE_DIR
    NAMES mpfi.h
    PATHS /usr/local/include
    /usr/include
    )

find_library(MPFI_LIBRARY
    NAMES mpfi
    PATHS /usr/local/lib
    /usr/lib
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFI
    REQUIRED_VARS MPFI_INCLUDE_DIR MPFI_LIBRARY)

if(MPFI_FOUND AND NOT TARGET MPFI::MPFI)
    add_library(MPFI::MPFI INTERFACE IMPORTED)
    target_include_directories(MPFI::MPFI INTERFACE ${MPFI_INCLUDE_DIR})
    target_link_libraries(MPFI::MPFI INTERFACE ${MPFI_LIBRARY})
endif()

mark_as_advanced(MPFI_INCLUDE_DIR)
mark_as_advanced(MPFI_LIBRARY)

endif (NOT TARGET MPFI::MPFI)
