if (NOT TARGET GMM::GMM)

set(GMM_DIR $ENV{GMM_DIR} CACHE PATH "GMM directory (contain gmm/gmm.h).")

find_path(GMM_INCLUDE_DIR
           NAMES gmm/gmm.h
           PATHS ${GMM_DIR}
                 ${GMM_DIR}/include
                 /usr/local/include
                 /usr/include
          )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMM DEFAULT_MSG GMM_INCLUDE_DIR)

if(GMM_FOUND AND NOT TARGET GMM::GMM)
    add_library(GMM::GMM INTERFACE IMPORTED)
    target_include_directories(GMM::GMM INTERFACE ${GMM_INCLUDE_DIR})
    if(MSVC)
        target_compile_definitions(GMM::GMM INTERFACE "_SCL_SECURE_NO_DEPRECATE")
    endif()
endif()

mark_as_advanced(GMM_INCLUDE_DIR)

endif(NOT TARGET GMM::GMM)
