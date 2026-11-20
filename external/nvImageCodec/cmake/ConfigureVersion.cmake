# Configures project version in library headers based on project version
function(configure_library_version libname input_file output_file)
    # Two decimal digits each, so packed versions stay unique and ordered
    if(PROJECT_VERSION_MINOR GREATER 99 OR PROJECT_VERSION_PATCH GREATER 99)
        message(FATAL_ERROR
            "Version ${PROJECT_VERSION} cannot be packed: minor and patch must be below 100. "
            "Widen NVIMGCODEC_MAKE_SEMANTIC_VERSION in include/nvimgcodec_version.h.in first.")
    endif()

    # Must match NVIMGCODEC_MAKE_SEMANTIC_VERSION in include/nvimgcodec_version.h.in
    math(EXPR TEMP_RESULT "${PROJECT_VERSION_MAJOR} * 10000 + ${PROJECT_VERSION_MINOR} * 100 + ${PROJECT_VERSION_PATCH}")
    set(${libname}_PROJECT_VERSION_NUMBER ${TEMP_RESULT})

    # Configure a header file to pass project version
    configure_file(
        "${input_file}"
        "${output_file}"
        @ONLY
    )
endfunction(configure_library_version)
