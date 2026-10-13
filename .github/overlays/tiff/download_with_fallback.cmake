# vcpkg_download_distfile fails fatally; use its download-mode runner for the
# optional primary attempt, then the normal helper for verification and SPDX.
function(lfs_download_with_fallback out_var)
    cmake_parse_arguments(PARSE_ARGV 1 arg ""
        "PRIMARY_URL;PRIMARY_FILENAME;PRIMARY_SHA512;MIRROR_URL;MIRROR_FILENAME;MIRROR_SHA512" "")
    foreach(field PRIMARY_URL PRIMARY_FILENAME PRIMARY_SHA512 MIRROR_URL MIRROR_FILENAME MIRROR_SHA512)
        if(NOT DEFINED arg_${field})
            message(FATAL_ERROR "Missing ${field}")
        endif()
    endforeach()

    set(primary "${DOWNLOADS}/${arg_PRIMARY_FILENAME}")
    set(primary_valid OFF)
    if(EXISTS "${primary}")
        file(SHA512 "${primary}" actual_hash)
        if(actual_hash STREQUAL arg_PRIMARY_SHA512)
            set(primary_valid ON)
        endif()
    endif()

    if(NOT primary_valid AND NOT _VCPKG_NO_DOWNLOADS)
        file(MAKE_DIRECTORY "${DOWNLOADS}")
        vcpkg_execute_in_download_mode(
            COMMAND "$ENV{VCPKG_COMMAND}" x-download "${arg_PRIMARY_FILENAME}"
                "--url=${arg_PRIMARY_URL}" "--sha512=${arg_PRIMARY_SHA512}"
            WORKING_DIRECTORY "${DOWNLOADS}"
            RESULT_VARIABLE download_result
        )
        if("${download_result}" STREQUAL "0" AND EXISTS "${primary}")
            file(SHA512 "${primary}" actual_hash)
            if(actual_hash STREQUAL arg_PRIMARY_SHA512)
                set(primary_valid ON)
            endif()
        endif()
    endif()

    if(primary_valid)
        vcpkg_download_distfile(archive
            URLS "${arg_PRIMARY_URL}" FILENAME "${arg_PRIMARY_FILENAME}"
            SHA512 "${arg_PRIMARY_SHA512}")
    else()
        message(STATUS "Primary archive unavailable or invalid; trying verified release mirror.")
        # This remains fatal if the mirror also fails. The standard helper
        # honours offline mode, asset caches and SHA512 verification.
        vcpkg_download_distfile(archive
            URLS "${arg_MIRROR_URL}" FILENAME "${arg_MIRROR_FILENAME}"
            SHA512 "${arg_MIRROR_SHA512}")
    endif()
    set("${out_var}" "${archive}" PARENT_SCOPE)
endfunction()
