# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)

function(lfs_hip_offload_compression_flags output compiler)
    option(LFS_HIP_OFFLOAD_COMPRESSION "Compress HIP device bundles with the selected SDK compiler" OFF)
    set(_flags)
    if(LFS_HIP_OFFLOAD_COMPRESSION)
        execute_process(COMMAND "${compiler}" --help RESULT_VARIABLE _result
            OUTPUT_VARIABLE _help ERROR_VARIABLE _error TIMEOUT 20)
        if(NOT _result STREQUAL "0" OR NOT _help MATCHES "--offload-compress")
            message(FATAL_ERROR "HIP offload compression requires a compiler supporting --offload-compress: ${compiler}. Disable LFS_HIP_OFFLOAD_COMPRESSION to preserve the uncompressed build.")
        endif()
        list(APPEND _flags --offload-compress)
    endif()
    set(${output} "${_flags}" PARENT_SCOPE)
endfunction()
