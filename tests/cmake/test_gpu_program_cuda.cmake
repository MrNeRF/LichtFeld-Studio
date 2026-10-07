# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/GpuProgramCuda.cmake")

function(check_architectures selection expected)
    set(CMAKE_CUDA_ARCHITECTURES "${selection}")
    lfs_gpu_program_cuda_flags(actual)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${selection}: expected '${expected}', got '${actual}'")
    endif()
endfunction()

check_architectures("89" "--generate-code=arch=compute_89,code=[sm_89,compute_89]")
check_architectures("86;89" "--generate-code=arch=compute_86,code=[sm_86,compute_86];--generate-code=arch=compute_89,code=[sm_89,compute_89]")
check_architectures("86-real;90-virtual" "--generate-code=arch=compute_86,code=[sm_86];--generate-code=arch=compute_90,code=[compute_90]")
check_architectures("90a" "--generate-code=arch=compute_90a,code=[sm_90a,compute_90a]")
check_architectures("89;89" "--generate-code=arch=compute_89,code=[sm_89,compute_89]")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "89-real")
check_architectures("native" "--generate-code=arch=compute_89,code=[sm_89]")
set(CMAKE_CUDA_ARCHITECTURES_ALL "80-real;86-real;90")
check_architectures("all" "--generate-code=arch=compute_80,code=[sm_80];--generate-code=arch=compute_86,code=[sm_86];--generate-code=arch=compute_90,code=[sm_90,compute_90]")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "80-real;90")
check_architectures("all-major" "--generate-code=arch=compute_80,code=[sm_80];--generate-code=arch=compute_90,code=[sm_90,compute_90]")
message(STATUS "CUDA GPU program architecture contracts: 8 passed")
