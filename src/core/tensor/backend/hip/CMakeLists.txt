# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Independent of the application and its CXX/CUDA compiler configuration.
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The native HIP runtime does not support cross-compilation")
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
    message(FATAL_ERROR "The native HIP runtime supports Windows and Linux")
endif()
set(LFS_HIP_ROOT "" CACHE PATH "Explicit ROCm development SDK root")
set(LFS_HIP_ARCHITECTURE "" CACHE STRING "Explicit HIP GPU target, e.g. gfx1150")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/HipArchitectures.cmake")
lfs_hip_architectures()
if(WIN32)
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_library "${CMAKE_BINARY_DIR}/lfs_hip_runtime.dll")
else()
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_library "${CMAKE_CURRENT_BINARY_DIR}/liblfs_hip_runtime.so")
    set(_compile_platform_flags -fPIC)
    set(_platform_flags "-Wl,-rpath,${LFS_HIP_ROOT}/lib" "-Wl,-rpath,${LFS_HIP_ROOT}/lib64")
endif()
if(NOT EXISTS "${_compiler}" OR NOT EXISTS "${LFS_HIP_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "LFS_HIP_ROOT must identify a complete ROCm development SDK")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_pointwise_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/core/detail")
set(_pointwise_generated "${CMAKE_CURRENT_BINARY_DIR}/pointwise-generated")
set(_pointwise_outputs "${_pointwise_generated}/hip_pointwise_functors.hpp"
    "${_pointwise_generated}/hip_pointwise_f32.inc" "${_pointwise_generated}/hip_pointwise_f16.inc"
    "${_pointwise_generated}/hip_pointwise_flags.inc" "${_pointwise_generated}/hip_pointwise_eval.inc"
    "${_pointwise_generated}/hip_pointwise_broadcast.inc" "${_pointwise_generated}/hip_pointwise_half_broadcast.inc"
    "${_pointwise_generated}/hip_pointwise_scalar_left.inc")
add_custom_command(OUTPUT ${_pointwise_outputs}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_pointwise.py"
        --source "${_pointwise_source}" --output "${_pointwise_generated}"
    DEPENDS prepare_pointwise.py "${_pointwise_source}/tensor_functors.hpp" "${_pointwise_source}/pointwise_ops.def" "${_pointwise_source}/pointwise_lowering.hpp"
    VERBATIM)
set(LFS_HIP_OFFLOAD_JOBS 2 CACHE STRING "Maximum concurrent GPU target compilations per HIP source")
if(NOT LFS_HIP_OFFLOAD_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "LFS_HIP_OFFLOAD_JOBS must be a positive integer")
endif()
function(lfs_hip_native_module target output)
    set(_objects)
    foreach(_unit IN ITEMS runtime pointwise operators)
        set(_object "${CMAKE_CURRENT_BINARY_DIR}/${target}/${_unit}.o")
        set(_defines)
        if(ARGN)
            list(APPEND _defines -include "${ARGN}")
            if(_unit STREQUAL "runtime")
                list(APPEND _defines -DLFS_HIP_TEST_EXPORTS)
            endif()
        endif()
        add_custom_command(OUTPUT "${_object}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${target}"
            COMMAND "${_compiler}" -x hip -std=c++20 -O3 -c
                ${_compile_platform_flags} ${LFS_HIP_COMPRESSION_FLAGS}
                "--offload-jobs=${LFS_HIP_OFFLOAD_JOBS}"
                "-DLFS_HIP_COMPILED_ARCH=\"${LFS_HIP_COMPILED_ARCHITECTURES}\""
                -DLFS_HIP_RUNTIME_EXPORTS "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${_pointwise_generated}" "-I${_pointwise_source}" ${LFS_HIP_OFFLOAD_FLAGS}
                "--hip-path=${LFS_HIP_ROOT}" ${_defines}
                -MMD -MF "${_object}.d"
                "${CMAKE_CURRENT_SOURCE_DIR}/${_unit}.hip" -o "${_object}"
            DEPENDS "${_unit}.hip" ${_pointwise_outputs} ${ARGN}
                "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
            DEPFILE "${_object}.d"
            VERBATIM)
        list(APPEND _objects "${_object}")
    endforeach()
    add_custom_command(OUTPUT "${output}"
        COMMAND "${_compiler}" --hip-link -shared ${_platform_flags}
            ${LFS_HIP_COMPRESSION_FLAGS} ${LFS_HIP_OFFLOAD_FLAGS}
            "--hip-path=${LFS_HIP_ROOT}" ${_objects} -o "${output}"
        DEPENDS ${_objects}
        VERBATIM)
    add_custom_target(${target} ALL DEPENDS "${output}")
endfunction()
lfs_hip_native_module(lfs_hip_runtime_build "${_library}")
set(LFS_HIP_RUNTIME_LIBRARY "${_library}" PARENT_SCOPE)

if(LFS_HIP_NATIVE_GPU_TESTS)
    get_filename_component(_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
    set(_fault_header "${_root}/tests/hip/runtime_faults.hpp")
    get_filename_component(_runtime_directory "${_library}" DIRECTORY)
    if(WIN32)
        set(_fault_library "${_runtime_directory}/lfs_hip_runtime_faults.dll")
    else()
        set(_fault_library "${_runtime_directory}/liblfs_hip_runtime_faults.so")
    endif()
    lfs_hip_native_module(lfs_hip_runtime_faults "${_fault_library}" "${_fault_header}")
    set(LFS_HIP_FAULT_LIBRARY "${_fault_library}" PARENT_SCOPE)
endif()
