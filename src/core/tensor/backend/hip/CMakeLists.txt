# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Independent of the application and its CXX/CUDA compiler configuration.
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The native HIP runtime does not support cross-compilation")
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
    message(FATAL_ERROR "The native HIP runtime supports Windows and Linux")
endif()
set(LFS_HIP_ROOT "" CACHE PATH "Explicit ROCm SDK root")
set(LFS_HIP_ARCHITECTURE "" CACHE STRING "Explicit HIP GPU target, e.g. gfx1150")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/HipArchitectures.cmake")
lfs_hip_architectures()
set(LFS_HIP_EFFECTIVE_ARCHITECTURES "${LFS_HIP_EFFECTIVE_ARCHITECTURES}" PARENT_SCOPE)
set(LFS_HIP_COMPILED_ARCHITECTURES "${LFS_HIP_COMPILED_ARCHITECTURES}" PARENT_SCOPE)
set(LFS_HIP_OFFLOAD_FLAGS "${LFS_HIP_OFFLOAD_FLAGS}" PARENT_SCOPE)
if(WIN32 OR CMAKE_HOST_WIN32)
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_library "${CMAKE_BINARY_DIR}/lfs_hip_runtime.dll")
else()
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_library "${CMAKE_CURRENT_BINARY_DIR}/liblfs_hip_runtime.so")
    set(_compile_platform_flags -fPIC)
    set(_platform_flags "-Wl,-rpath,$ORIGIN/.."
        "-Wl,-rpath,${LFS_HIP_ROOT}/lib" "-Wl,-rpath,${LFS_HIP_ROOT}/lib64")
endif()
if(NOT EXISTS "${_compiler}" OR NOT EXISTS "${LFS_HIP_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "LFS_HIP_ROOT must identify a complete ROCm development SDK")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(WIN32)
    set(_rtc_library "${LFS_HIP_ROOT}/lib/hiprtc.lib")
    file(GLOB _rtc_dlls "${LFS_HIP_ROOT}/bin/hiprtc*.dll")
    if(NOT EXISTS "${_rtc_library}" OR NOT _rtc_dlls)
        message(FATAL_ERROR "HIP fused expressions require HIPRTC from the selected ROCm SDK")
    endif()
    set(_rtc_modules ${_rtc_dlls})
    list(FILTER _rtc_modules EXCLUDE REGEX "hiprtc-builtins")
    list(LENGTH _rtc_modules _rtc_count)
    if(NOT _rtc_count EQUAL 1)
        message(FATAL_ERROR "The selected HIP SDK must provide one unambiguous HIPRTC runtime")
    endif()
    list(GET _rtc_modules 0 _rtc_module)
    get_filename_component(_rtc_filename "${_rtc_module}" NAME)
    # Keep the selected SDK runtime beside the module. The secure Windows
    # loader deliberately does not search PATH or change process DLL policy.
    file(GLOB _runtime_files "${LFS_HIP_ROOT}/bin/amdhip64*.dll" "${LFS_HIP_ROOT}/bin/amd_comgr*.dll" "${LFS_HIP_ROOT}/bin/hiprtc*.dll" "${LFS_HIP_ROOT}/bin/rocm_kpack*.dll")
    foreach(_runtime_file IN LISTS _runtime_files)
        get_filename_component(_runtime_name "${_runtime_file}" NAME)
        configure_file("${_runtime_file}" "${CMAKE_BINARY_DIR}/${_runtime_name}" COPYONLY)
    endforeach()
else()
    find_library(_rtc_library NAMES hiprtc PATHS "${LFS_HIP_ROOT}/lib" "${LFS_HIP_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
endif()
set(_tensor_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/core/detail")
set(_tensor_generated "${CMAKE_CURRENT_BINARY_DIR}/hip-tensor-generated")
set(_point_source "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
find_program(LFS_HIP_CORE_SLANGC NAMES slangc
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang"
          "${LFS_HIP_GLM_INCLUDE_DIR}/../tools/shader-slang" REQUIRED)
set(_shader_source "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders")
file(GLOB _shader_headers CONFIGURE_DEPENDS "${_shader_source}/*.slangh")
set(_extended_outputs)
function(lfs_hip_shader name source)
    set(_output "${_tensor_generated}/${name}.cu")
    add_custom_command(OUTPUT "${_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_tensor_generated}"
        COMMAND "${LFS_HIP_CORE_SLANGC}" "${_shader_source}/${source}.slang"
            -entry "${name}" -stage compute -target cuda -fp-mode precise -line-directive-mode none
            "-DLFS_KERNEL_ENTRY=${name}" ${ARGN} -o "${_output}"
        DEPENDS "${_shader_source}/${source}.slang" ${_shader_headers}
            "${_point_source}/include/core/rounded_math.slangh" "${LFS_HIP_CORE_SLANGC}"
        VERBATIM)
    set(_extended_outputs ${_extended_outputs} "${_output}" PARENT_SCOPE)
endfunction()
foreach(_storage IN ITEMS f32 f16)
    if(_storage STREQUAL "f32")
        set(_type float)
    else()
        set(_type half)
    endif()
    foreach(_operation IN ITEMS 0 1 2 4 5)
        lfs_hip_shader("inference_${_operation}_${_storage}" inference
            "-DLFS_INFERENCE_OPERATION=${_operation}" "-DLFS_INFERENCE_STORAGE=${_type}")
    endforeach()
endforeach()
foreach(_shader IN ITEMS export_kmeans export_kmeans_accumulate export_kmeans_group export_radix
        export_morton export_decimate export_decimate_cost export_decimate_merge point_raster)
    lfs_hip_shader("${_shader}_kernel" "${_shader}" -DLFS_DYNAMIC_PHASE=1)
endforeach()
set(_pointwise_source "${_tensor_source}")
set(_pointwise_generated "${_tensor_generated}")
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
                -DLFS_HIP_RUNTIME_EXPORTS "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${_pointwise_generated}" "-I${_pointwise_source}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../include" ${LFS_HIP_OFFLOAD_FLAGS}
                "--hip-path=${LFS_HIP_ROOT}" ${_defines}
                -MMD -MF "${_object}.d"
                "${CMAKE_CURRENT_SOURCE_DIR}/${_unit}.hip" -o "${_object}"
            DEPENDS "${_unit}.hip" ${_extended_outputs} ${ARGN}
                "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
            DEPFILE "${_object}.d"
            VERBATIM)
        list(APPEND _objects "${_object}")
    endforeach()
    add_custom_command(OUTPUT "${output}"
        COMMAND "${_compiler}" --hip-link -shared ${_platform_flags}
            ${LFS_HIP_COMPRESSION_FLAGS} ${LFS_HIP_OFFLOAD_FLAGS}
            "--hip-path=${LFS_HIP_ROOT}" ${_objects} "${_rtc_library}" -o "${output}"
        DEPENDS ${_objects}
        VERBATIM)
    add_custom_target(${target} ALL DEPENDS "${output}")
endfunction()
lfs_hip_native_module(lfs_hip_runtime_build "${_library}")
set(LFS_HIP_RUNTIME_LIBRARY "${_library}" PARENT_SCOPE)

if(LFS_TENSOR_HIP)
    set(_expression_sources "${CMAKE_CURRENT_SOURCE_DIR}/expression_hip.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../internal/expression_c_like.hpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_expression_sources})
    set(_expression_hashes "")
    foreach(_source IN LISTS _expression_sources)
        file(SHA256 "${_source}" _hash)
        string(APPEND _expression_hashes "${_hash}")
    endforeach()
    string(SHA256 _expression_id "${_expression_hashes}")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/hip_backend_ops.cpp"
        TARGET_DIRECTORY lfs_core PROPERTIES
        COMPILE_DEFINITIONS "LFS_HIP_EXPRESSION_EMITTER_ID=\"${_expression_id}\";$<$<PLATFORM_ID:Windows>:LFS_HIP_RTC_FILENAME=\"${_rtc_filename}\">")
endif()

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
