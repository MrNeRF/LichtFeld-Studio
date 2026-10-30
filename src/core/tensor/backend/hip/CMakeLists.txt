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
add_custom_command(OUTPUT "${_tensor_generated}/hip_point_filter.hpp" "${_tensor_generated}/hip_spatial_components.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_point_filter.py"
        --source "${_point_source}" --output "${_tensor_generated}/hip_point_filter.hpp"
    DEPENDS prepare_point_filter.py "${_point_source}/tensor/backend/cuda/kernels/simplify_merge.cu" "${_point_source}/include/core/morton.hpp" "${_point_source}/cuda/kernels/splat_transform.cu" "${_point_source}/tensor/internal/ppisp.hpp" "${_point_source}/include/core/tensor_ppisp.hpp" "${_point_source}/tensor/internal/point_filter.hpp"
        "${_point_source}/tensor/internal/point_math.hpp" "${_point_source}/tensor/internal/point_spatial.hpp" "${_point_source}/tensor/internal/nearest_point.hpp" "${_point_source}/tensor/internal/point_tree.hpp" "${_point_source}/tensor/internal/triangle_tree.hpp" "${_point_source}/include/core/detail/descriptors.hpp" "${_point_source}/tensor/backend/cuda/kernels/tensor_spatial.cu" "${_point_source}/include/core/tensor_spatial.hpp"
        "${_point_source}/tensor/internal/point_labels.hpp"
        "${_point_source}/tensor/internal/point_projection.hpp" "${_point_source}/tensor/internal/point_region.hpp"
    VERBATIM)
find_program(LFS_HIP_CORE_SLANGC NAMES slangc
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang"
          "${LFS_HIP_GLM_INCLUDE_DIR}/../tools/shader-slang" REQUIRED)
set(_inference_shader "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders/inference.slang")
add_custom_command(OUTPUT "${_tensor_generated}/hip_inference.hpp" BYPRODUCTS "${_tensor_generated}/inference.cu"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_tensor_generated}"
    COMMAND "${LFS_HIP_CORE_SLANGC}" "${_inference_shader}"
        -entry main -stage compute -target cuda -fp-mode precise -line-directive-mode none
        -o "${_tensor_generated}/inference.cu"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_inference.py"
        --source "${_tensor_generated}/inference.cu" --output "${_tensor_generated}/hip_inference.hpp"
    DEPENDS prepare_inference.py "${_inference_shader}" "${LFS_HIP_CORE_SLANGC}" VERBATIM)
add_custom_command(OUTPUT "${_tensor_generated}/hip_pointwise_functors.hpp" "${_tensor_generated}/hip_pointwise_f32.inc" "${_tensor_generated}/hip_pointwise_f16.inc" "${_tensor_generated}/hip_pointwise_eval.inc" "${_tensor_generated}/hip_pointwise_flags.inc" "${_tensor_generated}/hip_pointwise_broadcast.inc" "${_tensor_generated}/hip_pointwise_half_broadcast.inc" "${_tensor_generated}/hip_pointwise_scalar_left.inc"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_pointwise.py"
        --source "${_tensor_source}" --output "${_tensor_generated}"
    DEPENDS prepare_pointwise.py "${_tensor_source}/tensor_functors.hpp" "${_tensor_source}/pointwise_ops.def" "${_tensor_source}/pointwise_lowering.hpp" VERBATIM)
set(_lens_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/undistort/undistort.cu")
set(_lens_model "${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/undistort/distortion_model.cuh")
add_custom_command(OUTPUT "${_tensor_generated}/hip_lens.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_lens.py"
        --source "${_lens_source}" --output "${_tensor_generated}/hip_lens.hpp"
    DEPENDS prepare_lens.py "${_lens_source}" "${_lens_model}" VERBATIM)
set(_shared_mask_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/kernels/mask_normalization.cu")
set(_shared_image_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../../io/cuda/image_format_kernels.cu")
add_custom_command(OUTPUT "${_tensor_generated}/hip_shared_image.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_shared_image.py"
        --source "${_shared_image_source}" --mask-source "${_shared_mask_source}" --output "${_tensor_generated}/hip_shared_image.hpp"
    DEPENDS prepare_shared_image.py "${_shared_image_source}" "${_shared_mask_source}" VERBATIM)
set(_shared_resize_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../cuda/lanczos_resize/lanczos_resize.cu")
add_custom_command(OUTPUT "${_tensor_generated}/hip_shared_resize.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_shared_resize.py"
        --source "${_shared_resize_source}" --output "${_tensor_generated}/hip_shared_resize.hpp"
    DEPENDS prepare_shared_resize.py "${_shared_resize_source}" VERBATIM)
set(_export_shaders "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders")
add_custom_command(OUTPUT "${_tensor_generated}/hip_export.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_export.py"
        --source "${_export_shaders}" --slang "${LFS_HIP_CORE_SLANGC}" --output "${_tensor_generated}/hip_export.hpp"
    DEPENDS prepare_export.py "${_export_shaders}/export_kmeans.slang" "${_export_shaders}/export_kmeans_common.slangh"
        "${_export_shaders}/export_kmeans_accumulate.slang" "${_export_shaders}/export_kmeans_group.slang"
        "${_export_shaders}/export_radix.slang" "${_export_shaders}/export_morton.slang"
        "${_export_shaders}/export_decimate.slang" "${_export_shaders}/export_decimate_cost.slang"
        "${_export_shaders}/export_decimate_merge.slang" "${_export_shaders}/df64.slangh" "${_export_shaders}/point_raster.slang" "${_export_shaders}/byte_read.slangh"
        "${_point_source}/include/core/rounded_math.slangh" "${LFS_HIP_CORE_SLANGC}" VERBATIM)
set(_pointwise_source "${_tensor_source}")
set(_pointwise_generated "${_tensor_generated}")
set(_pointwise_outputs
    "${_tensor_generated}/hip_pointwise_functors.hpp"
    "${_tensor_generated}/hip_pointwise_f32.inc"
    "${_tensor_generated}/hip_pointwise_f16.inc"
    "${_tensor_generated}/hip_pointwise_eval.inc"
    "${_tensor_generated}/hip_pointwise_flags.inc"
    "${_tensor_generated}/hip_pointwise_broadcast.inc"
    "${_tensor_generated}/hip_pointwise_half_broadcast.inc"
    "${_tensor_generated}/hip_pointwise_scalar_left.inc"
)
set(_extended_outputs
    "${_tensor_generated}/hip_point_filter.hpp"
    "${_tensor_generated}/hip_spatial_components.hpp"
    "${_tensor_generated}/hip_inference.hpp"
    "${_tensor_generated}/hip_lens.hpp"
    "${_tensor_generated}/hip_shared_image.hpp"
    "${_tensor_generated}/hip_shared_resize.hpp"
    "${_tensor_generated}/hip_export.hpp"
)
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
            DEPENDS "${_unit}.hip" ${_pointwise_outputs} ${_extended_outputs} ${ARGN}
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
    set(_expression_emitter "${_tensor_generated}/hip_expression_emitter.hpp")
    set(_expression_source "${CMAKE_CURRENT_SOURCE_DIR}/../metal/expression_msl.cpp")
    add_custom_command(OUTPUT "${_expression_emitter}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_expression.py"
            --source "${_expression_source}" --output "${_expression_emitter}"
        DEPENDS prepare_expression.py "${_expression_source}" VERBATIM)
    add_custom_target(lfs_hip_expression_emitter_build DEPENDS "${_expression_emitter}")
    add_dependencies(lfs_core lfs_hip_expression_emitter_build)
    target_sources(lfs_core PRIVATE "${_expression_emitter}")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/hip_backend_ops.cpp"
        TARGET_DIRECTORY lfs_core PROPERTIES
        COMPILE_DEFINITIONS "LFS_HIP_EXPRESSION_JIT=1;$<$<PLATFORM_ID:Windows>:LFS_HIP_RTC_FILENAME=\"${_rtc_filename}\">"
        INCLUDE_DIRECTORIES "${_tensor_generated}")
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
