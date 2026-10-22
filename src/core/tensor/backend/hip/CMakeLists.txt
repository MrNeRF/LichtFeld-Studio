# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Deliberately independent of the application's CXX/CUDA compilers.
set(LFS_HIP_ROOT "" CACHE PATH "Explicit ROCm SDK root")
set(LFS_HIP_ARCHITECTURE "" CACHE STRING "Explicit HIP GPU target, e.g. gfx1150")
if(NOT LFS_HIP_ARCHITECTURE MATCHES "^gfx[0-9a-f]+$")
    message(FATAL_ERROR "The experimental HIP runtime needs one explicit LFS_HIP_ARCHITECTURE")
endif()
if(WIN32 OR CMAKE_HOST_WIN32)
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_library "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_runtime.dll")
else()
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_library "${CMAKE_CURRENT_BINARY_DIR}/liblfs_hip_runtime.so")
    set(_platform_flags -fPIC)
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
    foreach(_rtc_dll IN LISTS _rtc_dlls)
        get_filename_component(_rtc_name "${_rtc_dll}" NAME)
        configure_file("${_rtc_dll}" "${CMAKE_CURRENT_BINARY_DIR}/${_rtc_name}" COPYONLY)
        configure_file("${_rtc_dll}" "${CMAKE_BINARY_DIR}/${_rtc_name}" COPYONLY)
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
    DEPENDS prepare_point_filter.py "${_point_source}/include/core/morton.hpp" "${_point_source}/cuda/kernels/splat_transform.cu" "${_point_source}/tensor/internal/ppisp.hpp" "${_point_source}/include/core/tensor_ppisp.hpp" "${_point_source}/tensor/internal/point_filter.hpp"
        "${_point_source}/tensor/internal/point_math.hpp" "${_point_source}/tensor/internal/point_spatial.hpp" "${_point_source}/tensor/internal/nearest_point.hpp" "${_point_source}/tensor/internal/point_tree.hpp" "${_point_source}/include/core/detail/descriptors.hpp" "${_point_source}/tensor/backend/cuda/kernels/tensor_spatial.cu" "${_point_source}/include/core/tensor_spatial.hpp"
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
add_custom_command(OUTPUT "${_tensor_generated}/hip_tensor_functors.hpp" "${_tensor_generated}/hip_tensor_dispatch.inc" "${_tensor_generated}/hip_tensor_half_dispatch.inc" "${_tensor_generated}/hip_tensor_eval.inc" "${_tensor_generated}/hip_tensor_flags.inc" "${_tensor_generated}/hip_tensor_broadcast.inc"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_tensor_ops.py"
        --source "${_tensor_source}" --output "${_tensor_generated}"
    DEPENDS prepare_tensor_ops.py "${_tensor_source}/tensor_functors.hpp" "${_tensor_source}/pointwise_ops.def" VERBATIM)
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
add_custom_command(OUTPUT "${_library}"
    COMMAND "${_compiler}" -x hip -std=c++20 -O3 -shared
        ${_platform_flags}
        "-DLFS_HIP_COMPILED_ARCH=\"${LFS_HIP_ARCHITECTURE}\""
        "-I${_tensor_generated}" "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../include" -DLFS_HIP_RUNTIME_EXPORTS "--offload-arch=${LFS_HIP_ARCHITECTURE}"
        "--hip-path=${LFS_HIP_ROOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime.hip" -x none "${_rtc_library}" -o "${_library}"
    DEPENDS runtime.hip runtime_api.h expression_api.h expression_runtime.hpp "${_rtc_library}" bias_api.h bias_runtime.hpp integer_reduce_api.h integer_reduce_runtime.hpp inference_api.h inference_runtime.hpp gemm_api.h gemm_runtime.hpp "${_tensor_generated}/hip_inference.hpp" inverse_warp_api.h inverse_warp_runtime.hpp pool_runtime.hpp where_runtime.hpp where_api.h external_memory_api.h external_memory_runtime.hpp device_identity_api.h device_identity_runtime.hpp chain_reduce_runtime.hpp sh_codec_api.h sh_codec_runtime.hpp shared_warp_runtime.hpp shared_image_runtime.hpp shared_resize_runtime.hpp "${_tensor_generated}/hip_shared_resize.hpp" "${_tensor_generated}/hip_shared_image.hpp" "${_tensor_generated}/hip_lens.hpp" "${_lens_model}" "${_tensor_generated}/hip_tensor_functors.hpp" "${_tensor_generated}/hip_tensor_dispatch.inc" "${_tensor_generated}/hip_tensor_half_dispatch.inc" "${_tensor_generated}/hip_tensor_eval.inc" "${_tensor_generated}/hip_tensor_flags.inc" "${_tensor_generated}/hip_tensor_broadcast.inc" "${_compiler}"
        point_filter_api.h point_filter_runtime.hpp "${_tensor_generated}/hip_point_filter.hpp" "${_tensor_generated}/hip_spatial_components.hpp"
        point_tree_api.h point_tree_runtime.hpp ppisp_runtime.hpp affine_runtime.hpp morton_runtime.hpp multinomial_runtime.hpp export_runtime.hpp "${_tensor_generated}/hip_export.hpp" predicate_api.h predicate_runtime.hpp integer_pointwise_runtime.hpp arg_extreme_runtime.hpp cdist_runtime.hpp spatial_runtime.hpp
        timing_api.h timing_runtime.hpp
        "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
    VERBATIM)
add_custom_target(lfs_hip_runtime_build ALL DEPENDS "${_library}")
# Consumers load this module explicitly; it cannot replace a CUDA runtime or
# register an incomplete tensor/training backend merely by being built.
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
        COMPILE_DEFINITIONS LFS_HIP_EXPRESSION_JIT=1
        INCLUDE_DIRECTORIES "${_tensor_generated}")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/hip_unsupported_ops.cpp"
        TARGET_DIRECTORY lfs_core PROPERTIES COMPILE_DEFINITIONS LFS_HIP_EXPRESSION_JIT=1)
endif()
