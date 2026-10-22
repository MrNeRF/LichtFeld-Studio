# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipRasterKernels.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipArchitectures.cmake")
lfs_hip_architectures()
# A leftover profile from a different build must never be paired with a newer
# generic module just because both happen to target the same architecture.
file(GLOB_RECURSE _module_identity_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hip" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/*.h")
file(GLOB _module_config_inputs "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Hip*.cmake")
list(APPEND _module_identity_inputs ${_module_config_inputs} "${CMAKE_CURRENT_LIST_FILE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../core/tensor/backend/hip/runtime_api.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../core/tensor/backend/hip/external_memory_api.h")
set(_module_paths)
set(_module_configure_inputs)
foreach(_input IN LISTS _module_identity_inputs)
    file(REAL_PATH "${_input}" _absolute)
    list(APPEND _module_paths "${_absolute}")
    # Included CMake files are already tracked. Re-registering equivalent
    # noncanonical spellings can create duplicate Ninja regeneration outputs.
    if(NOT _absolute MATCHES "(\\.cmake|/CMakeLists\\.txt)$")
        list(APPEND _module_configure_inputs "${_absolute}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _module_paths)
list(REMOVE_DUPLICATES _module_configure_inputs)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_module_configure_inputs})
set(_module_identity "${LFS_HIP_COMPILED_ARCHITECTURES}")
if(EXISTS "${LFS_HIP_ROOT}/include/hip/hip_version.h")
    file(SHA256 "${LFS_HIP_ROOT}/include/hip/hip_version.h" _sdk_identity)
    string(APPEND _module_identity "${_sdk_identity}")
endif()
foreach(_input IN LISTS _module_paths)
    file(SHA256 "${_input}" _hash)
    string(APPEND _module_identity "${_hash}")
endforeach()
string(SHA256 _module_set_id "${_module_identity}")
lfs_hip_raster_objects(_raster_include _raster_objects)
option(LFS_HIP_FAST_RUNTIME "Compile the native HIP FastGS path" ON)
if(LFS_HIP_FAST_RUNTIME)
    include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipFastKernels.cmake")
    lfs_hip_fast_objects(_fast_include _fast_objects)
    set(_fast_flags -DLFS_HIP_FAST_RUNTIME "-I${_fast_include}")
    if(LFS_HIP_FAST_PACKED_GRADIENTS)
        list(APPEND _fast_flags -DLFS_HIP_FAST_PACKED_GRADIENTS)
    endif()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipColorKernels.cmake")
lfs_hip_color_objects(_color_include _color_objects)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipMaskKernels.cmake")
lfs_hip_mask_objects(_mask_include _mask_objects)
set(_application_mrnf_directory "${CMAKE_CURRENT_BINARY_DIR}/hip-application-mrnf-generated")
set(_application_mrnf "${_application_mrnf_directory}/application_mrnf_generated.hpp")
add_custom_command(OUTPUT "${_application_mrnf}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_mrnf.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/mrnf_kernels.cu" --output "${_application_mrnf}"
    DEPENDS prepare_application_mrnf.py "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/mrnf_kernels.cu" VERBATIM)
get_filename_component(_gpu_slang_directory "${LFS_GPU_SLANGC}" DIRECTORY)
find_program(LFS_HIP_SLANGC NAMES slangc HINTS "${_gpu_slang_directory}"
    "${LFS_HIP_GLM_INCLUDE_DIR}/../tools/shader-slang"
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang" REQUIRED)
set(_application_lpips_directory "${CMAKE_CURRENT_BINARY_DIR}/hip-application-lpips-generated")
set(_application_blob_directory "${CMAKE_CURRENT_BINARY_DIR}/hip-application-blob-generated")
set(_application_blob "${_application_blob_directory}/application_blob_generated.hpp")
set(_application_blob_shader "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders/blob.slang")
add_custom_command(OUTPUT "${_application_blob}" BYPRODUCTS "${_application_blob_directory}/blob.cu"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_application_blob_directory}"
    COMMAND "${LFS_HIP_SLANGC}" "${_application_blob_shader}"
        -entry main -stage compute -target cuda -fp-mode precise -line-directive-mode none
        -o "${_application_blob_directory}/blob.cu"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_blob.py"
        --source "${_application_blob_directory}/blob.cu" --output "${_application_blob}"
    DEPENDS prepare_application_blob.py "${_application_blob_shader}" "${LFS_HIP_SLANGC}" VERBATIM)
set(_application_structure_directory "${CMAKE_CURRENT_BINARY_DIR}/hip-application-structure-generated")
set(_application_structure "${_application_structure_directory}/application_structure_generated.hpp")
set(_application_structure_shader "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders/structure.slang")
add_custom_command(OUTPUT "${_application_structure}" BYPRODUCTS "${_application_structure_directory}/structure.cu"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_application_structure_directory}"
    COMMAND "${LFS_HIP_SLANGC}" "${_application_structure_shader}"
        -entry main -stage compute -target cuda -fp-mode precise -line-directive-mode none
        -o "${_application_structure_directory}/structure.cu"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_structure.py"
        --source "${_application_structure_directory}/structure.cu" --output "${_application_structure}"
    DEPENDS prepare_application_structure.py "${_application_structure_shader}" "${LFS_HIP_SLANGC}" VERBATIM)
set(_application_lpips "${_application_lpips_directory}/application_lpips_generated.hpp")
set(_application_lpips_shader "${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders/lpips.slang")
set(_application_lpips_core "${CMAKE_CURRENT_SOURCE_DIR}/../../core/tensor/backend/vulkan/shaders")
add_custom_command(OUTPUT "${_application_lpips}" BYPRODUCTS "${_application_lpips_directory}/lpips.cu"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_application_lpips_directory}"
    COMMAND "${LFS_HIP_SLANGC}" "${_application_lpips_shader}" -I "${_application_lpips_core}"
        -entry main -stage compute -target cuda -fp-mode precise -line-directive-mode none
        -o "${_application_lpips_directory}/lpips.cu"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_lpips.py"
        --source "${_application_lpips_directory}/lpips.cu" --output "${_application_lpips}"
    DEPENDS prepare_application_lpips.py "${_application_lpips_shader}" "../vulkan/shaders/lpips_common.slangh"
        "${_application_lpips_core}/half_bits.slangh" "${LFS_HIP_SLANGC}" VERBATIM)
set(_application_morton "${CMAKE_CURRENT_BINARY_DIR}/hip-application-morton-generated/application_morton_generated.hpp")
add_custom_command(OUTPUT "${_application_morton}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_morton.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/morton_reorder_kernels.cu"
        --codec "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/joint_adam_codec.cuh"
        --output "${_application_morton}"
    DEPENDS prepare_application_morton.py prepare_application_mcmc.py "../kernels/morton_reorder_kernels.cu" "../kernels/joint_adam_codec.cuh" VERBATIM)
set(_application_extra "${CMAKE_CURRENT_BINARY_DIR}/hip-application-extra-generated/application_extra_generated.hpp")
set(_application_popspa_directory "${CMAKE_CURRENT_BINARY_DIR}/hip-application-popspa-generated")
set(_application_popspa "${_application_popspa_directory}/application_popspa_generated.hpp")
add_custom_command(OUTPUT "${_application_popspa}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_popspa.py"
        --popspa "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/popspa.cu"
        --effective-rank "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/effective_rank_regularization.cu"
        --output "${_application_popspa}"
    DEPENDS prepare_application_popspa.py prepare_application_mcmc.py
        "../kernels/popspa.cu" "../kernels/effective_rank_regularization.cu" VERBATIM)
add_custom_command(OUTPUT "${_application_extra}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_extra.py"
        --regularization "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/regularization.cu"
        --output "${_application_extra}"
    DEPENDS prepare_application_extra.py prepare_application_mcmc.py "../kernels/regularization.cu" VERBATIM)
set(_application_image "${CMAKE_CURRENT_BINARY_DIR}/hip-application-image-generated/application_image_generated.hpp")
add_custom_command(OUTPUT "${_application_image}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_image.py"
        --image "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/image_kernels.cu"
        --roi "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/roi_weight_map.cu"
        --heatmap "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/camera_loss_heatmap.cu"
        --background "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/grad_alpha.cu"
        --output "${_application_image}"
    DEPENDS prepare_application_image.py prepare_application_mcmc.py "../kernels/image_kernels.cu"
        "../kernels/roi_weight_map.cu" "../kernels/camera_loss_heatmap.cu" "../kernels/grad_alpha.cu" VERBATIM)
set(_application_sh "${CMAKE_CURRENT_BINARY_DIR}/hip-application-sh-generated/application_sh_generated.hpp")
add_custom_command(OUTPUT "${_application_sh}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_sh.py"
        --layout "${CMAKE_CURRENT_SOURCE_DIR}/../../core/cuda/kernels/sh_layout.cu"
        --quant "${CMAKE_CURRENT_SOURCE_DIR}/../../core/cuda/sh_value_quant_kernels.cu"
        --codec "${CMAKE_CURRENT_SOURCE_DIR}/../../core/include/core/sh_value_codec.cuh"
        --output "${_application_sh}"
    DEPENDS prepare_application_sh.py "../../core/cuda/kernels/sh_layout.cu"
        "../../core/cuda/sh_value_quant_kernels.cu" "../../core/include/core/sh_value_codec.cuh"
        "../../core/include/core/sh_layout.hpp" VERBATIM)
set(_application_adam "${CMAKE_CURRENT_BINARY_DIR}/hip-application-adam-generated/application_adam_generated.hpp")
add_custom_command(OUTPUT "${_application_adam}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_adam.py"
        --api "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/optimizer/include/adam_api.h"
        --kernels "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/optimizer/include/adam_kernels.cuh"
        --sh "${CMAKE_CURRENT_SOURCE_DIR}/../rasterization/fastgs/optimizer/src/adam_shN_joint.cu"
        --output "${_application_adam}"
    DEPENDS prepare_application_adam.py "../rasterization/fastgs/optimizer/include/adam_api.h"
        "../rasterization/fastgs/optimizer/include/adam_kernels.cuh" "../rasterization/fastgs/optimizer/src/adam_shN_joint.cu" VERBATIM)
set(_application_photo "${CMAKE_CURRENT_BINARY_DIR}/hip-application-photo-generated/application_photo_generated.hpp")
add_custom_command(OUTPUT "${_application_photo}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_photo.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/ssim.cu"
        --reduction "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/ssim_reduction.cu"
        --l1 "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/l1_loss.cu"
        --warp "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/warp_reduce.cuh"
        --output "${_application_photo}"
    DEPENDS prepare_application_photo.py prepare_application_mcmc.py application_photo_plane_kernel.hpp "../kernels/ssim.cu"
        "../kernels/ssim_reduction.cu" "../kernels/l1_loss.cu" "../kernels/warp_reduce.cuh" VERBATIM)
set(_refine "${CMAKE_CURRENT_BINARY_DIR}/hip-refine-generated/application_refine_generated.hpp")
add_custom_command(OUTPUT "${_refine}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_refine.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/densification_kernels.cu"
        --pruning "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/pruning_kernels.cu"
        --screen "${CMAKE_CURRENT_SOURCE_DIR}/../include/lfs/training/screen_share.cuh"
        --output "${_refine}"
    DEPENDS prepare_application_refine.py prepare_application_mcmc.py "../kernels/densification_kernels.cu"
        "../kernels/pruning_kernels.cu" "../include/lfs/training/screen_share.cuh" VERBATIM)
set(_mcmc "${CMAKE_CURRENT_BINARY_DIR}/hip-mcmc-generated/application_mcmc_generated.hpp")
add_custom_command(OUTPUT "${_mcmc}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_application_mcmc.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/mcmc_kernels.cu" --output "${_mcmc}"
    DEPENDS prepare_application_mcmc.py "../kernels/mcmc_kernels.cu" VERBATIM)
set(_photometric "${CMAKE_CURRENT_BINARY_DIR}/hip-photometric-generated/photometric.hpp")
add_custom_command(OUTPUT "${_photometric}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_photometric.py"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/ssim.cu" --output "${_photometric}"
    DEPENDS prepare_photometric.py "../kernels/ssim.cu" VERBATIM)
if(CMAKE_HOST_WIN32)
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_module "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_training.dll")
    set(_platform_flags -lgdi32)
    set(_identity_object "${CMAKE_CURRENT_BINARY_DIR}/adapter_identity.obj")
    add_custom_command(OUTPUT "${_identity_object}"
        COMMAND "${_compiler}" -x c++ -std=c++20 -O2 -c
            "${CMAKE_CURRENT_SOURCE_DIR}/adapter_identity_win32.cpp" -o "${_identity_object}"
        DEPENDS adapter_identity_win32.cpp "${_compiler}" VERBATIM)
else()
    set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_module "${CMAKE_CURRENT_BINARY_DIR}/liblfs_hip_training.so")
    set(_platform_flags -fPIC "-Wl,-rpath,$ORIGIN/.." "-Wl,-rpath,$ORIGIN/../../..")
endif()
function(lfs_add_hip_training_module module fast_include fast_objects packed offload_flags architectures)
set(_fast_flags)
if(LFS_HIP_FAST_RUNTIME)
    set(_fast_flags -DLFS_HIP_FAST_RUNTIME "-I${fast_include}")
    if(packed)
        list(APPEND _fast_flags -DLFS_HIP_FAST_PACKED_GRADIENTS)
    endif()
endif()
add_custom_command(OUTPUT "${module}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<PATH:GET_PARENT_PATH,${module}>"
    COMMAND "${_compiler}" -x hip -std=c++20 -O3 -shared ${_platform_flags} ${_fast_flags}
        -DLFS_HIP_TRAIN_EXPORTS ${offload_flags} "--hip-path=${LFS_HIP_ROOT}"
        "-DLFS_HIP_COMPILED_ARCH=\"${architectures}\""
        "-DLFS_HIP_MODULE_SET_ID=\"${_module_set_id}\""
        "-I${_raster_include}" "-I${LFS_HIP_GLM_INCLUDE_DIR}"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-photometric-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-mcmc-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-refine-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-photo-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-adam-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-sh-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-image-generated"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-extra-generated"
        "-I${_application_popspa_directory}"
        "-I${CMAKE_CURRENT_BINARY_DIR}/hip-application-morton-generated"
        "-I${_application_lpips_directory}"
        "-I${_application_structure_directory}"
        "-I${_application_blob_directory}"
        "-I${_application_mrnf_directory}"
        "-I${_color_include}"
        "-I${_mask_include}"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/../../core/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/training.hip" -x none ${_raster_objects} ${fast_objects} ${_color_objects} ${_mask_objects} ${_identity_object} -o "${module}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../core/tensor/backend/hip/architecture_runtime.hpp" training.hip training_api.h application_dispatch_api.h application_geometry.hpp application_color_api.h application_bilateral.hpp application_ppisp.hpp ${_color_objects} application_mrnf.hpp application_mrnf_api.h "${_application_mrnf}" application_controller.hpp application_controller_api.h application_blob.hpp application_blob_api.h "${_application_blob}" application_structure.hpp application_structure_api.h "${_application_structure}" application_lpips.hpp application_lpips_api.h "${_application_lpips}" application_fast.hpp application_fast_api.h application_mcmc.hpp application_mcmc_api.h application_refine.hpp application_refine_api.h application_photo.hpp photo_plane_policy.hpp application_photo_api.h application_adam.hpp application_adam_api.h application_sh.hpp application_sh_api.h application_image.hpp application_image_api.h application_extra.hpp application_extra_api.h application_morton.hpp application_morton_api.h ${_raster_objects} ${fast_objects} ${_identity_object} "${_photometric}" "${_mcmc}" "${_refine}" "${_application_photo}" "${_application_adam}" "${_application_sh}" "${_application_image}" "${_application_extra}" "${_application_morton}"
        application_masks.hpp application_masks_api.h ${_mask_objects}
        application_gsplat.hpp application_gsplat_api.h
        application_popspa.hpp application_popspa_api.h "${_application_popspa}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../core/tensor/backend/hip/runtime_api.h"
        "${_compiler}" "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
        "${LFS_HIP_ROOT}/include/hipcub/hipcub.hpp"
    VERBATIM)
endfunction()
lfs_add_hip_training_module("${_module}" "${_fast_include}" "${_fast_objects}"
    "${LFS_HIP_FAST_PACKED_GRADIENTS}" "${LFS_HIP_OFFLOAD_FLAGS}" "${LFS_HIP_COMPILED_ARCHITECTURES}")
add_custom_target(lfs_hip_training_build ALL DEPENDS "${_module}")
add_dependencies(lfs_hip_training_build lfs_hip_raster_kernels)
add_dependencies(lfs_hip_training_build lfs_hip_color_kernels)
add_dependencies(lfs_hip_training_build lfs_hip_mask_kernels)
if(LFS_HIP_FAST_RUNTIME)
    add_dependencies(lfs_hip_training_build lfs_hip_fast_kernels)
endif()
set(LFS_HIP_TRAINING_LIBRARY "${_module}" PARENT_SCOPE)
set(LFS_HIP_PHOTOMETRIC_HEADER "${_photometric}" PARENT_SCOPE)

option(LFS_HIP_ARCH_PROFILES "Include validated training profiles in multiarchitecture builds" ON)
list(LENGTH LFS_HIP_EFFECTIVE_ARCHITECTURES _architecture_count)
if(LFS_HIP_FAST_RUNTIME AND LFS_HIP_ARCH_PROFILES AND _architecture_count GREATER 1
        AND "gfx1150" IN_LIST LFS_HIP_EFFECTIVE_ARCHITECTURES)
    function(lfs_add_gfx1150_profile)
        set(LFS_HIP_ARCHITECTURE gfx1150)
        set(LFS_HIP_ARCHITECTURES gfx1150)
        set(LFS_HIP_VALIDATED_TUNING ON)
        get_cmake_property(_variables VARIABLES)
        foreach(_option IN LISTS _variables)
            if(_option MATCHES "^LFS_HIP_FAST_")
                set(${_option} ON)
            endif()
        endforeach()
        lfs_hip_fast_objects(_profile_include _profile_objects lfs_hip_fast_gfx1150)
        get_filename_component(_module_name "${_module}" NAME)
        set(_profile_module "${CMAKE_CURRENT_BINARY_DIR}/profiles/gfx1150/${_module_name}")
        lfs_add_hip_training_module("${_profile_module}" "${_profile_include}" "${_profile_objects}"
            ON "--offload-arch=gfx1150" gfx1150)
        add_custom_target(lfs_hip_gfx1150_profile ALL DEPENDS "${_profile_module}")
        add_dependencies(lfs_hip_gfx1150_profile lfs_hip_fast_gfx1150)
        set(LFS_HIP_PROFILE_LIBRARY "${_profile_module}" PARENT_SCOPE)
    endfunction()
    lfs_add_gfx1150_profile()
    set(LFS_HIP_PROFILE_LIBRARY "${LFS_HIP_PROFILE_LIBRARY}" PARENT_SCOPE)
endif()

# The app's host compiler owns this adapter; only the isolated module uses HIP.
if(CMAKE_CXX_COMPILER_LOADED)
    add_library(lfs_hip_training_host STATIC session.cpp)
    target_compile_features(lfs_hip_training_host PUBLIC cxx_std_20)
    set_target_properties(lfs_hip_training_host PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    target_include_directories(lfs_hip_training_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(lfs_hip_training_host PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(lfs_hip_training_host lfs_hip_training_build)
endif()
