# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
function(lfs_hip_mask_objects include_output objects_output)
    set(_target lfs_hip_mask_kernels)
    set(_suffix "")
    if(ARGC GREATER 2)
        set(_target "${ARGV2}")
        set(_suffix "-${_target}")
    endif()
    set(_dependency lfs_hip_color_kernels)
    if(ARGC GREATER 3)
        set(_dependency "${ARGV3}")
    endif()
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HipArchitectures.cmake")
    lfs_hip_architectures()
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/hip-mask-generated${_suffix}")
    set(_inputs "${_root}/src/training/hip/prepare_mask_kernels.py"
        "${_root}/src/training/kernels/mask_preprocess.cu" "${_root}/src/training/kernels/mask_preprocess.hpp"
        "${_root}/src/training/kernels/mesh_mask.cu" "${_root}/src/core/include/core/tensor_image.hpp"
        "${_root}/src/training/include/lfs/training/ops/domain_types.hpp"
        "${_root}/src/training/include/lfs/training/ops/masks.hpp"
        "${_root}/src/core/cuda/undistort/distortion_model.cuh")
    set(_names mask_preprocess.cu mask_preprocess.hpp mesh_coverage.cu hip_mask_types.hpp hip_mask_distortion.hpp hip_mask_distortion_model.hpp)
    set(_outputs)
    foreach(_name IN LISTS _names)
        list(APPEND _outputs "${_generated}/${_name}")
    endforeach()
    add_custom_command(OUTPUT "${_generated}/prepared.stamp" BYPRODUCTS ${_outputs}
        COMMAND "${Python3_EXECUTABLE}" "${_root}/src/training/hip/prepare_mask_kernels.py"
            --source "${_root}/src/training/kernels" --output "${_generated}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_generated}/prepared.stamp"
        DEPENDS ${_inputs} VERBATIM)
    if(CMAKE_HOST_WIN32)
        set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
        set(_extension obj)
    else()
        set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
        set(_extension o)
        set(_platform_flags -fPIC)
    endif()
    set(_objects)
    foreach(_unit mask_preprocess mesh_coverage)
        set(_object "${_generated}/${_unit}.${_extension}")
        add_custom_command(OUTPUT "${_object}"
            COMMAND "${_compiler}" -x hip -std=c++20 -O3 -DNDEBUG ${_platform_flags}
                ${LFS_HIP_OFFLOAD_FLAGS} ${LFS_HIP_COMPRESSION_FLAGS} ${LFS_HIP_TEST_WAVE_FLAGS} "--hip-path=${LFS_HIP_ROOT}"
                "-I${_generated}" "-I${_color_include}" "-I${_root}/src/training/hip" "-I${_root}/src/core/include"
                -c "${_generated}/${_unit}.cu" -o "${_object}"
            DEPENDS ${LFS_HIP_TEST_RUNTIME_DEPENDS} ${_outputs} ${_dependency} "${_root}/src/training/hip/application_masks_api.h"
                "${_root}/src/training/hip/application_color_api.h" "${_root}/src/training/hip/training_api.h"
                "${_root}/src/core/include/core/camera_types.h" "${_compiler}" VERBATIM)
        list(APPEND _objects "${_object}")
    endforeach()
    add_custom_target(${_target} DEPENDS "${_generated}/prepared.stamp" ${_objects})
    set(${include_output} "${_generated}" PARENT_SCOPE)
    set(${objects_output} "${_objects}" PARENT_SCOPE)
endfunction()
