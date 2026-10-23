# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
function(lfs_hip_color_objects include_output objects_output)
    set(_target lfs_hip_color_kernels)
    set(_suffix "")
    if(ARGC GREATER 2)
        set(_target "${ARGV2}")
        set(_suffix "-${_target}")
    endif()
    set(_dependency lfs_hip_raster_kernels)
    if(ARGC GREATER 3)
        set(_dependency "${ARGV3}")
    endif()
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HipArchitectures.cmake")
    lfs_hip_architectures()
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/hip-color-generated${_suffix}")
    set(_names bilateral_grid.cuh bilateral_grid_forward.cu bilateral_grid_backward.cu bilateral_grid_tv.cu
        memory_ops.cuh warp_reduce.cuh ppisp_math.cuh ppisp_math_bwd.cuh ppisp.cuh ppisp.cu
        depth_loss.hpp depth_loss.cu normal_loss.hpp normal_loss.cu normal_consistency_loss.hpp normal_consistency_loss.cu)
    set(_inputs "${_root}/src/training/hip/logical_subgroup32.hpp")
    set(_outputs "${_generated}/logical_subgroup32.hpp")
    foreach(_name IN LISTS _names)
        list(APPEND _inputs "${_root}/src/training/kernels/${_name}")
        list(APPEND _outputs "${_generated}/${_name}")
    endforeach()
    list(APPEND _inputs "${_root}/src/training/include/lfs/training/ops/geometry_types.hpp")
    list(APPEND _outputs "${_generated}/hip_application_diagnostics.hpp" "${_generated}/hip_application_stream.hpp"
        "${_generated}/hip_geometry_scratch.hpp" "${_generated}/lfs/training/ops/geometry_types.hpp")
    add_custom_command(OUTPUT "${_generated}/prepared.stamp" BYPRODUCTS ${_outputs}
        COMMAND "${Python3_EXECUTABLE}" "${_root}/src/training/hip/prepare_color_kernels.py"
            --source "${_root}/src/training/kernels" --output "${_generated}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_generated}/prepared.stamp"
        DEPENDS "${_root}/src/training/hip/prepare_color_kernels.py" ${_inputs} VERBATIM)
    if(CMAKE_HOST_WIN32)
        set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
        set(_extension obj)
    else()
        set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
        set(_extension o)
        set(_platform_flags -fPIC)
    endif()
    set(_objects)
    foreach(_unit bilateral_grid_forward bilateral_grid_backward bilateral_grid_tv ppisp depth_loss normal_loss normal_consistency_loss)
        set(_object "${_generated}/${_unit}.${_extension}")
        add_custom_command(OUTPUT "${_object}"
            COMMAND "${_compiler}" -x hip -std=c++20 -O3 -DNDEBUG ${_platform_flags}
                ${LFS_HIP_OFFLOAD_FLAGS} ${LFS_HIP_COMPRESSION_FLAGS} ${LFS_HIP_TEST_WAVE_FLAGS} "--hip-path=${LFS_HIP_ROOT}"
                "-I${_generated}" "-I${_raster_include}" "-I${LFS_HIP_GLM_INCLUDE_DIR}"
                -c "${_generated}/${_unit}.cu" -o "${_object}"
            DEPENDS ${LFS_HIP_TEST_RUNTIME_DEPENDS} ${_outputs} ${_dependency} "${_compiler}" VERBATIM)
        list(APPEND _objects "${_object}")
    endforeach()
    add_custom_target(${_target} DEPENDS "${_generated}/prepared.stamp" ${_objects})
    set(${include_output} "${_generated}" PARENT_SCOPE)
    set(${objects_output} "${_objects}" PARENT_SCOPE)
endfunction()
