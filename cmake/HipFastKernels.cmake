# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

function(lfs_hip_fast_objects include_output objects_output)
    set(_validated_default OFF)
    if(LFS_HIP_ARCHITECTURE STREQUAL "gfx1150")
        set(_validated_default ON)
    endif()
    option(LFS_HIP_VALIDATED_TUNING "Enable the validated gfx1150 HIP kernel profile" ${_validated_default})
    if(LFS_HIP_VALIDATED_TUNING AND NOT LFS_HIP_ARCHITECTURE STREQUAL "gfx1150")
        message(FATAL_ERROR "The validated HIP tuning profile currently targets gfx1150; use LFS_HIP_VALIDATED_TUNING=OFF for other architectures")
    endif()
    option(LFS_HIP_FAST_NATIVE_WAVE "Use native full-wave collectives in private wave32 FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_APPROX_FUNCTIONS "Test selective approximate math in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_DEVICE_ATOMICS "Use hardware FP32 atomics on HIP-owned coarse-grained gradient buffers" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_WAVE_REDUCE "Use AMD device-library full-wave FP32 reductions" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_INTRINSIC_MATH "Use selected FP32 exponential and bounded division intrinsics" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_SCATTER_REDUCE "Distribute reduced gradient fields across a HIP wave" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_PACKED_GRADIENTS "Use packed 16-field screen-space gradients in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_WARP_STAGING "Stage backward batches independently in native HIP wave32 registers" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_ADJACENT_PIXELS "Pair adjacent backward subtiles in private HIP wave staging" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_UNIFORM_BROADCAST "Read uniform staged HIP wave payloads through scalar lanes" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_FORWARD_EIGHT_PIXELS "Use single-wave eight-pixel RGB forward in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_BUTTERFLY_FIELDS "Use field-distributed reductions in private HIP RGB backward" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_EIGHT_PIXELS "Use a single-wave eight-pixel RGB backward in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_FOUR_PIXELS "Use guarded four-pixel RGB backward in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_FORWARD_ADJACENT "Pair adjacent forward subtiles in private HIP FastGS" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_RGB_SPECIALIZATION "Specialize HIP backward when depth and edge outputs are absent" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_FACTOR_ADJOINT "Reuse the color adjoint in private HIP FastGS backward" ${LFS_HIP_VALIDATED_TUNING})
    option(LFS_HIP_FAST_RADIX_TUNING "Use experimental gfx1150-specific FastGS radix configuration" ${LFS_HIP_VALIDATED_TUNING})
    # Generic mode is authoritative even when a reused cache contains tuned
    # switches from an earlier build. Keep cache values for expert diagnostics.
    if(NOT LFS_HIP_VALIDATED_TUNING)
        foreach(_option IN ITEMS
            LFS_HIP_FAST_NATIVE_WAVE
            LFS_HIP_FAST_APPROX_FUNCTIONS
            LFS_HIP_FAST_DEVICE_ATOMICS
            LFS_HIP_FAST_WAVE_REDUCE
            LFS_HIP_FAST_INTRINSIC_MATH
            LFS_HIP_FAST_SCATTER_REDUCE
            LFS_HIP_FAST_PACKED_GRADIENTS
            LFS_HIP_FAST_WARP_STAGING
            LFS_HIP_FAST_ADJACENT_PIXELS
            LFS_HIP_FAST_UNIFORM_BROADCAST
            LFS_HIP_FAST_FORWARD_EIGHT_PIXELS
            LFS_HIP_FAST_BUTTERFLY_FIELDS
            LFS_HIP_FAST_EIGHT_PIXELS
            LFS_HIP_FAST_FOUR_PIXELS
            LFS_HIP_FAST_FORWARD_ADJACENT
            LFS_HIP_FAST_RGB_SPECIALIZATION
            LFS_HIP_FAST_FACTOR_ADJOINT
            LFS_HIP_FAST_RADIX_TUNING
        )
            set(${_option} OFF)
        endforeach()
    endif()
    if(LFS_HIP_FAST_UNIFORM_BROADCAST AND (NOT LFS_HIP_FAST_WARP_STAGING OR NOT LFS_HIP_FAST_NATIVE_WAVE))
        message(FATAL_ERROR "HIP uniform payload reads require native wave staging")
    endif()
    if(LFS_HIP_FAST_FORWARD_EIGHT_PIXELS AND NOT LFS_HIP_FAST_UNIFORM_BROADCAST)
        message(FATAL_ERROR "HIP single-wave forward requires uniform native payload reads")
    endif()
    if(LFS_HIP_FAST_BUTTERFLY_FIELDS AND (NOT LFS_HIP_FAST_EIGHT_PIXELS OR NOT LFS_HIP_FAST_SCATTER_REDUCE))
        message(FATAL_ERROR "HIP butterfly fields require the scattered eight-pixel backward")
    endif()
    if(LFS_HIP_FAST_FOUR_PIXELS AND (NOT LFS_HIP_FAST_ADJACENT_PIXELS OR NOT LFS_HIP_FAST_SCATTER_REDUCE))
        message(FATAL_ERROR "HIP four-pixel backward requires adjacent wave staging and scattered reductions")
    endif()
    if(LFS_HIP_FAST_EIGHT_PIXELS AND NOT LFS_HIP_FAST_FOUR_PIXELS)
        message(FATAL_ERROR "HIP eight-pixel backward requires the four-pixel clipping layout")
    endif()
    if(LFS_HIP_FAST_ADJACENT_PIXELS AND NOT LFS_HIP_FAST_WARP_STAGING)
        message(FATAL_ERROR "HIP adjacent backward subtiles require private wave staging")
    endif()
    if(LFS_HIP_FAST_RADIX_TUNING AND NOT LFS_HIP_ARCHITECTURE STREQUAL "gfx1150")
        message(FATAL_ERROR "HIP Fast radix tuning is currently validated only for gfx1150")
    endif()
    if(NOT TARGET lfs_hip_fast_kernels)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
        set(_generator "${_root}/src/training/hip/prepare_fast.py")
        set(_generated "${CMAKE_CURRENT_BINARY_DIR}/hip-fast-generated")
        set(_inputs src/training/kernels/warp_reduce.cuh src/training/kernels/effective_rank.cuh
            src/training/kernels/joint_adam_codec.cuh src/training/include/lfs/training/mean_step_scale.cuh
            src/training/include/lfs/training/screen_share.cuh src/core/include/core/sh_value_codec.cuh
            src/training/include/lfs/training/ops/domain_types.hpp)
        set(_names backward.h forward.h buffer_utils.h fused_adam_types.h kernel_utils.cuh
            kernels_backward.cuh kernels_forward.cuh rasterization_config.h visibility.cuh rasterization_api.h pop_scores.h)
        foreach(_name IN LISTS _names)
            list(APPEND _inputs "src/training/rasterization/fastgs/rasterization/include/${_name}")
        endforeach()
        list(APPEND _inputs src/training/rasterization/fastgs/utils/helper_math.h
            src/training/rasterization/fastgs/utils/utils.h
            src/training/rasterization/fastgs/rasterization/src/forward.cu
            src/training/rasterization/fastgs/rasterization/src/backward.cu
            src/training/rasterization/fastgs/rasterization/src/pop_scores.cu)
        set(_outputs)
        foreach(_input IN LISTS _inputs)
            get_filename_component(_name "${_input}" NAME)
            list(APPEND _outputs "${_generated}/${_name}")
        endforeach()
        list(APPEND _outputs "${_generated}/hip_fast_diagnostics.hpp")
        list(APPEND _outputs "${_generated}/hip_fast_wave.hpp")
        list(APPEND _outputs "${_generated}/fast_instance_sort.hpp")
        list(APPEND _outputs "${_generated}/fast_depth_sort.hpp")
        list(APPEND _outputs "${_generated}/fast_tile_workload.hpp")
        list(TRANSFORM _inputs PREPEND "${_root}/")
        add_custom_command(OUTPUT "${_generated}/prepared.stamp"
            BYPRODUCTS ${_outputs}
            COMMAND "${Python3_EXECUTABLE}" "${_generator}" --root "${_root}" --output "${_generated}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${_generated}/prepared.stamp"
            DEPENDS "${_generator}" "${_root}/src/training/hip/fast_warp_staging.py" "${_root}/src/training/hip/fast_four_pixels.py" "${_root}/src/training/hip/fast_forward_pixels.py" "${_root}/src/training/hip/fast_compact_sort.py" "${_root}/src/training/hip/fast_instance_sort.hpp" "${_root}/src/training/hip/fast_depth_sort.hpp" "${_root}/src/training/hip/fast_tile_workload.hpp" ${_inputs} VERBATIM)
        if(CMAKE_HOST_WIN32)
            set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
            set(_extension obj)
        else()
            set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
            set(_extension o)
            set(_platform_flags -fPIC)
        endif()
        set(_objects)
        set(_wave_flags)
        if(LFS_HIP_FAST_APPROX_FUNCTIONS)
            # Keep NaN/Inf assumptions and denormal mode unchanged. Do not
            # substitute global fast-math flags; validate each private kernel.
            set(_wave_flags -fapprox-func -freciprocal-math -ffp-contract=fast)
        endif()
        if(LFS_HIP_FAST_NATIVE_WAVE)
            list(APPEND _wave_flags -DLFS_HIP_FAST_NATIVE_WAVE)
        endif()
        if(LFS_HIP_FAST_DEVICE_ATOMICS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_DEVICE_ATOMICS)
        endif()
        if(LFS_HIP_FAST_WAVE_REDUCE)
            list(APPEND _wave_flags -DLFS_HIP_FAST_WAVE_REDUCE)
        endif()
        if(LFS_HIP_FAST_INTRINSIC_MATH)
            list(APPEND _wave_flags -DLFS_HIP_FAST_INTRINSIC_MATH)
        endif()
        if(LFS_HIP_FAST_SCATTER_REDUCE)
            list(APPEND _wave_flags -DLFS_HIP_FAST_SCATTER_REDUCE)
        endif()
        if(LFS_HIP_FAST_PACKED_GRADIENTS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_PACKED_GRADIENTS)
        endif()
        if(LFS_HIP_FAST_WARP_STAGING)
            list(APPEND _wave_flags -DLFS_HIP_FAST_WARP_STAGING)
        endif()
        if(LFS_HIP_FAST_ADJACENT_PIXELS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_ADJACENT_PIXELS)
        endif()
        if(LFS_HIP_FAST_UNIFORM_BROADCAST)
            list(APPEND _wave_flags -DLFS_HIP_FAST_UNIFORM_BROADCAST)
        endif()
        if(LFS_HIP_FAST_FORWARD_EIGHT_PIXELS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_FORWARD_EIGHT_PIXELS)
        endif()
        if(LFS_HIP_FAST_BUTTERFLY_FIELDS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_BUTTERFLY_FIELDS)
        endif()
        if(LFS_HIP_FAST_EIGHT_PIXELS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_EIGHT_PIXELS)
        endif()
        if(LFS_HIP_FAST_FOUR_PIXELS)
            list(APPEND _wave_flags -DLFS_HIP_FAST_FOUR_PIXELS)
        endif()
        if(LFS_HIP_FAST_FORWARD_ADJACENT)
            list(APPEND _wave_flags -DLFS_HIP_FAST_FORWARD_ADJACENT)
        endif()
        if(LFS_HIP_FAST_RGB_SPECIALIZATION)
            list(APPEND _wave_flags -DLFS_HIP_FAST_RGB_SPECIALIZATION)
        endif()
        if(LFS_HIP_FAST_FACTOR_ADJOINT)
            list(APPEND _wave_flags -DLFS_HIP_FAST_FACTOR_ADJOINT)
        endif()
        if(LFS_HIP_FAST_RADIX_TUNING)
            list(APPEND _wave_flags -DLFS_HIP_FAST_RADIX_TUNING)
        endif()
        foreach(_unit forward backward pop_scores)
            set(_object "${_generated}/${_unit}.${_extension}")
            add_custom_command(OUTPUT "${_object}"
                COMMAND "${_compiler}" -x hip -std=c++20 -O3 -DNDEBUG ${_platform_flags} ${_wave_flags}
                    "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
                    "-I${_generated}" -c "${_generated}/${_unit}.cu" -o "${_object}"
                DEPENDS ${_outputs} "${_compiler}" VERBATIM)
            list(APPEND _objects "${_object}")
        endforeach()
        add_custom_target(lfs_hip_fast_kernels DEPENDS "${_generated}/prepared.stamp" ${_objects})
        set_target_properties(lfs_hip_fast_kernels PROPERTIES
            LFS_HIP_INCLUDE "${_generated}" LFS_HIP_OBJECTS "${_objects}")
    endif()
    get_target_property(_include lfs_hip_fast_kernels LFS_HIP_INCLUDE)
    get_target_property(_objects lfs_hip_fast_kernels LFS_HIP_OBJECTS)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${objects_output} "${_objects}" PARENT_SCOPE)
endfunction()
