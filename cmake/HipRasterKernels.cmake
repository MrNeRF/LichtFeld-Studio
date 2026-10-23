# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

function(lfs_hip_raster_objects include_output objects_output)
    if(NOT TARGET lfs_hip_raster_kernels)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        find_path(LFS_HIP_GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)
        get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
        set(_source "${_root}/src/training/rasterization/gsplat")
        set(_generator "${_root}/src/training/hip/prepare_gsplat.py")
        set(_generated "${CMAKE_CURRENT_BINARY_DIR}/hip-raster-generated")
        set(_names Common.h Cameras.h Cameras.cuh FromWorldRay.cuh Rasterization.h
            TileBatch.h Utils.cuh RasterizeToPixelsFromWorld3DGSFwd.cu
            RasterizeToPixelsFromWorld3DGSBwd.cu Projection.h Intersect.h
            ProjectionUT3DGSFused.cu IntersectTile.cu SphericalHarmonics.h SphericalHarmonicsCUDA.cu GeometryFeatures.h GeometryFeatures.cu)
        list(APPEND _names PopScores.h PopScores.cu)
        set(_inputs ${_names})
        list(TRANSFORM _inputs PREPEND "${_source}/")
        set(_outputs ${_names} hip_check.hpp logical_subgroup32.hpp)
        list(TRANSFORM _outputs PREPEND "${_generated}/")
        add_custom_command(OUTPUT "${_generated}/prepared.stamp"
            BYPRODUCTS ${_outputs}
            COMMAND "${Python3_EXECUTABLE}" "${_generator}" --source "${_source}" --output "${_generated}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${_generated}/prepared.stamp"
            DEPENDS "${_generator}" "${_root}/src/training/hip/logical_subgroup32.hpp" ${_inputs} VERBATIM)
        file(MAKE_DIRECTORY "${_generated}")
        file(GENERATE OUTPUT "${_generated}/raster.hip" CONTENT
            "#include \"RasterizeToPixelsFromWorld3DGSFwd.cu\"\n#include \"RasterizeToPixelsFromWorld3DGSBwd.cu\"\n")
        file(GENERATE OUTPUT "${_generated}/binning.hip" CONTENT
            "#include \"ProjectionUT3DGSFused.cu\"\n#include \"IntersectTile.cu\"\n")
        file(GENERATE OUTPUT "${_generated}/sh.hip" CONTENT
            "#include \"SphericalHarmonicsCUDA.cu\"\n")
        file(GENERATE OUTPUT "${_generated}/geometry.hip" CONTENT
            "#include \"GeometryFeatures.cu\"\n")
        file(GENERATE OUTPUT "${_generated}/pop_scores.hip" CONTENT
            "#include \"PopScores.cu\"\n")
        if(CMAKE_HOST_WIN32)
            set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
            set(_extension obj)
        else()
            set(_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
            set(_extension o)
            set(_platform_flags -fPIC)
        endif()
        include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HipArchitectures.cmake")
        lfs_hip_architectures()
        if(NOT EXISTS "${_compiler}")
            message(FATAL_ERROR "HIP raster kernels need LFS_HIP_ROOT")
        endif()
        set(_headers ${_names})
        list(FILTER _headers INCLUDE REGEX "\\.(h|cuh)$")
        list(APPEND _headers hip_check.hpp)
        list(TRANSFORM _headers PREPEND "${_generated}/")
        set(_raster_sources RasterizeToPixelsFromWorld3DGSFwd.cu RasterizeToPixelsFromWorld3DGSBwd.cu)
        set(_binning_sources ProjectionUT3DGSFused.cu IntersectTile.cu)
        set(_sh_sources SphericalHarmonicsCUDA.cu)
        set(_geometry_sources GeometryFeatures.cu)
        set(_pop_scores_sources PopScores.cu)
        set(_objects)
        foreach(_unit raster binning sh geometry pop_scores)
            set(_object "${_generated}/${_unit}.${_extension}")
            set(_unit_inputs ${_${_unit}_sources})
            list(TRANSFORM _unit_inputs PREPEND "${_generated}/")
            add_custom_command(OUTPUT "${_object}"
                COMMAND "${_compiler}" -x hip -std=c++20 -O3 ${_platform_flags}
                    ${LFS_HIP_OFFLOAD_FLAGS} ${LFS_HIP_COMPRESSION_FLAGS} ${LFS_HIP_TEST_WAVE_FLAGS} "--hip-path=${LFS_HIP_ROOT}"
                    "-I${_generated}" "-I${LFS_HIP_GLM_INCLUDE_DIR}"
                    -c "${_generated}/${_unit}.hip" -o "${_object}"
                DEPENDS ${LFS_HIP_TEST_RUNTIME_DEPENDS} ${_unit_inputs} ${_headers} "${_generated}/${_unit}.hip" "${_compiler}"
                    "${LFS_HIP_ROOT}/include/hip/hip_runtime.h" VERBATIM)
            list(APPEND _objects "${_object}")
        endforeach()
        add_custom_target(lfs_hip_raster_kernels DEPENDS "${_generated}/prepared.stamp" ${_objects})
        set_target_properties(lfs_hip_raster_kernels PROPERTIES
            LFS_HIP_INCLUDE "${_generated}" LFS_HIP_OBJECTS "${_objects}")
    endif()
    get_target_property(_include lfs_hip_raster_kernels LFS_HIP_INCLUDE)
    get_target_property(_objects lfs_hip_raster_kernels LFS_HIP_OBJECTS)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${objects_output} "${_objects}" PARENT_SCOPE)
endfunction()
