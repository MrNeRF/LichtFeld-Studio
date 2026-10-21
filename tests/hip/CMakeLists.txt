# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Standalone bring-up project. Never included by the application build.
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipBringup LANGUAGES NONE)

option(LFS_HIP_BUILD_PROBES "Compile isolated HIP bring-up kernels" OFF)
if(NOT LFS_HIP_BUILD_PROBES)
    message(STATUS "HIP probes disabled; no SDK or compiler is required")
    return()
endif()

set(LFS_HIP_ROOT "" CACHE PATH "Explicit ROCm SDK root for this isolated project")
set(LFS_HIP_ARCHITECTURE "gfx1150" CACHE STRING "Single AMD target for bring-up")
if(NOT IS_DIRECTORY "${LFS_HIP_ROOT}/include/hip")
    message(FATAL_ERROR "LFS_HIP_ROOT must contain the selected SDK's include/hip directory")
endif()
if(NOT LFS_HIP_ARCHITECTURE MATCHES "^gfx[0-9a-f]+$")
    message(FATAL_ERROR "Set one explicit LFS_HIP_ARCHITECTURE, for example gfx1150")
endif()
if(CMAKE_HOST_WIN32)
    set(_hip_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
else()
    set(_hip_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
endif()
if(NOT EXISTS "${_hip_compiler}")
    message(FATAL_ERROR "HIP compiler not found at ${_hip_compiler}")
endif()
# An explicit invocation avoids CMake's CXX module scanner overriding -x hip
# and avoids changing the C++ compiler/flags of any containing application.
if(CMAKE_HOST_WIN32)
    set(_hip_smoke "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_kernel_smoke.exe")
else()
    set(_hip_smoke "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_kernel_smoke")
endif()
add_custom_command(OUTPUT "${_hip_smoke}"
    COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3
        "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel_smoke.hip" -o "${_hip_smoke}"
    DEPENDS kernel_smoke.hip "${_hip_compiler}"
        "${LFS_HIP_ROOT}/include/hip/hip_runtime.h"
    VERBATIM)
add_custom_target(hip_bringup ALL DEPENDS "${_hip_smoke}")

enable_testing()
option(LFS_HIP_PROBE_FAST_MATH "Test private FastGS HIP mathematical dependencies" OFF)
option(LFS_HIP_PROBE_FAST_RASTER "Test private FastGS HIP forward and backward" OFF)
if(LFS_HIP_PROBE_FAST_MATH OR LFS_HIP_PROBE_FAST_RASTER)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HipFastKernels.cmake")
    lfs_hip_fast_objects(_fast_generated _fast_objects)
    set(_fast_wave_flags)
    if(LFS_HIP_FAST_APPROX_FUNCTIONS)
        set(_fast_wave_flags -fapprox-func -freciprocal-math -ffp-contract=fast)
    endif()
    if(LFS_HIP_FAST_NATIVE_WAVE)
        list(APPEND _fast_wave_flags -DLFS_HIP_FAST_NATIVE_WAVE)
    endif()
    if(LFS_HIP_FAST_WAVE_REDUCE)
        list(APPEND _fast_wave_flags -DLFS_HIP_FAST_WAVE_REDUCE)
    endif()
    if(LFS_HIP_FAST_UNIFORM_BROADCAST)
        list(APPEND _fast_wave_flags -DLFS_HIP_FAST_UNIFORM_BROADCAST)
    endif()
    if(LFS_HIP_FAST_PACKED_GRADIENTS)
        list(APPEND _fast_wave_flags -DLFS_HIP_FAST_PACKED_GRADIENTS)
    endif()
    foreach(_probe math raster)
        string(TOUPPER "${_probe}" _probe_upper)
        if(NOT LFS_HIP_PROBE_FAST_${_probe_upper})
            continue()
        endif()
        set(_fast_executable "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_fast_${_probe}")
        if(CMAKE_HOST_WIN32)
            string(APPEND _fast_executable ".exe")
        endif()
        set(_link_objects)
        if(_probe STREQUAL "raster")
            set(_link_objects ${_fast_objects})
        endif()
        add_custom_command(OUTPUT "${_fast_executable}"
            COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3 -DNDEBUG ${_fast_wave_flags}
                "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
                "-I${_fast_generated}" "${CMAKE_CURRENT_SOURCE_DIR}/fast_${_probe}.hip"
                -x none ${_link_objects} -o "${_fast_executable}"
            DEPENDS "fast_${_probe}.hip" ${_fast_objects} "${_hip_compiler}" VERBATIM)
        add_custom_target(hip_fast_${_probe} ALL DEPENDS "${_fast_executable}")
        add_dependencies(hip_fast_${_probe} lfs_hip_fast_kernels)
        add_test(NAME HipBringup.Fast_${_probe} COMMAND "${_fast_executable}")
        set_tests_properties(HipBringup.Fast_${_probe} PROPERTIES TIMEOUT 120)
    endforeach()
endif()
add_test(NAME HipBringup.CompiledKernels COMMAND "${_hip_smoke}")
set_tests_properties(HipBringup.CompiledKernels PROPERTIES TIMEOUT 120)

option(LFS_HIP_PROBE_GSPLAT_TRAINING "Test wave32 gsplat differentiation and toy training" OFF)
if(LFS_HIP_PROBE_GSPLAT_TRAINING)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HipRasterKernels.cmake")
    lfs_hip_raster_objects(_gsplat_generated _hip_raster_objects)
    list(GET _hip_raster_objects 0 _gsplat_object)
    list(GET _hip_raster_objects 1 _binning_object)
    if(CMAKE_HOST_WIN32)
        set(_gsplat_training "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_training.exe")
    else()
        set(_gsplat_training "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_training")
    endif()
    set(_gsplat_flags -std=c++20 -O3 "--offload-arch=${LFS_HIP_ARCHITECTURE}"
        "--hip-path=${LFS_HIP_ROOT}" "-I${_gsplat_generated}" "-I${LFS_HIP_GLM_INCLUDE_DIR}")
    add_custom_command(OUTPUT "${_gsplat_training}"
        COMMAND "${_hip_compiler}" -x hip ${_gsplat_flags}
            "${CMAKE_CURRENT_SOURCE_DIR}/gsplat_training.hip" -x none "${_gsplat_object}"
            -o "${_gsplat_training}"
        DEPENDS gsplat_training.hip "${_gsplat_object}"
        VERBATIM)
    add_custom_target(hip_gsplat_training ALL DEPENDS "${_gsplat_training}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-training-results")
    add_test(NAME HipTraining.GsplatGradientsAndConvergence COMMAND "${_gsplat_training}")
    set_tests_properties(HipTraining.GsplatGradientsAndConvergence PROPERTIES TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-training-results")
    if(CMAKE_HOST_WIN32)
        set(_gsplat_dense "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_training_dense.exe")
    else()
        set(_gsplat_dense "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_training_dense")
    endif()
    add_custom_command(OUTPUT "${_gsplat_dense}"
        COMMAND "${_hip_compiler}" -x hip ${_gsplat_flags} -DLFS_HIP_TEST_GAUSSIANS=263
            "${CMAKE_CURRENT_SOURCE_DIR}/gsplat_training.hip" -x none "${_gsplat_object}"
            -o "${_gsplat_dense}"
        DEPENDS gsplat_training.hip "${_gsplat_object}"
        VERBATIM)
    add_custom_target(hip_gsplat_training_dense ALL DEPENDS "${_gsplat_dense}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-training-dense-results")
    add_test(NAME HipTraining.GsplatBatchTailsAndConvergence COMMAND "${_gsplat_dense}")
    set_tests_properties(HipTraining.GsplatBatchTailsAndConvergence PROPERTIES TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-training-dense-results")
    if(CMAKE_HOST_WIN32)
        set(_binning_training "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_binning.exe")
    else()
        set(_binning_training "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_gsplat_binning")
    endif()
    add_custom_command(OUTPUT "${_binning_training}"
        COMMAND "${_hip_compiler}" -x hip ${_gsplat_flags}
            -DLFS_HIP_TEST_GAUSSIANS=263 -DLFS_HIP_TEST_BINNING=1
            "${CMAKE_CURRENT_SOURCE_DIR}/gsplat_training.hip" -x none
            "${_gsplat_object}" "${_binning_object}" -o "${_binning_training}"
        DEPENDS gsplat_training.hip gsplat_binning.hpp "${_gsplat_object}" "${_binning_object}"
        VERBATIM)
    add_custom_target(hip_gsplat_binning ALL DEPENDS "${_binning_training}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-binning-results")
    add_test(NAME HipTraining.GsplatDynamicBinningAndConvergence COMMAND "${_binning_training}")
    set_tests_properties(HipTraining.GsplatDynamicBinningAndConvergence PROPERTIES TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gsplat-binning-results")
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/tensor/backend/hip"
    "${CMAKE_CURRENT_BINARY_DIR}/runtime")
option(LFS_HIP_PROBE_TRAINING_SESSION "Build and test the HIP training session C ABI" OFF)
if(LFS_HIP_PROBE_TRAINING_SESSION)
    if(CMAKE_HOST_WIN32)
        enable_language(CXX)
    endif()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../src/training/hip"
        "${CMAKE_CURRENT_BINARY_DIR}/training-runtime")
    get_filename_component(_photo_include "${LFS_HIP_PHOTOMETRIC_HEADER}" DIRECTORY)
    set(_photo_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_photometric${CMAKE_EXECUTABLE_SUFFIX}")
    if(CMAKE_HOST_WIN32)
        set(_photo_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_photometric.exe")
    endif()
    add_custom_command(OUTPUT "${_photo_test}"
        COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3
            "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
            "-I${_photo_include}" "${CMAKE_CURRENT_SOURCE_DIR}/photometric.hip" -o "${_photo_test}"
        DEPENDS photometric.hip photometric_oracle.hpp "${LFS_HIP_PHOTOMETRIC_HEADER}" VERBATIM)
    add_custom_target(hip_photometric_contract ALL DEPENDS "${_photo_test}")
    add_test(NAME HipTraining.PhotometricCpuValueAndGradient COMMAND "${_photo_test}")
    set_tests_properties(HipTraining.PhotometricCpuValueAndGradient PROPERTIES TIMEOUT 120)
    get_target_property(_sh_objects lfs_hip_raster_kernels LFS_HIP_OBJECTS)
    get_target_property(_sh_include lfs_hip_raster_kernels LFS_HIP_INCLUDE)
    list(GET _sh_objects 2 _sh_object)
    set(_sh_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_sh${CMAKE_EXECUTABLE_SUFFIX}")
    if(CMAKE_HOST_WIN32)
        set(_sh_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_sh.exe")
    endif()
    add_custom_command(OUTPUT "${_sh_test}"
        COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3
            "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
            "-I${_sh_include}" "-I${LFS_HIP_GLM_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/spherical_harmonics.hip" -x none "${_sh_object}" -o "${_sh_test}"
        DEPENDS spherical_harmonics.hip sh_oracle.hpp "${_sh_object}" VERBATIM)
    add_custom_target(hip_sh_contract ALL DEPENDS "${_sh_test}")
    add_test(NAME HipTraining.SphericalHarmonicsValueGradientAndLayout COMMAND "${_sh_test}")
    set_tests_properties(HipTraining.SphericalHarmonicsValueGradientAndLayout PROPERTIES TIMEOUT 120)
    if(CMAKE_HOST_WIN32)
        find_program(_session_msvc NAMES cl REQUIRED)
        set(_session_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_training_session.exe")
        if(LFS_HIP_PROBE_CUDA_COEXISTENCE)
            set(_session_compile_flags /DLFS_HIP_SESSION_WITH_CUDA "/I${LFS_HIP_CUDA_ROOT}/include")
            set(_session_link_flags /link "${LFS_HIP_CUDA_ROOT}/lib/x64/cudart.lib")
        endif()
        add_custom_command(OUTPUT "${_session_test}"
            COMMAND "${_session_msvc}" /nologo /std:c++20 /EHsc /MD ${_session_compile_flags}
                "${CMAKE_CURRENT_SOURCE_DIR}/training_session.cpp"
                "$<TARGET_FILE:lfs_hip_training_host>"
                "/Fe:${_session_test}" "/Fo:${CMAKE_CURRENT_BINARY_DIR}/training_session.obj"
                ${_session_link_flags}
            DEPENDS training_session.cpp training_fixture.hpp training_session_fixture.hpp photometric_oracle.hpp training_sh_session.hpp sh_oracle.hpp
                "$<TARGET_FILE:lfs_hip_training_host>"
                "../../src/training/hip/session.cpp" "../../src/training/hip/session.hpp"
                "../../src/training/hip/training_api.h"
                "../../src/core/tensor/backend/hip/runtime_api.h"
            VERBATIM)
        set(_core_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_core_backend.exe")
        add_custom_command(OUTPUT "${_core_test}"
            COMMAND "${_session_msvc}" /nologo /std:c++latest /EHsc /MD /DLFS_HAS_CUDA=0 /DLFS_STATIC_BUILD /DLFS_TENSOR_HIP /DLFS_TENSOR_PRIVATE_ACCESS
                "/I${CMAKE_CURRENT_SOURCE_DIR}/../../src" "/I${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include" "/I${LFS_HIP_GLM_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/core_backend.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/tensor/backend/hip/hip_backend_ops.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/tensor/backend/hip/hip_unsupported_ops.cpp"
                "/Fe:${_core_test}" "/Fo:${CMAKE_CURRENT_BINARY_DIR}/"
            DEPENDS core_backend.cpp "../../src/core/tensor/backend/hip/hip_backend_ops.cpp"
                "../../src/core/tensor/backend/hip/hip_backend_ops.hpp" "../../src/core/tensor/backend/hip/hip_unsupported_ops.cpp"
                "../../src/core/tensor/backend/hip/runtime_api.h" VERBATIM)
        add_custom_target(hip_core_backend_contract ALL DEPENDS "${_core_test}")
        add_dependencies(hip_core_backend_contract lfs_hip_runtime_build)
        add_test(NAME HipTensor.CoreBackendPointwiseAndOwnership COMMAND "${_core_test}" "${LFS_HIP_RUNTIME_LIBRARY}")
        set_tests_properties(HipTensor.CoreBackendPointwiseAndOwnership PROPERTIES TIMEOUT 120)
        set(_storage_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_splat_storage_layout.exe")
        add_custom_command(OUTPUT "${_storage_test}"
            COMMAND "${_session_msvc}" /nologo /std:c++latest /EHsc /MD /DLFS_HAS_CUDA=0 /DLFS_STATIC_BUILD
                "/I${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include" "/I${LFS_HIP_GLM_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/splat_storage_layout.cpp"
                "/Fe:${_storage_test}" "/Fo:${CMAKE_CURRENT_BINARY_DIR}/splat_storage_layout.obj"
            DEPENDS splat_storage_layout.cpp "../../src/training/hip/splat_storage_layout.hpp"
                "../../src/core/include/core/splat_exportable_storage.hpp" VERBATIM)
        add_custom_target(hip_core_storage_contract ALL DEPENDS "${_storage_test}")
        add_test(NAME HipTraining.CoreSplatStorageLayoutContract COMMAND "${_storage_test}")
        add_custom_target(hip_training_session_contract ALL DEPENDS "${_session_test}")
        if(LFS_HIP_FAST_RUNTIME)
            add_executable(lfs_hip_host_targets host_targets.cpp)
            target_compile_features(lfs_hip_host_targets PRIVATE cxx_std_20)
            set_target_properties(lfs_hip_host_targets PROPERTIES CXX_SCAN_FOR_MODULES OFF)
            target_link_libraries(lfs_hip_host_targets PRIVATE lfs_hip_training_host)
            add_test(NAME HipTraining.OwnedHostTargets COMMAND lfs_hip_host_targets "${LFS_HIP_TRAINING_LIBRARY}")
            set_tests_properties(HipTraining.OwnedHostTargets PROPERTIES TIMEOUT 60)
            add_executable(lfs_hip_fast_session fast_session.cpp)
            target_compile_features(lfs_hip_fast_session PRIVATE cxx_std_20)
            set_target_properties(lfs_hip_fast_session PROPERTIES CXX_SCAN_FOR_MODULES OFF)
            target_link_libraries(lfs_hip_fast_session PRIVATE lfs_hip_training_host)
            add_test(NAME HipTraining.FastSessionConvergenceAndCancellation
                COMMAND lfs_hip_fast_session "${LFS_HIP_TRAINING_LIBRARY}")
            set_tests_properties(HipTraining.FastSessionConvergenceAndCancellation PROPERTIES TIMEOUT 180)
            set(_resident_sh_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_resident_sh.exe")
            add_custom_command(OUTPUT "${_resident_sh_test}"
                COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3 -fms-runtime-lib=dll
                    "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/resident_sh.hip" -x none
                    "$<TARGET_FILE:lfs_hip_training_host>" -o "${_resident_sh_test}"
                DEPENDS resident_sh.hip training_session_fixture.hpp training_fixture.hpp
                    lfs_hip_training_host VERBATIM)
            add_custom_target(hip_resident_sh_contract ALL DEPENDS "${_resident_sh_test}")
            add_dependencies(hip_resident_sh_contract lfs_hip_training_build)
            add_test(NAME HipTraining.ResidentShAliasingAndGuards
                COMMAND "${_resident_sh_test}" "${LFS_HIP_TRAINING_LIBRARY}")
            set_tests_properties(HipTraining.ResidentShAliasingAndGuards PROPERTIES TIMEOUT 120)
            add_test(NAME HipTraining.ResidentShFusedAliasingAndGuards
                COMMAND "${_resident_sh_test}" "${LFS_HIP_TRAINING_LIBRARY}" --fused)
            set_tests_properties(HipTraining.ResidentShFusedAliasingAndGuards PROPERTIES TIMEOUT 120)
        endif()
        add_dependencies(hip_training_session_contract lfs_hip_training_build lfs_hip_training_host)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/training-session-results")
        add_test(NAME HipTraining.SessionCAbiBudgetAndConcurrency
            COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}")
        set_tests_properties(HipTraining.SessionCAbiBudgetAndConcurrency PROPERTIES TIMEOUT 120
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/training-session-results")
        add_test(NAME HipTraining.HostAdapterLifetimeAndRecovery
            COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}" --host-adapter)
        set_tests_properties(HipTraining.HostAdapterLifetimeAndRecovery PROPERTIES TIMEOUT 120)
        add_test(NAME HipTraining.PhotometricSessionLossAndRecovery
            COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}" --photometric)
        set_tests_properties(HipTraining.PhotometricSessionLossAndRecovery PROPERTIES TIMEOUT 120)
        add_test(NAME HipTraining.ShSessionBudgetActivationAndConvergence
            COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}" --sh)
        set_tests_properties(HipTraining.ShSessionBudgetActivationAndConvergence PROPERTIES TIMEOUT 120)
        option(LFS_HIP_PROBE_LARGE_SESSION "Test one million model entries with a 512 MiB allocation budget" OFF)
        if(LFS_HIP_PROBE_LARGE_SESSION)
            add_test(NAME HipTraining.SessionOneMillionModelBudget
                COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}" --large-model)
            set_tests_properties(HipTraining.SessionOneMillionModelBudget PROPERTIES TIMEOUT 120)
            add_test(NAME HipTraining.ShSessionOneMillionBudget
                COMMAND "${_session_test}" "${LFS_HIP_TRAINING_LIBRARY}" --large-sh)
            set_tests_properties(HipTraining.ShSessionOneMillionBudget PROPERTIES TIMEOUT 120)
        endif()
        if(LFS_HIP_PROBE_CUDA_COEXISTENCE)
            set_tests_properties(HipTraining.SessionCAbiBudgetAndConcurrency HipTraining.HostAdapterLifetimeAndRecovery
                HipTraining.PhotometricSessionLossAndRecovery HipTraining.ShSessionBudgetActivationAndConvergence PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${LFS_HIP_CUDA_ROOT}/bin")
            if(LFS_HIP_PROBE_LARGE_SESSION)
                set_tests_properties(HipTraining.SessionOneMillionModelBudget HipTraining.ShSessionOneMillionBudget PROPERTIES
                    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${LFS_HIP_CUDA_ROOT}/bin")
            endif()
        endif()
    endif()
endif()
if(CMAKE_HOST_WIN32)
    find_program(_hip_host_compiler NAMES cl REQUIRED)
    set(_runtime_test "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_runtime_contract.exe")
    add_custom_command(OUTPUT "${_runtime_test}"
        COMMAND "${_hip_host_compiler}" /nologo /std:c++20 /EHsc /MD
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime_contract.cpp"
            "/Fe:${_runtime_test}" "/Fo:${CMAKE_CURRENT_BINARY_DIR}/runtime_contract.obj"
        DEPENDS runtime_contract.cpp shared_resize_contract.inc "../../src/core/tensor/backend/hip/runtime_api.h"
        VERBATIM)
    add_custom_target(hip_runtime_contract ALL DEPENDS "${_runtime_test}")
    add_dependencies(hip_runtime_contract lfs_hip_runtime_build)
    add_test(NAME HipRuntime.OwnershipQueuesAndBounds
        COMMAND "${_runtime_test}" "${LFS_HIP_RUNTIME_LIBRARY}")
    set_tests_properties(HipRuntime.OwnershipQueuesAndBounds PROPERTIES TIMEOUT 120)
endif()

option(LFS_HIP_PROBE_CUDA_COEXISTENCE "Test MSVC/CUDA host with Clang/HIP DLL" OFF)
if(LFS_HIP_PROBE_CUDA_COEXISTENCE)
    if(NOT CMAKE_HOST_WIN32)
        message(FATAL_ERROR "The MSVC coexistence probe is Windows-only")
    endif()
    set(LFS_HIP_CUDA_ROOT "" CACHE PATH "CUDA SDK root for the optional coexistence probe")
    if(NOT EXISTS "${LFS_HIP_CUDA_ROOT}/include/cuda_runtime_api.h")
        message(FATAL_ERROR "LFS_HIP_CUDA_ROOT must identify the CUDA SDK")
    endif()
    find_program(_hip_msvc NAMES cl REQUIRED)
    set(_hip_provider "${CMAKE_CURRENT_BINARY_DIR}/hip_probe.dll")
    set(_hip_coexistence "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_coexistence.exe")
    add_custom_command(OUTPUT "${_hip_provider}"
        COMMAND "${_hip_compiler}" -x hip -std=c++20 -O3 -shared
            "--offload-arch=${LFS_HIP_ARCHITECTURE}" "--hip-path=${LFS_HIP_ROOT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/provider.hip" -o "${_hip_provider}"
        DEPENDS provider.hip probe_api.h "${_hip_compiler}"
        VERBATIM)
    add_custom_command(OUTPUT "${_hip_coexistence}"
        COMMAND "${_hip_msvc}" /nologo /std:c++20 /EHsc /MD
            "/I${LFS_HIP_CUDA_ROOT}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/coexistence.cpp"
            "/Fe:${_hip_coexistence}"
            "/Fo:${CMAKE_CURRENT_BINARY_DIR}/coexistence.obj"
            /link "/LIBPATH:${LFS_HIP_CUDA_ROOT}/lib/x64" cudart.lib
        DEPENDS coexistence.cpp probe_api.h
        VERBATIM)
    add_custom_target(hip_cuda_coexistence ALL DEPENDS "${_hip_provider}" "${_hip_coexistence}")
    add_test(NAME HipBringup.CudaCoexistence COMMAND "${_hip_coexistence}" "${_hip_provider}")
    set_tests_properties(HipBringup.CudaCoexistence PROPERTIES TIMEOUT 120
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${LFS_HIP_CUDA_ROOT}/bin")
endif()

option(LFS_HIP_PROBE_VULKAN_INTEROP "Test experimental Windows HIP/Vulkan memory and GPU ordering" OFF)
if(LFS_HIP_PROBE_VULKAN_INTEROP)
    if(NOT CMAKE_HOST_WIN32)
        message(FATAL_ERROR "The native shared-fence bring-up probe is Windows-only")
    endif()
    find_program(_hip_msvc NAMES cl REQUIRED)
    find_package(Vulkan REQUIRED)
    set(_hip_interop "${CMAKE_CURRENT_BINARY_DIR}/lfs_hip_interop.exe")
    if(LFS_HIP_PROBE_TRAINING_SESSION)
        set(_interop_training_flags /DLFS_HIP_INTEROP_TRAINING)
        set(_interop_training_object "$<TARGET_FILE:lfs_hip_training_host>")
    endif()
    add_custom_command(OUTPUT "${_hip_interop}"
        COMMAND "${_hip_msvc}" /nologo /std:c++latest /DLFS_HAS_CUDA=0 /DLFS_STATIC_BUILD /EHsc /MD
            /D__HIP_PLATFORM_AMD__ ${_interop_training_flags}
            "/I${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include" "/I${LFS_HIP_GLM_INCLUDE_DIR}"
            "/I${LFS_HIP_ROOT}/include" "/I${Vulkan_INCLUDE_DIRS}"
            "${CMAKE_CURRENT_SOURCE_DIR}/interop_win32.cpp"
            ${_interop_training_object}
            "/Fe:${_hip_interop}" "/Fo:${CMAKE_CURRENT_BINARY_DIR}/interop_win32.obj"
            /link "${Vulkan_LIBRARY}" d3d12.lib dxgi.lib gdi32.lib
        DEPENDS interop_win32.cpp shared_model_training.hpp training_fixture.hpp
            "../../src/training/hip/splat_storage_layout.hpp" ${_interop_training_object}
            "${LFS_HIP_ROOT}/include/hip/hip_runtime_api.h"
        VERBATIM)
    add_custom_target(hip_vulkan_interop ALL DEPENDS "${_hip_interop}")
    add_test(NAME HipBringup.VulkanSharedMemoryGpuOrdering
        COMMAND "${_hip_interop}" --d3d12-wait-value "${LFS_HIP_ROOT}/bin/amdhip64_7.dll")
    set_tests_properties(HipBringup.VulkanSharedMemoryGpuOrdering PROPERTIES TIMEOUT 120)
    if(LFS_HIP_PROBE_TRAINING_SESSION)
        add_dependencies(hip_vulkan_interop lfs_hip_training_build)
        add_test(NAME HipTraining.VulkanSharedModelOneMillion
            COMMAND "${_hip_interop}" --training-shared-model "${LFS_HIP_ROOT}/bin/amdhip64_7.dll" "${LFS_HIP_TRAINING_LIBRARY}")
        set_tests_properties(HipTraining.VulkanSharedModelOneMillion PROPERTIES TIMEOUT 120)
        add_test(NAME HipTraining.VulkanSharedModelPhotometric
            COMMAND "${_hip_interop}" --training-shared-photo "${LFS_HIP_ROOT}/bin/amdhip64_7.dll" "${LFS_HIP_TRAINING_LIBRARY}")
        set_tests_properties(HipTraining.VulkanSharedModelPhotometric PROPERTIES TIMEOUT 120)
        add_test(NAME HipTraining.VulkanApplicationLayoutAndGuards
            COMMAND "${_hip_interop}" --training-shared-layout "${LFS_HIP_ROOT}/bin/amdhip64_7.dll" "${LFS_HIP_TRAINING_LIBRARY}")
        set_tests_properties(HipTraining.VulkanApplicationLayoutAndGuards PROPERTIES TIMEOUT 120)
    endif()
endif()
