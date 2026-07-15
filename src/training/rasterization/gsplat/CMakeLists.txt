cmake_minimum_required(VERSION 3.30)

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Ensure CUDA targets are available from parent
if(NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "CUDA::cudart not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

if(NOT TARGET CUDA::curand)
    message(FATAL_ERROR "CUDA::curand not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# All gsplat sources together
set(GSPLAT_SOURCES
        # C++ files
        Common.cpp
        Intersect.cpp
        Projection.cpp
        QuatToRotmat.cpp
        Rasterization.cpp
        Relocation.cpp
        SphericalHarmonics.cpp

        # CUDA files
        IntersectTile.cu
        ProjectionUT3DGSFused.cu
        QuatToRotmatCUDA.cu
        RasterizeToPixelsFromWorld3DGSBwd.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        RelocationCUDA.cu
        SphericalHarmonicsCUDA.cu
)

# lfs suffix avoids conflict with legacy gsplat_backend
add_library(gsplat_backend_lfs STATIC ${GSPLAT_SOURCES})

set_target_properties(gsplat_backend_lfs PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(gsplat_backend_lfs
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(gsplat_backend_lfs
        PUBLIC
        CUDA::cudart
        CUDA::curand
        glm::glm
)

target_compile_options(gsplat_backend_lfs PRIVATE
  # CUDA device code + MSVC host compiler flags (Windows only)
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -lineinfo --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221 -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

  # CUDA device code for non-Windows
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -lineinfo --expt-relaxed-constexpr>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

  # C++ MSVC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

  # C++ no-MSVC (GCC/Clang/AppleClang/IntelLLVM)
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

# GLM_COMPILER must not be set externally — platform.h redefines it unconditionally (C4005)
target_compile_definitions(gsplat_backend_lfs PRIVATE
    GLM_FORCE_CUDA
)

target_compile_definitions(gsplat_backend_lfs PUBLIC
    LFS_CUDA_FAILURE_INJECTION_ENABLED=$<BOOL:${LFS_ENABLE_CUDA_FAILURE_INJECTION}>)

# Print debug info
message(STATUS "gsplat_backend_lfs configuration:")
message(STATUS "  - Sources: ${GSPLAT_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - C++ Standard: 23")
message(STATUS "  - CUDA Standard: 20")
message(STATUS "  - CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
