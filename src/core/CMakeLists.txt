# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# core module - New LibTorch-free core module with lfs::core namespace
# ============================================================================

# Add tensor library
add_subdirectory(tensor)

# Add unified CUDA library (private to core module)
# Contains: memory_arena, lanczos_resize, core_kernels (kmeans, morton)
add_subdirectory(cuda)

# Core module - fundamental data structures and utilities
# Note: lfs_tensor library is now in src/core/tensor
set(CORE_SOURCES
        argument_parser.cpp
        camera.cpp
        converter.cpp
        image_io.cpp
        logger.cpp
        parameters.cpp
        splat_data.cpp
        splat_data_export.cpp
        splat_data_mirror.cpp
        splat_data_transform.cpp
        sogs.cpp
        tensor_debug.cpp
        tinyply.cpp
        training_snapshot.cpp
)

add_library(lfs_core STATIC ${CORE_SOURCES})

target_include_directories(lfs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include  # For generated config.h
        ${CMAKE_SOURCE_DIR}/src/training/rasterization/gsplat  # For Common.h (camera model types)
        ${CMAKE_SOURCE_DIR}/fastgs    # For fastgs headers
        ${CUDAToolkit_INCLUDE_DIRS}   # For CUDA interop in Camera
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src       # For inter-module includes (training/, visualizer/, etc.)
        ${CMAKE_SOURCE_DIR}/external  # For tinyply and STB headers
        ${OPENGL_INCLUDE_DIRS}
)

# Find the actual WebP library target
if(TARGET WebP::webp)
    set(WEBP_LIB WebP::webp)
elseif(TARGET WebP::libwebp)
    set(WEBP_LIB WebP::libwebp)
elseif(TARGET webp)
    set(WEBP_LIB webp)
else()
    find_library(WEBP_LIBRARY NAMES webp libwebp REQUIRED)
    set(WEBP_LIB ${WEBP_LIBRARY})
endif()

# Find the actual LibArchive library target
if(TARGET LibArchive::LibArchive)
    set(ARCHIVE_LIB LibArchive::LibArchive)
elseif(TARGET archive)
    set(ARCHIVE_LIB archive)
elseif(TARGET archive_static)
    set(ARCHIVE_LIB archive_static)
else()
    find_library(ARCHIVE_LIBRARY NAMES archive libarchive REQUIRED)
    set(ARCHIVE_LIB ${ARCHIVE_LIBRARY})
endif()

target_link_libraries(lfs_core
    PUBLIC
        lfs_core_cuda   # Unified CUDA library (tensor, memory arena, kernels, lanczos)
        lfs_geometry    # For events.hpp -> bounding_box.hpp
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        Threads::Threads
        CUDA::cudart    # For Camera CUDA operations
        spdlog::spdlog
        ${WEBP_LIB}     # WebP for SOG compression
        ${ARCHIVE_LIB}  # LibArchive for SOG bundle support
        lfs_io          # For CacheLoader used in camera.cpp
    PRIVATE
        taywee::args    # Only used in argument_parser.cpp
        # NOTE: lfs_training, lfs_visualizer, lfs_project removed - all create circular dependencies
        # These should be linked at the application level, not in the core library
)

# Platform-specific settings
if(UNIX)
    target_link_libraries(lfs_core PUBLIC dl)
endif()

# Compiler options
if(MSVC)
    target_compile_options(lfs_core PRIVATE
        $<$<CONFIG:Debug>:/Od /Z7>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(lfs_core PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )
endif()

# AVX2 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(lfs_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(lfs_core PRIVATE HAS_AVX2_SUPPORT)
        message(STATUS "✓ AVX2 support enabled for gs_core")
    else()
        message(WARNING "✗ Compiler does not support AVX2")
    endif()
endif()

# Debug symbols in debug builds
set_target_properties(lfs_core PROPERTIES DEBUG_POSTFIX d)

# Set properties
set_target_properties(lfs_core PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    EXPORT_COMPILE_COMMANDS ON
)

# Configure build type
function(configure_build_type target)
    get_target_property(target_type ${target} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} INTERFACE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} INTERFACE RELEASE_BUILD)
        endif()
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
        endif()
    endif()
endfunction()

configure_build_type(lfs_core)

# Allocation profiling
if(ENABLE_ALLOCATION_PROFILING)
    target_compile_definitions(lfs_core PUBLIC ENABLE_ALLOCATION_PROFILING=1)
    message(STATUS "  • Allocation profiling: ENABLED")
else()
    target_compile_definitions(lfs_core PUBLIC ENABLE_ALLOCATION_PROFILING=0)
    message(STATUS "  • Allocation profiling: DISABLED")
endif()

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  core - LibTorch-free Core Module                         ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Namespace: lfs::core")
message(STATUS "  • Tensor library: lfs_tensor")
