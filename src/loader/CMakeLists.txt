# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(gs_loader STATIC
        # Implementation file only (header is in include/loader/)
        loader.cpp

        # Internal implementation files
        # Core interfaces and types
        loader_interface.hpp
        loader_registry.hpp
        loader_service.hpp
        loader_service.cpp

        # Format implementations
        formats/ply.hpp
        formats/ply.cpp
        formats/colmap.hpp
        formats/colmap.cpp
        formats/transforms.hpp
        formats/transforms.cpp
        formats/sogs.hpp
        formats/sogs.cpp

        # Concrete loader implementations
        loaders/ply_loader.hpp
        loaders/ply_loader.cpp
        loaders/colmap_loader.hpp
        loaders/colmap_loader.cpp
        loaders/blender_loader.hpp
        loaders/blender_loader.cpp
        loaders/sogs_loader.hpp
        loaders/sogs_loader.cpp
        cache_image_loader.cpp
)

# Set include directories
target_include_directories(gs_loader
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

# Find the actual WebP library target
if(TARGET WebP::webpdecoder)
    set(WEBP_DECODE_LIB WebP::webpdecoder)
elseif(TARGET WebP::webp)
    set(WEBP_DECODE_LIB WebP::webp)
elseif(TARGET webpdecoder)
    set(WEBP_DECODE_LIB webpdecoder)
else()
    find_library(WEBP_DECODE_LIBRARY NAMES webpdecoder webp libwebpdecoder libwebp REQUIRED)
    set(WEBP_DECODE_LIB ${WEBP_DECODE_LIBRARY})
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

# Link dependencies
target_link_libraries(gs_loader
        PUBLIC
        gs_core            # Link to modular core library for SplatData, Camera, etc.
        PRIVATE
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        ${WEBP_DECODE_LIB}     # WebP for SOG decompression
        ${ARCHIVE_LIB}         # LibArchive for SOG bundle support
)

# Set C++ standard
target_compile_features(gs_loader PUBLIC cxx_std_23)

# Compiler options
target_compile_options(gs_loader PRIVATE
        $<$<CXX_COMPILER_FRONTEND_VARIANT:GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>:/W4>
)

# AVX2 support for ply.cpp
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if (COMPILER_SUPPORTS_AVX2)
        target_compile_options(gs_loader PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(gs_loader PRIVATE HAS_AVX2_SUPPORT)
    endif ()
endif ()

# Export only the public header
install(FILES ${CMAKE_SOURCE_DIR}/include/loader/loader.hpp
        DESTINATION include/loader)

message(STATUS "Loader module configured with SOG support")
message(STATUS "  WebP decoder library: ${WEBP_DECODE_LIB}")
message(STATUS "  Archive library: ${ARCHIVE_LIB}")
