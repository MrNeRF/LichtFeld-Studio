# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

set(glslang_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glslang"
    CACHE PATH "Path to vcpkg-installed glslang config" FORCE)
find_package(glslang CONFIG REQUIRED)

if(NOT TARGET lfs_shader_compiler)
    add_executable(lfs_shader_compiler EXCLUDE_FROM_ALL
        "${CMAKE_SOURCE_DIR}/src/visualizer/tools/shader_compiler.cpp")
    target_compile_features(lfs_shader_compiler PRIVATE cxx_std_23)
    target_link_libraries(lfs_shader_compiler
        PRIVATE
            glslang::glslang
            glslang::SPIRV
            glslang::glslang-default-resource-limits)
endif()
