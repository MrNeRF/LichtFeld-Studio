# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)
find_program(LFS_GPU_SLANGC NAMES slangc
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/shader-slang"
          "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shader-slang"
    PATH_SUFFIXES tools/shader-slang REQUIRED)

# One Slang module, any number of compute entries, and optional raster entries.
# Artifacts are embedded, so installed builds never depend on source/build paths.
# MSL is generated at build time; Metal loads it using the system compiler. This
# works with Command Line Tools, without the optional offline Metal Toolchain.
function(lfs_add_gpu_program target name)
    # FindPython variables are directory-scoped; callers may be sibling directories.
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    # DEFINES (NAME or NAME=VALUE) specialize the module; build each variant
    # as its own program.
    cmake_parse_arguments(PROGRAM "" "SOURCE" "COMPUTE;VERTEX;FRAGMENT;DEFINES" ${ARGN})
    list(TRANSFORM PROGRAM_DEFINES PREPEND "-D" OUTPUT_VARIABLE defines)
    get_filename_component(source "${PROGRAM_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(directory "${CMAKE_CURRENT_BINARY_DIR}/gpu_programs/${name}")
    file(MAKE_DIRECTORY "${directory}")
    set(outputs)
    set(embed_args)
    foreach(stage IN ITEMS COMPUTE VERTEX FRAGMENT)
        string(TOLOWER "${stage}" slang_stage)
        if(stage STREQUAL "COMPUTE")
            set(cpp_stage Compute)
        elseif(stage STREQUAL "VERTEX")
            set(cpp_stage Vertex)
        else()
            set(cpp_stage Fragment)
        endif()
        foreach(entry IN LISTS PROGRAM_${stage})
            if(LFS_TENSOR_VULKAN)
                set(output "${directory}/${entry}.spv")
                add_custom_command(OUTPUT "${output}" "${output}.json"
                    COMMAND "${LFS_GPU_SLANGC}" "${source}" ${defines} -entry "${entry}" -stage "${slang_stage}"
                        -target spirv -profile glsl_460 -emit-spirv-directly -fvk-use-entrypoint-name
                        -fvk-use-scalar-layout -fp-mode precise -line-directive-mode none -o "${output}" -reflection-json "${output}.json"
                    DEPENDS "${source}" "${LFS_GPU_SLANGC}" VERBATIM)
                list(APPEND outputs "${output}" "${output}.json")
                list(APPEND embed_args Vulkan "${cpp_stage}" "${entry}" "${output}" "${output}.json")
            endif()
            if(LFS_TENSOR_METAL)
                set(output "${directory}/${entry}.metal")
                add_custom_command(OUTPUT "${output}" "${output}.json"
                    COMMAND "${LFS_GPU_SLANGC}" "${source}" ${defines} -entry "${entry}" -stage "${slang_stage}"
                        -target metal -fp-mode precise -line-directive-mode none -o "${output}" -reflection-json "${output}.json"
                    DEPENDS "${source}" "${LFS_GPU_SLANGC}" VERBATIM)
                list(APPEND outputs "${output}" "${output}.json")
                list(APPEND embed_args Metal "${cpp_stage}" "${entry}" "${output}" "${output}.json")
            endif()
            if(LFS_HAS_CUDA AND stage STREQUAL "COMPUTE")
                set(cuda_source "${directory}/${entry}.cu")
                set(output "${directory}/${entry}.ptx")
                add_custom_command(OUTPUT "${output}" "${output}.json" BYPRODUCTS "${cuda_source}"
                    COMMAND "${LFS_GPU_SLANGC}" "${source}" ${defines} -entry "${entry}" -stage compute
                        -target cuda -fp-mode precise -line-directive-mode none -o "${cuda_source}" -reflection-json "${output}.json"
                    COMMAND "${CMAKE_CUDA_COMPILER}" --ptx --std=c++17 --fmad=false
                        "${cuda_source}" -o "${output}"
                    DEPENDS "${source}" "${LFS_GPU_SLANGC}" VERBATIM)
                list(APPEND outputs "${output}" "${output}.json")
                list(APPEND embed_args CUDA Compute "${entry}" "${output}" "${output}.json")
            endif()
        endforeach()
    endforeach()
    add_custom_command(OUTPUT "${directory}/${name}.cpp" "${directory}/${name}.hpp"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/embed_gpu_program.py"
            "${directory}" "${name}" ${embed_args}
        DEPENDS ${outputs} "${CMAKE_SOURCE_DIR}/tools/embed_gpu_program.py" VERBATIM)
    # The syntax-only Windows preflight builds lfs_shader_headers before replaying
    # compile commands, so the generated header must be reachable from it.
    string(MAKE_C_IDENTIFIER "${target}_${name}_gpu_program" program_target)
    add_custom_target("${program_target}" DEPENDS "${directory}/${name}.cpp" "${directory}/${name}.hpp")
    add_dependencies(${target} "${program_target}")
    if(NOT TARGET lfs_shader_headers)
        add_custom_target(lfs_shader_headers)
    endif()
    add_dependencies(lfs_shader_headers "${program_target}")
    target_sources(${target} PRIVATE "${directory}/${name}.cpp")
    target_include_directories(${target} PRIVATE "${directory}")
endfunction()
