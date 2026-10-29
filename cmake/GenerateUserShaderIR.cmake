# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Writes manifest.json beside finished IR modules for the runtime compiler,
# which loads modules from IR only and never reads Slang sources. The shipped
# package lists each host's IR, reflection, dependencies and imported modules.
# A test package instead holds one minimal host. The ABI hash covers every
# public API source and is also written as a C++ header, so a binary rejects
# packages built from a different API.
#
# cmake -DSLANGC=<slangc> -DSOURCE_DIR=<dir> -DOUTPUT_DIR=<dir>
#       -DREFLECTION_DIR=<dir> -DHOSTS=<name:define,...|...> -DABI_HEADER=<file>
#       | -DTEST_HOST=<module>
#       -P GenerateUserShaderIR.cmake

cmake_minimum_required(VERSION 3.30)

set(api_sources reveal_noise.inc lfs_helpers.slang lfs.slang)
set(api_modules lfs_helpers lfs) # dependency order
# The renderer pins every splat pipeline to config.slang's SUBGROUP_SIZE.
file(STRINGS "${SOURCE_DIR}/config.slang" host_subgroup_size REGEX "^#define SUBGROUP_SIZE [0-9]+$")
string(REGEX REPLACE "^#define SUBGROUP_SIZE " "" host_subgroup_size "${host_subgroup_size}")
if(NOT host_subgroup_size MATCHES "^[0-9]+$")
    message(FATAL_ERROR "config.slang must define SUBGROUP_SIZE once")
endif()
# Modules the hosts may import beside the API, in load order.
set(host_modules lfs_reveal spherical_harmonics utils)

# Hash a finished module; compilation belongs to its depfile-backed build rule.
function(module_entry name file out_var)
    file(SHA256 "${OUTPUT_DIR}/${file}.slang-module" hash)
    set(${out_var} "{\"name\": \"${name}\", \"file\": \"${file}.slang-module\", \"sha256\": \"${hash}\"}" PARENT_SCOPE)
endfunction()

# The manifest entry of a host IR module the build already compiled, with the
# interface its reflection reports.
function(host_entry host defines out_var)
    file(SHA256 "${OUTPUT_DIR}/${host}.slang-module" hash)
    file(READ "${REFLECTION_DIR}/${host}.json" reflection)
    set(bindings "")
    string(JSON count LENGTH "${reflection}" parameters)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
        string(JSON name GET "${reflection}" parameters ${i} name)
        string(JSON kind GET "${reflection}" parameters ${i} binding kind)
        string(JSON index GET "${reflection}" parameters ${i} binding index)
        string(JSON space ERROR_VARIABLE no_space GET "${reflection}" parameters ${i} binding space)
        if(no_space)
            set(space 0)
        endif()
        if(NOT kind STREQUAL "descriptorTableSlot")
            message(FATAL_ERROR "${host}: unsupported host parameter ${name} (${kind})")
        endif()
        list(APPEND bindings "{\"name\": \"${name}\", \"set\": ${space}, \"binding\": ${index}, \"count\": 1}")
    endforeach()
    list(JOIN bindings ", " bindings)
    # Entry-point uniforms are the push constants; varyings have no binding.
    set(push_constant_bytes 0)
    string(JSON count LENGTH "${reflection}" entryPoints 0 parameters)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
        string(JSON kind ERROR_VARIABLE varying GET "${reflection}" entryPoints 0 parameters ${i} binding kind)
        if(NOT varying AND kind STREQUAL "uniform")
            string(JSON size GET "${reflection}" entryPoints 0 parameters ${i} binding size)
            math(EXPR push_constant_bytes "${push_constant_bytes} + ${size}")
        endif()
    endforeach()
    set(workgroup "")
    foreach(axis 0 1 2)
        string(JSON size GET "${reflection}" entryPoints 0 threadGroupSize ${axis})
        list(APPEND workgroup ${size})
    endforeach()
    list(JOIN workgroup ", " workgroup)
    # The modules whose sources the host's compile read, as REFLECTION_DIR/<host>.deps
    # lists them. Banded hosts import utils built for banded export.
    file(READ "${REFLECTION_DIR}/${host}.deps" sources)
    set(imports "")
    foreach(module IN LISTS host_modules)
        if(NOT sources MATCHES "[/\\\\]${module}\\.slang")
            continue()
        endif()
        if(module STREQUAL "utils" AND "LFS_BANDED_EXPORT_RENDER=1" IN_LIST defines)
            set(module utils_banded)
        endif()
        list(APPEND imports "\"${module}.slang-module\"")
    endforeach()
    list(JOIN imports ", " imports)
    list(TRANSFORM defines PREPEND "\"")
    list(TRANSFORM defines APPEND "\"")
    list(JOIN defines ", " defines)
    set(${out_var} "{
      \"module\": {\"name\": \"${host}\", \"file\": \"${host}.slang-module\", \"sha256\": \"${hash}\"},
      \"imports\": [${imports}],
      \"stage\": \"compute\",
      \"entry\": \"main\",
      \"defines\": [${defines}],
      \"bindings\": [${bindings}],
      \"push_constant_bytes\": ${push_constant_bytes},
      \"workgroup\": [${workgroup}],
      \"subgroup_size\": ${host_subgroup_size}
    }" PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

set(abi_input "")
foreach(name IN LISTS api_sources)
    file(SHA256 "${SOURCE_DIR}/${name}" hash)
    string(APPEND abi_input "${name}:${hash}\n")
endforeach()
string(SHA256 abi_hash "${abi_input}")

# slangc prints its build tag to stderr on some platforms.
execute_process(COMMAND "${SLANGC}" -version OUTPUT_VARIABLE build ERROR_VARIABLE build COMMAND_ERROR_IS_FATAL ANY)
string(STRIP "${build}" build)
string(REPLACE "\\" "\\\\" build "${build}")
string(REPLACE "\"" "\\\"" build "${build}")
string(REPLACE "\n" "\\n" build "${build}")

set(api "")
foreach(name IN LISTS api_modules)
    module_entry(${name} ${name} module)
    list(APPEND api "${module}")
endforeach()
list(JOIN api ",\n    " api)

set(modules "")
set(hosts "")
if(TEST_HOST)
    host_entry(${TEST_HOST} "" hosts)
else()
    # What the hosts import beside the API. Banded hosts import utils built
    # for banded export under the same module name.
    module_entry(lfs_reveal lfs_reveal reveal)
    module_entry(spherical_harmonics spherical_harmonics sh)
    module_entry(utils utils utils)
    module_entry(utils utils_banded utils_banded)
    set(modules "${reveal},\n    ${sh},\n    ${utils},\n    ${utils_banded}")
    string(REPLACE "|" ";" host_specs "${HOSTS}")
    foreach(spec IN LISTS host_specs)
        string(REPLACE ":" ";" spec "${spec}")
        list(POP_FRONT spec host)
        string(REPLACE "," ";" defines "${spec}")
        host_entry(${host} "${defines}" entry)
        list(APPEND hosts "${entry}")
    endforeach()
    list(JOIN hosts ",\n    " hosts)
endif()

# Options match the legacy SPIR-V recipes in
# src/rendering/rasterizer/vulkan/CMakeLists.txt (-O -fp-mode fast, slangc's
# SPIR-V 1.5 default and column-major matrices); the runtime applies them.
# Each host lists the interface the renderer provides to it.
string(CONFIGURE [=[{
  "version": 1,
  "abi_hash": "@abi_hash@",
  "compiler_build": "@build@",
  "profile": "spirv_1_5",
  "optimization": 1,
  "fast_math": true,
  "column_major": true,
  "api": [
    @api@
  ],
  "modules": [
    @modules@
  ],
  "hosts": [
    @hosts@
  ]
}
]=] manifest @ONLY)
string(JSON ignored ERROR_VARIABLE json_error TYPE "${manifest}")
if(json_error)
    message(FATAL_ERROR "Invalid shader manifest: ${json_error}")
endif()
file(WRITE "${OUTPUT_DIR}/manifest.json.tmp" "${manifest}")
file(RENAME "${OUTPUT_DIR}/manifest.json.tmp" "${OUTPUT_DIR}/manifest.json")

if(NOT ABI_HEADER)
    return()
endif()
# Rewrite only on change so API-neutral host edits do not recompile C++.
set(header "// Generated by cmake/GenerateUserShaderIR.cmake. Do not edit.\n#pragma once\n\nnamespace lfs::vis::shaders {\n    inline constexpr char kAbiHash[] = \"${abi_hash}\";\n}\n")
set(current "")
if(EXISTS "${ABI_HEADER}")
    file(READ "${ABI_HEADER}" current)
endif()
if(NOT current STREQUAL header)
    file(WRITE "${ABI_HEADER}" "${header}")
endif()
