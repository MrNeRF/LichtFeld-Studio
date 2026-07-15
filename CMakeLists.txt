# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# windows/linux platform independent
set(RAW_VCPKG_PATH $ENV{VCPKG_ROOT})
file(TO_CMAKE_PATH "${RAW_VCPKG_PATH}" VCPKG_ROOT)

if ("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
    message(STATUS "user didn't specify CMAKE_TOOLCHAIN_FILE manually, so let's use default -> ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

cmake_minimum_required(VERSION 3.30)

# Keep test-only packages out of the application dependency graph. Command-line
# cache values are visible before project(), when the vcpkg toolchain resolves
# manifest features.
list(REMOVE_ITEM VCPKG_MANIFEST_FEATURES tests)
if(BUILD_TESTS OR BUILD_UNICODE_TEST_ONLY)
    list(APPEND VCPKG_MANIFEST_FEATURES tests)
    list(REMOVE_DUPLICATES VCPKG_MANIFEST_FEATURES)
endif()

# vcpkg manifest install runs during CMake configure/regeneration, outside the
# native build tool's `cmake --build -j ...` limit. Cap vcpkg separately before
# the toolchain is loaded by project().
set(LFS_VCPKG_MAX_CONCURRENCY "" CACHE STRING "Maximum number of concurrent jobs vcpkg may use while building packages; empty uses an explicit vcpkg setting or caps auto-detection at 6")
set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY "${LFS_VCPKG_MAX_CONCURRENCY}")
if(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY STREQUAL "")
    if(DEFINED ENV{VCPKG_MAX_CONCURRENCY} AND NOT "$ENV{VCPKG_MAX_CONCURRENCY}" STREQUAL "")
        set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY "$ENV{VCPKG_MAX_CONCURRENCY}")
    elseif(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL} AND NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" STREQUAL "")
        set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
        if(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY MATCHES "^[1-9][0-9]*$" AND
           _LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY GREATER 6)
            set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY 6)
        endif()
    else()
        include(ProcessorCount)
        ProcessorCount(_LFS_HOST_PROCESSOR_COUNT)
        if(_LFS_HOST_PROCESSOR_COUNT EQUAL 0)
            set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY 6)
        elseif(_LFS_HOST_PROCESSOR_COUNT GREATER 6)
            set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY 6)
        else()
            set(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY "${_LFS_HOST_PROCESSOR_COUNT}")
        endif()
    endif()
endif()
if(NOT _LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "LFS_VCPKG_MAX_CONCURRENCY must resolve to a positive integer, got '${_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY}'")
endif()
set(ENV{VCPKG_MAX_CONCURRENCY} "${_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY}")
message(STATUS "vcpkg package build concurrency limited to ${_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY}")
unset(_LFS_EFFECTIVE_VCPKG_MAX_CONCURRENCY)
unset(_LFS_HOST_PROCESSOR_COUNT)

option(LFS_ENFORCE_LINUX_GUI_BACKENDS "Fail configure when Linux SDL3 would lack X11 and Wayland support" ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinuxGuiSanity.cmake")
lfs_linux_preflight_gui_backends()

# Suppress CMake developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

project(LichtFeld-Studio VERSION 0.5.3 LANGUAGES CUDA CXX C)

# On Linux, hide symbols pulled in from third-party image / compression static
# archives so they don't escape into the dynamic symbol tables of the binaries
# we ship. Without this, every .so/exe we build re-exports e.g.
# png_destroy_read_struct from its statically-linked libpng16. When the GTK
# file-chooser dialog asks gdk-pixbuf to decode a PNG icon, gdk-pixbuf
# allocates a png_struct against the system libpng but the dynamic linker
# resolves png_destroy_read_struct to one of our re-exported copies; the two
# libpng versions have incompatible png_struct layouts and the icon-decode
# path crashes inside free(). The list is targeted at libraries with known
# clashes against system loaders (libpng + libz for gdk-pixbuf, libjpeg /
# libtiff / libwebp for the same code path); our own static archives
# (lfs_tensor_kernels, lfs_core_cuda, ...) stay exported so the rest of our
# .so graph keeps resolving cross-module symbols.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # ld accepts comma- or colon-separated archive lists for --exclude-libs,
    # but the comma form collides with the way `-Wl,` itself splits on commas
    # (each comma becomes a separate ld argument). Use colons so the whole
    # list reaches ld as a single argument.
    set(_LFS_EXCLUDE_STATIC_ARCHIVES
        "libpng16.a:libpng.a:libz.a:libjpeg.a:libturbojpeg.a:libtiff.a:libwebp.a:libwebpdecoder.a:libsharpyuv.a:liblzma.a:libdeflate.a:libzstd.a"
    )
    string(APPEND CMAKE_EXE_LINKER_FLAGS    " -Wl,--exclude-libs,${_LFS_EXCLUDE_STATIC_ARCHIVES}")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--exclude-libs,${_LFS_EXCLUDE_STATIC_ARCHIVES}")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " -Wl,--exclude-libs,${_LFS_EXCLUDE_STATIC_ARCHIVES}")
    unset(_LFS_EXCLUDE_STATIC_ARCHIVES)
endif()

# The application ships CUDA DLLs and links CUDA::cudart throughout the target graph.
# Keep CUDA language targets on the shared runtime as well; MSVC otherwise adds
# cudart_static.lib implicitly, which collides with cudart.lib when linking DLLs.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "CUDA runtime linkage" FORCE)

# On Windows with a multi-config generator (MSVC/Visual Studio, Ninja Multi-Config),
# route all build outputs to build/<Config>/ so Debug and Release binaries are
# separated. Single-config generators (Ninja, NMake) use a flat binary dir —
# the selected CMAKE_BUILD_TYPE is baked in at configure time, so no subdirectory
# is needed; each target's per-target RUNTIME_OUTPUT_DIRECTORY (set to
# ${CMAKE_BINARY_DIR}) ensures a flat layout under build/.
# Linux: same per-target convention applies; no override needed here.
# The else() branch explicitly sets to "" to override any stale cached value.
get_property(_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _IS_MULTI_CONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "")
endif()

# Disable C++20 module scanning
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(GIT_SRC_DIR "${CMAKE_SOURCE_DIR}")
set(GIT_VERSION_HEADER "${CMAKE_BINARY_DIR}/include/git_version.h")
include("${CMAKE_SOURCE_DIR}/cmake/GenerateGitVersion.cmake")
add_custom_target(lfs_git_version ALL
    BYPRODUCTS "${CMAKE_BINARY_DIR}/include/git_version.h"
    COMMAND ${CMAKE_COMMAND}
            "-DGIT_SRC_DIR=${CMAKE_SOURCE_DIR}"
            "-DGIT_VERSION_HEADER=${CMAKE_BINARY_DIR}/include/git_version.h"
            -P "${CMAKE_SOURCE_DIR}/cmake/GenerateGitVersion.cmake"
    COMMENT "Refreshing git_version.h"
    VERBATIM
)

# App/core ABI tripwire. Hash every lfs_core implementation and interface input,
# because a dirty-worktree rebuild cannot rely on the Git commit alone to
# distinguish a newly linked app from a stale shared library.
file(GLOB_RECURSE LFS_CORE_BUILD_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/*.cu"
    "${CMAKE_SOURCE_DIR}/src/core/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/core/*.hpp"
)
list(APPEND LFS_CORE_BUILD_INPUTS
    "${CMAKE_SOURCE_DIR}/src/core/CMakeLists.txt"
)
list(SORT LFS_CORE_BUILD_INPUTS)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LFS_CORE_BUILD_INPUTS}
)
set(_lfs_core_abi_material
    "${PROJECT_VERSION}|${GIT_COMMIT_HASH_SHORT}|${CMAKE_BUILD_TYPE}|${CMAKE_SYSTEM_NAME}|${CMAKE_SIZEOF_VOID_P}|${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CUDA_COMPILER_ID}|${CMAKE_CUDA_COMPILER_VERSION}|${CMAKE_CUDA_ARCHITECTURES}|${VCPKG_TARGET_TRIPLET}")
foreach(_lfs_core_build_input IN LISTS LFS_CORE_BUILD_INPUTS)
    file(SHA256 "${_lfs_core_build_input}" _lfs_core_build_input_hash)
    string(APPEND _lfs_core_abi_material "|${_lfs_core_build_input_hash}")
endforeach()
string(SHA256 LFS_CORE_ABI_STAMP "${_lfs_core_abi_material}")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/lfs_core_abi_stamp.h.in"
    "${CMAKE_BINARY_DIR}/include/lfs_core_abi_stamp.h"
    @ONLY
)

# Setup
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_UNICODE_TEST_ONLY "Build only Unicode path test (no CUDA)" OFF)
option(LFS_FAST_COMPILE "Optimize measured slow C++ components for build speed instead of runtime speed (WIN32 only)" OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeveloperOptions.cmake")

# Keep immutable third-party archives outside individual build trees. A clean
# configure still verifies each declared checksum and extracts into its own
# tree, but it does not redownload hundreds of MiB already fetched locally.
if(WIN32 AND DEFINED ENV{LOCALAPPDATA})
    file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}/LichtFeldStudio/downloads" _lfs_download_cache_default)
elseif(APPLE AND DEFINED ENV{HOME})
    set(_lfs_download_cache_default "$ENV{HOME}/Library/Caches/LichtFeldStudio/downloads")
elseif(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_lfs_download_cache_default "$ENV{XDG_CACHE_HOME}/lichtfeld/downloads")
elseif(DEFINED ENV{HOME})
    set(_lfs_download_cache_default "$ENV{HOME}/.cache/lichtfeld/downloads")
else()
    set(_lfs_download_cache_default "${CMAKE_BINARY_DIR}/_downloads")
endif()
set(LFS_DOWNLOAD_CACHE_DIR "${_lfs_download_cache_default}" CACHE PATH
    "Shared cache for checksum-pinned third-party download archives")
file(MAKE_DIRECTORY "${LFS_DOWNLOAD_CACHE_DIR}")
unset(_lfs_download_cache_default)

# Add -rdynamic for backtrace symbol resolution
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
endif()

# PTX-only: smaller binary (~66MB vs ~750MB), JIT compiles at runtime
option(BUILD_CUDA_PTX_ONLY "Build PTX-only (JIT at runtime, ~5-15s first startup)" OFF)

# Portable: PTX-only + bundled libs, no tests
option(BUILD_PORTABLE "Build portable distribution" OFF)
if(BUILD_PORTABLE)
    set(BUILD_CUDA_PTX_ONLY ON CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_compile_definitions(LFS_BUILD_PORTABLE=1)
    message(STATUS "Portable: PTX-only, tests disabled")
endif()

# Compiler caching is part of the default developer loop. Disable it for clean
# compiler benchmarks or when diagnosing a launcher-specific problem.
option(ENABLE_COMPILER_CACHE "Auto-detect and use sccache/ccache compiler launchers" ON)
if(ENABLE_COMPILER_CACHE)
    # CUDA caching disabled on Windows due to nvcc compatibility issues.
    find_program(SCCACHE_PROGRAM sccache)
    find_program(CCACHE_PROGRAM ccache)
    if(SCCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
        if(WIN32)
            message(STATUS "Compiler cache: sccache (C/C++ only, CUDA uncached)")
        else()
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "Compiler cache: sccache")
        endif()
    elseif(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        if(WIN32)
            message(STATUS "Compiler cache: ccache (C/C++ only, CUDA uncached)")
        else()
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "Compiler cache: ccache")
        endif()
    else()
        message(STATUS "Compiler cache: requested, but no launcher was found")
    endif()
else()
    message(STATUS "Compiler cache: disabled")
endif()

# Minimum SM for PTX-only (75=Turing, 80=Ampere, 86=GA102, 89=Ada, 90=Hopper)
set(BUILD_CUDA_MIN_SM "75" CACHE STRING "Minimum SM for PTX-only builds")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA setup
# On Windows, nvidia-smi.exe is in System32, but CMake might be running as 32-bit
# which causes WOW64 redirection. Explicitly search in Sysnative to avoid this.
if(WIN32)
    find_program(NVIDIA_SMI nvidia-smi
        PATHS "C:/Windows/Sysnative" "C:/Windows/System32"
        NO_DEFAULT_PATH)
    if(NOT NVIDIA_SMI)
        find_program(NVIDIA_SMI nvidia-smi)
    endif()
else()
    find_program(NVIDIA_SMI nvidia-smi)
endif()
if(NVIDIA_SMI)
    execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE "\n.*" "" DETECTED_COMPUTE_CAP "${DETECTED_COMPUTE_CAP}")

    execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=driver_version --format=csv,noheader
            OUTPUT_VARIABLE NVIDIA_DRIVER_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE "\n.*" "" NVIDIA_DRIVER_VERSION "${NVIDIA_DRIVER_VERSION}")
    message(STATUS "Detected NVIDIA driver: ${NVIDIA_DRIVER_VERSION}")
    if(NVIDIA_DRIVER_VERSION VERSION_LESS "570")
        message(FATAL_ERROR "NVIDIA driver ${NVIDIA_DRIVER_VERSION} is too old. "
            "CUDA Toolkit 12.8 requires driver >= 570. "
            "Please update: https://www.nvidia.com/drivers")
    endif()

    message(STATUS "Detected GPU compute capability: ${DETECTED_COMPUTE_CAP}")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
        string(REPLACE "." "" LichtFeld-Studio_CUDA_ARCH "${DETECTED_COMPUTE_CAP}")
    else()
        set(LichtFeld-Studio_CUDA_ARCH "native")
    endif()
else()
    message(WARNING "nvidia-smi not found! We are likely in CI, default to 8.6.")
    set(DETECTED_COMPUTE_CAP 8.6)
    set(LichtFeld-Studio_CUDA_ARCH "86")
endif()


if(DEFINED TORCH_CUDA_ARCH_LIST)
    unset(TORCH_CUDA_ARCH_LIST CACHE)
endif()

if(BUILD_CUDA_PTX_ONLY)
    # PTX-only: JIT compile at runtime, smaller binary
    if(NOT BUILD_CUDA_MIN_SM MATCHES "^[0-9]+$")
        message(FATAL_ERROR "BUILD_CUDA_MIN_SM must be an integer (70, 75, 86, 90, ...)")
    endif()
    if(BUILD_CUDA_MIN_SM LESS 70)
        message(FATAL_ERROR "BUILD_CUDA_MIN_SM must be >= 70")
    endif()

    set(LichtFeld-Studio_CUDA_ARCH "${BUILD_CUDA_MIN_SM}-virtual")

    math(EXPR _maj "${BUILD_CUDA_MIN_SM}/10")
    math(EXPR _min "${BUILD_CUDA_MIN_SM}%10")
    set(TORCH_CUDA_ARCH_LIST "${_maj}.${_min}+PTX" CACHE STRING "" FORCE)

    message(STATUS "CUDA: PTX-only (SM >= ${BUILD_CUDA_MIN_SM}, JIT at runtime)")
else()
    # Native GPU only
    set(TORCH_CUDA_ARCH_LIST "${DETECTED_COMPUTE_CAP}" CACHE STRING "" FORCE)
    message(STATUS "CUDA: native (${LichtFeld-Studio_CUDA_ARCH})")
endif()

enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 221,1394,1388 -Xcompiler=/wd4267 -Xcompiler=/Zc:preprocessor")
endif()

# cuDNN 9 runtime libraries are loaded indirectly by ONNX Runtime's CUDA
# provider, so CMake's runtime dependency scanner only sees part of the set.
# Discover the CUDA-versioned cuDNN bin directory explicitly and copy every
# cudnn*.dll beside the executable / into portable installs on Windows.
set(LFS_CUDNN_BIN_DIR "" CACHE PATH "Path to the cuDNN runtime DLL directory")
set(CUDNN_ROOT_DIR "" CACHE PATH "Path to the cuDNN installation root")
set(_lfs_cudnn_runtime_libs "")
set(_lfs_cudnn_bin_dir "")
if(WIN32)
    function(_lfs_append_cudnn_bin_candidate OUT_VAR DIR)
        if(DIR)
            file(TO_CMAKE_PATH "${DIR}" _candidate)
            list(APPEND ${OUT_VAR} "${_candidate}")
            set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
        endif()
    endfunction()

    function(_lfs_append_cudnn_root_candidates OUT_VAR ROOT)
        if(ROOT)
            file(TO_CMAKE_PATH "${ROOT}" _root)
            if(CUDAToolkit_VERSION_MAJOR AND CUDAToolkit_VERSION_MINOR)
                _lfs_append_cudnn_bin_candidate(${OUT_VAR} "${_root}/bin/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/x64")
            endif()
            if(CUDAToolkit_VERSION_MAJOR)
                file(GLOB _major_matches LIST_DIRECTORIES true "${_root}/bin/${CUDAToolkit_VERSION_MAJOR}.*/x64")
                list(SORT _major_matches COMPARE NATURAL ORDER DESCENDING)
                foreach(_major_match IN LISTS _major_matches)
                    _lfs_append_cudnn_bin_candidate(${OUT_VAR} "${_major_match}")
                endforeach()
            endif()
            _lfs_append_cudnn_bin_candidate(${OUT_VAR} "${_root}/bin/x64")
            _lfs_append_cudnn_bin_candidate(${OUT_VAR} "${_root}/bin")
            set(${OUT_VAR} "${${OUT_VAR}}" PARENT_SCOPE)
        endif()
    endfunction()

    set(_lfs_cudnn_bin_candidates "")
    _lfs_append_cudnn_bin_candidate(_lfs_cudnn_bin_candidates "${LFS_CUDNN_BIN_DIR}")
    _lfs_append_cudnn_root_candidates(_lfs_cudnn_bin_candidates "${CUDNN_ROOT_DIR}")
    _lfs_append_cudnn_root_candidates(_lfs_cudnn_bin_candidates "$ENV{CUDNN_ROOT_DIR}")
    _lfs_append_cudnn_bin_candidate(_lfs_cudnn_bin_candidates "$ENV{CUDNN_BIN_DIR}")
    _lfs_append_cudnn_root_candidates(_lfs_cudnn_bin_candidates "$ENV{CUDNN_ROOT}")
    _lfs_append_cudnn_root_candidates(_lfs_cudnn_bin_candidates "$ENV{CUDNN_PATH}")

    file(GLOB _lfs_common_cudnn_roots LIST_DIRECTORIES true "C:/Program Files/NVIDIA/CUDNN/v*")
    list(SORT _lfs_common_cudnn_roots COMPARE NATURAL ORDER DESCENDING)
    foreach(_lfs_common_cudnn_root IN LISTS _lfs_common_cudnn_roots)
        _lfs_append_cudnn_root_candidates(_lfs_cudnn_bin_candidates "${_lfs_common_cudnn_root}")
    endforeach()
    list(REMOVE_DUPLICATES _lfs_cudnn_bin_candidates)

    foreach(_lfs_cudnn_candidate IN LISTS _lfs_cudnn_bin_candidates)
        file(GLOB _lfs_cudnn_candidate_dlls CONFIGURE_DEPENDS "${_lfs_cudnn_candidate}/cudnn*.dll")
        if(_lfs_cudnn_candidate_dlls)
            set(_lfs_cudnn_bin_dir "${_lfs_cudnn_candidate}")
            set(_lfs_cudnn_runtime_libs ${_lfs_cudnn_candidate_dlls})
            list(SORT _lfs_cudnn_runtime_libs)
            break()
        endif()
    endforeach()

    if(_lfs_cudnn_runtime_libs)
        list(LENGTH _lfs_cudnn_runtime_libs _lfs_cudnn_runtime_lib_count)
        message(STATUS "cuDNN runtime DLLs: ${_lfs_cudnn_runtime_lib_count} found in ${_lfs_cudnn_bin_dir}")
    elseif(BUILD_PORTABLE)
        message(FATAL_ERROR "cuDNN runtime DLLs were not found. Set CUDNN_ROOT_DIR to the cuDNN root, or LFS_CUDNN_BIN_DIR to the CUDA-versioned cuDNN bin directory.")
    else()
        message(WARNING "cuDNN runtime DLLs were not found. Set CUDNN_ROOT_DIR or LFS_CUDNN_BIN_DIR if CUDA preprocessing/runtime inference needs cuDNN.")
    endif()
endif()

# Create nvToolsExt target BEFORE finding Torch (PyTorch expects this)
if(NOT TARGET CUDA::nvToolsExt)
    find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt
            PATHS
            ${CUDAToolkit_LIBRARY_ROOT}
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_TARGET_DIR}/lib64
            /usr/local/cuda-12.8/lib64
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
    )

    if(NVTOOLSEXT_LIBRARY)
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
        )
        message(STATUS "Created CUDA::nvToolsExt target: ${NVTOOLSEXT_LIBRARY}")
    else()
        # Create empty interface library (nvToolsExt is deprecated in CUDA 12+)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        message(STATUS "nvToolsExt not found (normal for CUDA 12+), created empty interface target")
    endif()
endif()

message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}")

# Find packages
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(SDL3 REQUIRED)
lfs_validate_linux_sdl_video_backends()
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(implot REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupNativeFileDialog.cmake")
find_package(nlohmann_json REQUIRED)
find_package(args REQUIRED)
find_package(spdlog REQUIRED)
find_package(Freetype REQUIRED)
find_package(WebP REQUIRED)
find_package(LibArchive REQUIRED)
find_package(libdeflate CONFIG REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(OpenSSL REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupOnnxRuntime.cmake")
find_package(pxr CONFIG REQUIRED)
find_package(assimp REQUIRED)
find_package(RmlUi CONFIG REQUIRED)

# OpenImageIO's static vcpkg target exposes fmt::fmt-header-only through
# OpenImageIO_Util. That definition is needed by TUs which include OIIO, but it
# otherwise makes every source in a consumer instantiate fmt. Preserve OIIO's
# own public compile interface while making the static archive closure link-only;
# each of our five OIIO source files opts into FMT_HEADER_ONLY explicitly.
add_library(lfs_oiio INTERFACE)
target_link_libraries(lfs_oiio INTERFACE "$<LINK_ONLY:OpenImageIO::OpenImageIO>")
get_target_property(_lfs_oiio_includes OpenImageIO::OpenImageIO INTERFACE_INCLUDE_DIRECTORIES)
if(_lfs_oiio_includes)
    target_include_directories(lfs_oiio SYSTEM INTERFACE ${_lfs_oiio_includes})
endif()
get_target_property(_lfs_oiio_definitions OpenImageIO::OpenImageIO INTERFACE_COMPILE_DEFINITIONS)
if(_lfs_oiio_definitions)
    target_compile_definitions(lfs_oiio INTERFACE ${_lfs_oiio_definitions})
endif()
get_target_property(_lfs_oiio_features OpenImageIO::OpenImageIO INTERFACE_COMPILE_FEATURES)
if(_lfs_oiio_features)
    target_compile_features(lfs_oiio INTERFACE ${_lfs_oiio_features})
endif()
unset(_lfs_oiio_includes)
unset(_lfs_oiio_definitions)
unset(_lfs_oiio_features)

# Vulkan-native depth: glm::perspective/ortho/frustum must emit clip-space z in
# [0, 1], not OpenGL's [-1, 1]. Keeps gl_FragDepth, the grid pass, the depth-
# blit pass and the CPU-side NDC bound checks on a single convention.
target_compile_definitions(glm::glm INTERFACE GLM_FORCE_DEPTH_ZERO_TO_ONE)

# TORCH_CUDA_ARCH_LIST is already defined by Torch
# use it here as the global default CUDA architecture list too
if(LichtFeld-Studio_CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
endif()

# Create configuration header (static options)
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
        @ONLY
)


# Project-wide definitions
add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -O0")
endif()

# External dependencies (EXCLUDE_FROM_ALL to handle install ourselves)
add_subdirectory(external EXCLUDE_FROM_ALL)

# Keep LichtFeld's ordinary MSVC translation units on nvcc's conforming host
# preprocessor and report the selected C++ language version accurately.
# This is intentionally added after external/ so vendored targets keep their
# upstream compiler settings.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    )
endif()

if(LFS_ENABLE_CUDA_DEVICE_DEBUG)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
endif()

add_compile_definitions(
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:RELEASE_BUILD>
)

# =============================================================================
# LIBRARY MODULES
# =============================================================================
add_subdirectory(src/diagnostics) # Shared diagnostics/profiling module
add_subdirectory(src/core)       # Core module (lfs::core namespace)
add_subdirectory(src/training)   # Training module (lfs::training namespace)
add_subdirectory(src/io)         # I/O module (lfs::io namespace)
add_subdirectory(src/rendering)  # Rendering module (lfs::rendering namespace)
add_subdirectory(src/sequencer)  # Sequencer module (lfs::sequencer namespace)
add_subdirectory(src/preprocessing) # CLI preprocessing module (ONNXRuntime)
add_subdirectory(src/visualizer) # Visualizer module (lfs::vis namespace)
add_subdirectory(src/tcp)        # TCP event server (ZMQ responder + publisher)
add_subdirectory(src/geometry)   # Geometry utilities
add_subdirectory(src/mcp)        # MCP server (AI agent integration)
add_subdirectory(src/python)

# =============================================================================
# MAIN EXECUTABLES
# =============================================================================

# -----------------------------------------------------------------------------
# LichtFeld-Studio
# -----------------------------------------------------------------------------

add_executable(${PROJECT_NAME}
    src/app/main.cpp
    src/app/application.cpp
    src/app/mcp_gui_tools.cpp
    src/app/mcp_operator_tools.cpp
    src/app/mcp_runtime_tools.cpp
    src/app/mcp_sequencer_tools.cpp
    src/app/mcp_ui_registry_tools.cpp
    src/app/converter.cpp
    resources/lichtfeld-studio.rc
)

foreach(_lfs_git_version_consumer IN ITEMS
        ${PROJECT_NAME}
        lfs_visualizer
        lfs_py
        lfs_mcp)
    if(TARGET ${_lfs_git_version_consumer})
        add_dependencies(${_lfs_git_version_consumer} lfs_git_version)
    endif()
endforeach()


set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/app/include  # For app headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src              # For training/, visualizer/, etc.
        ${CMAKE_CURRENT_SOURCE_DIR}/src/visualizer   # For internal visualizer headers used by GUI MCP
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/include  # For rendering headers (MCP GUI tools)
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tcp          # For tcp event server
        ${CMAKE_CURRENT_SOURCE_DIR}/external         # For external headers

        ${CUDAToolkit_INCLUDE_DIRS}
)

if(CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(LFS_APP_INTERNAL_LINK_LIBS
        "$<LINK_GROUP:RESCAN,lfs_visualizer,lfs_mcp,lfs_sequencer,lfs_rendering_tensor,lfs_training,lfs_preprocessing,lfs_geometry,lfs_core,lfs_io,lfs_video,fastlfs_backend,lfs_python_utils,lfs_tcp>"
    )
else()
    set(LFS_APP_INTERNAL_LINK_LIBS
        lfs_visualizer
        lfs_tcp
        lfs_mcp
        lfs_sequencer
        lfs_rendering_tensor
        lfs_training
        lfs_preprocessing
        lfs_geometry
        lfs_core
        lfs_io
        lfs_video
        fastlfs_backend
        lfs_python_utils
    )
endif()

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${LFS_APP_INTERNAL_LINK_LIBS}

        CUDA::cudart
        CUDA::curand
        spdlog::spdlog
        taywee::args
        SDL3::SDL3
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
        volk::volk_headers)

if(BUILD_PORTABLE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    get_filename_component(_lfs_vulkan_loader_realpath "${Vulkan_LIBRARY}" REALPATH)
    get_filename_component(_lfs_vulkan_loader_realname "${_lfs_vulkan_loader_realpath}" NAME)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/lib"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_lfs_vulkan_loader_realpath}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/lib/${_lfs_vulkan_loader_realname}"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_lfs_vulkan_loader_realname}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/lib/libvulkan.so.1"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "libvulkan.so.1"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/lib/libvulkan.so"
            COMMENT "Copying Vulkan loader to portable runtime directory")
endif()

# The app embeds Python and loads the nanobind module at runtime for panels,
# menus, and operators. Keep the runtime artifact in lockstep with the app.
add_dependencies(${PROJECT_NAME} lfs_py)

# Copy nvImageCodec shared libraries next to executable for easy deployment
if(TARGET nvimgcodec)
    # Force build of nvimgcodec even with EXCLUDE_FROM_ALL
    add_dependencies(${PROJECT_NAME} nvimgcodec)

    # Add post-build command to copy libraries and create symlinks (Unix only)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvimgcodec>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying nvImageCodec library"
    )

    # Create symlinks only on Unix-like systems (not Windows)
    if(UNIX)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "$<TARGET_FILE_NAME:nvimgcodec>"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/libnvimgcodec.so.0"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "libnvimgcodec.so.0"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/libnvimgcodec.so"
            COMMENT "Creating nvImageCodec symlinks"
        )
    endif()

    message(STATUS "Post-build: nvImageCodec libraries will be copied to executable directory")
endif()

# ONNX Runtime dlopens its CUDA execution providers at runtime. Keep the core
# runtime and CUDA/shared provider libraries beside the exe so local and packaged
# preprocess runs do not silently fall back to CPU.
set(_lfs_ort_runtime_libs ${LFS_ONNXRUNTIME_RUNTIME_LIBS})
if(NOT _lfs_ort_runtime_libs AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    file(GLOB _lfs_ort_runtime_libs
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/*onnxruntime*"
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/*onnxruntime*")
endif()
list(FILTER _lfs_ort_runtime_libs INCLUDE REGEX "\\.(dll|dylib|so(\\.[0-9]+)*)$")
list(REMOVE_DUPLICATES _lfs_ort_runtime_libs)
if(_lfs_ort_runtime_libs)
    foreach(_lfs_ort_runtime_lib IN LISTS _lfs_ort_runtime_libs)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_lfs_ort_runtime_lib}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying ONNX Runtime runtime library"
        )
    endforeach()
else()
    message(WARNING "ONNX Runtime runtime libraries were not found; preprocess CUDA inference may fall back to CPU or fail at startup")
endif()

# Copy Python DLLs for Windows runtime (Python is embedded in the application)
if(WIN32)
    get_filename_component(VCPKG_PYTHON_DIR "${Python_EXECUTABLE}" DIRECTORY)

    # Glob python*.dll - check both tools/python3/ and parent bin/ (vcpkg layout varies)
    file(GLOB PYTHON_RUNTIME_DLLS
        "${VCPKG_PYTHON_DIR}/python*.dll"
        "${VCPKG_PYTHON_DIR}/../bin/python*.dll"
    )
    if(PYTHON_RUNTIME_DLLS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PYTHON_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying Python DLLs to executable directory"
        )
        message(STATUS "Post-build: Python DLLs will be copied: ${PYTHON_RUNTIME_DLLS}")
    endif()
endif()

# Copy CUDA runtime DLLs for Windows (needed for Python module loading)
if(WIN32 AND CUDAToolkit_FOUND)
    file(GLOB CUDA_DLLS_1 "${CUDAToolkit_BIN_DIR}/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/curand64*.dll")
    file(GLOB CUDA_DLLS_2 "${CUDAToolkit_BIN_DIR}/x64/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/x64/curand64*.dll")
    set(CUDA_RUNTIME_DLLS ${CUDA_DLLS_1} ${CUDA_DLLS_2})
    foreach(CUDA_DLL ${CUDA_RUNTIME_DLLS})
        if(EXISTS "${CUDA_DLL}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CUDA_DLL}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying ${CUDA_DLL} to executable directory"
            )
        endif()
    endforeach()
    message(STATUS "Post-build: CUDA DLLs will be copied from ${CUDAToolkit_BIN_DIR}")
endif()

if(WIN32 AND _lfs_cudnn_runtime_libs)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_lfs_cudnn_runtime_libs}
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying cuDNN DLLs to executable directory"
    )
    message(STATUS "Post-build: cuDNN DLLs will be copied from ${_lfs_cudnn_bin_dir}")
endif()

# =============================================================================
# VCPKG DLL COPYING - Ensure all vcpkg DLLs are available at runtime
# =============================================================================
# vcpkg's VCPKG_APPLOCAL_DEPS feature may not copy all needed DLLs.
# This explicitly copies vcpkg DLLs to ensure they're available at runtime.
# 
# IMPORTANT: For Release builds, copy ONLY release DLLs to avoid loading debug CRT.
# For Debug builds, copy both because some vcpkg packages (e.g., zlib) are release-only.
if(WIN32 AND DEFINED VCPKG_TARGET_TRIPLET)
    set(_vcpkg_debug_bin_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/bin")
    set(_vcpkg_release_bin_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")
    file(GLOB _vcpkg_release_dlls CONFIGURE_DEPENDS "${_vcpkg_release_bin_dir}/*.dll")
    file(GLOB _vcpkg_debug_dlls CONFIGURE_DEPENDS "${_vcpkg_debug_bin_dir}/*.dll")
    
    # Always copy release DLLs - needed for both Release and Debug builds
    if(_vcpkg_release_dlls)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_vcpkg_release_dlls}
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying vcpkg release DLLs"
        )
    endif()
    
    # Only copy debug DLLs for Debug builds (they link to debug CRT which breaks Release)
    # For single-config generators (Ninja, Makefiles): check CMAKE_BUILD_TYPE
    # For multi-config generators (VS, Xcode): use generator expression
    get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_is_multi_config)
        # Multi-config: use generator expression to conditionally copy
        if(_vcpkg_debug_dlls)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
                    ${_vcpkg_debug_dlls}
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying vcpkg debug DLLs (Debug config only)"
            )
        endif()
        message(STATUS "Post-build: vcpkg DLLs - release always, debug for Debug config only (multi-config)")
    else()
        # Single-config: check CMAKE_BUILD_TYPE at configure time
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND _vcpkg_debug_dlls)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_vcpkg_debug_dlls}
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying vcpkg debug DLLs"
            )
            message(STATUS "Post-build: vcpkg DLLs - copying both release and debug (Debug build)")
        else()
            message(STATUS "Post-build: vcpkg DLLs - copying release only (${CMAKE_BUILD_TYPE} build)")
        endif()
    endif()
endif()

# Before extension post-build copying: remove stale 'extensions' file if present (prevent mkdir failure)
set(_target_extensions_dir "${CMAKE_BINARY_DIR}/extensions")
if(EXISTS "${_target_extensions_dir}" AND NOT IS_DIRECTORY "${_target_extensions_dir}")
    file(REMOVE "${_target_extensions_dir}")
    message(STATUS "Removed stale file '${_target_extensions_dir}' to make way for extensions/ directory")
endif()

# =============================================================================
# EXTENSIONS COPYING - Copy nvImageCodec extensions to build/extensions
# =============================================================================
if(TARGET nvjpeg_ext)
    # Force build of nvjpeg_ext even with EXCLUDE_FROM_ALL
    add_dependencies(${PROJECT_NAME} nvjpeg_ext)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvjpeg_ext>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/extensions/"
        COMMENT "Copying nvjpeg extension binary only"
    )
endif()

if(TARGET nvjpeg2k_ext)
    # Force build of nvjpeg2k_ext even with EXCLUDE_FROM_ALL
    add_dependencies(${PROJECT_NAME} nvjpeg2k_ext)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvjpeg2k_ext>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/extensions/"
        COMMENT "Copying nvjpeg2k extension binary only"
    )

    if(WITH_DYNAMIC_NVJPEG2K)
        if(CUDAToolkit_VERSION_MAJOR)
            set(_lfs_nvjpeg2k_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
        else()
            set(_lfs_nvjpeg2k_cuda_major "12")
        endif()
        set(_lfs_nvjpeg2k_runtime_dir
            "${CMAKE_BINARY_DIR}/_deps/nvjpeg2k_headers-src/lib/${_lfs_nvjpeg2k_cuda_major}")
        if(WIN32)
            set(_lfs_nvjpeg2k_runtime_name "nvjpeg2k_0.dll")
            set(_lfs_nvjpeg2k_runtime_file
                "${_lfs_nvjpeg2k_runtime_dir}/${_lfs_nvjpeg2k_runtime_name}")
            if(EXISTS "${_lfs_nvjpeg2k_runtime_file}")
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_lfs_nvjpeg2k_runtime_file}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${_lfs_nvjpeg2k_runtime_name}"
                    COMMENT "Copying nvjpeg2k runtime library"
                )
            else()
                message(WARNING
                    "WITH_DYNAMIC_NVJPEG2K is ON but nvjpeg2k_0.dll was not found in "
                    "${_lfs_nvjpeg2k_runtime_dir}")
            endif()
        elseif(UNIX)
            file(GLOB _lfs_nvjpeg2k_runtime_candidates
                "${_lfs_nvjpeg2k_runtime_dir}/libnvjpeg2k.so.*.*")
            list(SORT _lfs_nvjpeg2k_runtime_candidates)
            if(_lfs_nvjpeg2k_runtime_candidates)
                list(GET _lfs_nvjpeg2k_runtime_candidates -1 _lfs_nvjpeg2k_runtime_file)
                get_filename_component(_lfs_nvjpeg2k_runtime_name
                    "${_lfs_nvjpeg2k_runtime_file}" NAME)
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_lfs_nvjpeg2k_runtime_file}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${_lfs_nvjpeg2k_runtime_name}"
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "${_lfs_nvjpeg2k_runtime_name}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/libnvjpeg2k.so.0"
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "libnvjpeg2k.so.0"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/libnvjpeg2k.so"
                    COMMENT "Copying nvjpeg2k runtime library"
                )
            else()
                message(WARNING
                    "WITH_DYNAMIC_NVJPEG2K is ON but no libnvjpeg2k.so runtime was found in "
                    "${_lfs_nvjpeg2k_runtime_dir}")
            endif()
        endif()
    endif()
endif()

# =============================================================================
# RESOURCE COPYING - Copy all resources to build/resources for runtime access
# =============================================================================
set(RESOURCE_DIR "$<TARGET_FILE_DIR:${PROJECT_NAME}>/resources")

# Create resource directory structure
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/assets/icon"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/assets/icon/scene"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/assets/fonts"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/assets/themes"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/assets/environments"
    COMMENT "Creating resource directory structure"
)

if(TARGET lfs_vulkan_rasterizer)
    get_target_property(_LFS_VULKAN_RASTERIZER_SPV_DIR
        lfs_vulkan_rasterizer
        LFS_VULKAN_RASTERIZER_SPV_DIR)
    if(_LFS_VULKAN_RASTERIZER_SPV_DIR)
        # The runtime loads SPIR-V from the resource dir first, so staging must
        # follow every shader rebuild. A POST_BUILD on the executable misses
        # shader-only edits (no relink → stale staged shaders); an always-run
        # staging target costs one timestamp compare per build.
        add_custom_target(lfs_stage_vulkan_shaders ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${RESOURCE_DIR}/shaders/vulkan_rasterizer"
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                "${_LFS_VULKAN_RASTERIZER_SPV_DIR}"
                "${RESOURCE_DIR}/shaders/vulkan_rasterizer"
            COMMENT "Staging Vulkan rasterizer shaders"
        )
        add_dependencies(lfs_stage_vulkan_shaders lfs_vulkan_rasterizer_shaders)
        add_dependencies(${PROJECT_NAME} lfs_stage_vulkan_shaders)
    endif()
endif()

# Copy visualizer icons
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/icon"
        "${RESOURCE_DIR}/assets/icon"
    COMMENT "Copying visualizer icons"
)

# Copy fonts (from visualizer)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/fonts"
        "${RESOURCE_DIR}/assets/fonts"
    COMMENT "Copying visualizer fonts"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/rendering/resources/assets/JetBrainsMono-Regular.ttf"
        "${RESOURCE_DIR}/assets/fonts/JetBrainsMono-Regular.ttf"
    COMMENT "Copying rendering monospace font"
)

# Copy themes
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/themes"
        "${RESOURCE_DIR}/assets/themes"
    COMMENT "Copying themes"
)

# Copy bundled environment maps
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/environments"
        "${RESOURCE_DIR}/assets/environments"
    COMMENT "Copying environment maps"
)

# Copy additional assets (logo, splash images, etc.)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-icon.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-logo.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-logo-dark.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-loading.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-loading-dark.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/core11-logo.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/core11-logo-dark.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/volinga-logo.png"
        "${RESOURCE_DIR}/assets/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/volinga-logo-dark.png"
        "${RESOURCE_DIR}/assets/"
    COMMENT "Copying application assets"
)

# Copy locale files (named target so it can be run without rebuilding the exe)
add_custom_target(copy_locale_files
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCE_DIR}/locales"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/src/visualizer/gui/resources/locales"
        "${RESOURCE_DIR}/locales"
    COMMENT "Syncing locale files"
)
add_dependencies(${PROJECT_NAME} copy_locale_files)

# OpenUSD runtime resources (schema/file-format plugin metadata).
# On Windows: copy to <exe>/usd/ so that relative LibraryPaths resolve to the
# DLL copies already loaded from the exe directory.
# On Linux: copy to <exe>/../lib/usd/ (conventional layout).
set(_USD_PLUGIN_SOURCE_DIR "")
foreach(_candidate
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/usd"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin/usd"
    "${CMAKE_BINARY_DIR}/src/python/usd")
    # Require an actual plugin sub-directory (not just the root index).
    if(NOT _USD_PLUGIN_SOURCE_DIR AND EXISTS "${_candidate}/ar/resources/plugInfo.json")
        set(_USD_PLUGIN_SOURCE_DIR "${_candidate}")
    endif()
endforeach()

set(_USD_ROOT_INDEX "")
foreach(_candidate
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/usd/plugInfo.json"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin/usd/plugInfo.json"
    "${CMAKE_BINARY_DIR}/src/python/usd/plugInfo.json")
    if(NOT _USD_ROOT_INDEX AND EXISTS "${_candidate}")
        set(_USD_ROOT_INDEX "${_candidate}")
    endif()
endforeach()

if(WIN32)
    set(_USD_PLUGIN_DEST "$<TARGET_FILE_DIR:${PROJECT_NAME}>/usd")
else()
    set(_USD_PLUGIN_DEST "$<TARGET_FILE_DIR:${PROJECT_NAME}>/../lib/usd")
endif()

if(EXISTS "${_USD_PLUGIN_SOURCE_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_USD_PLUGIN_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${_USD_PLUGIN_SOURCE_DIR}"
            "${_USD_PLUGIN_DEST}"
        COMMENT "Copying OpenUSD plugin resources (per-module)"
    )
    message(STATUS "Post-build: OpenUSD plugin resources will be copied from ${_USD_PLUGIN_SOURCE_DIR}")
endif()

if(EXISTS "${_USD_ROOT_INDEX}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_USD_ROOT_INDEX}"
            "${_USD_PLUGIN_DEST}/plugInfo.json"
        COMMENT "Copying OpenUSD root plugin index"
    )
endif()

message(STATUS "Post-build: Resources will be copied to \${BUILD_DIR}/resources/")

# Per-config output directory variants for the exe: multi-config only.
# Single-config Ninja on Windows uses the flat binary dir (build/).
get_property(_exe_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _exe_is_multi_config)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG        "${CMAKE_BINARY_DIR}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE      "${CMAKE_BINARY_DIR}/Release"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL   "${CMAKE_BINARY_DIR}/MinSizeRel"
    )
endif()

# Platform-specific settings
if(UNIX)
    if(BUILD_PORTABLE)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH_USE_LINK_PATH FALSE
        )
    else()
        set(_lichtfeld_runtime_rpath_common
                "$ORIGIN"
                "$ORIGIN/lib"
                "${CUDAToolkit_LIBRARY_DIR}"
        )
        if(TARGET OpenMeshCore)
            list(APPEND _lichtfeld_runtime_rpath_common "$<TARGET_FILE_DIR:OpenMeshCore>")
        endif()
        set(_lichtfeld_runtime_rpath_release "${_lichtfeld_runtime_rpath_common}")
        set(_lichtfeld_runtime_rpath_debug "${_lichtfeld_runtime_rpath_common}")
        if(DEFINED VCPKG_TARGET_TRIPLET)
            set(_vcpkg_release_runtime_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib")
            if(EXISTS "${_vcpkg_release_runtime_dir}")
                list(APPEND _lichtfeld_runtime_rpath_release "${_vcpkg_release_runtime_dir}")
                list(APPEND _lichtfeld_runtime_rpath_debug "${_vcpkg_release_runtime_dir}")
            endif()

            set(_vcpkg_debug_runtime_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib")
            if(EXISTS "${_vcpkg_debug_runtime_dir}")
                list(APPEND _lichtfeld_runtime_rpath_debug "${_vcpkg_debug_runtime_dir}")
            endif()
        endif()

        list(REMOVE_DUPLICATES _lichtfeld_runtime_rpath_release)
        list(REMOVE_DUPLICATES _lichtfeld_runtime_rpath_debug)
        list(JOIN _lichtfeld_runtime_rpath_release ":" _lichtfeld_runtime_rpath_release_string)
        list(JOIN _lichtfeld_runtime_rpath_debug ":" _lichtfeld_runtime_rpath_debug_string)

        if(CMAKE_CONFIGURATION_TYPES)
            set(_lichtfeld_runtime_rpath_string
                    "$<$<CONFIG:Debug>:${_lichtfeld_runtime_rpath_debug_string}>$<$<NOT:$<CONFIG:Debug>>:${_lichtfeld_runtime_rpath_release_string}>")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(_lichtfeld_runtime_rpath_string "${_lichtfeld_runtime_rpath_debug_string}")
        else()
            set(_lichtfeld_runtime_rpath_string "${_lichtfeld_runtime_rpath_release_string}")
        endif()

        set_target_properties(${PROJECT_NAME} PROPERTIES
                INSTALL_RPATH "${_lichtfeld_runtime_rpath_string}"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

elseif(WIN32)
    # On Windows, DLLs in the same directory as the .exe are automatically found.
    # Multi-config generators use $<CONFIG> subdir; single-config (Ninja) uses
    # the flat binary dir (build/) set via the global CMAKE_RUNTIME_OUTPUT_DIRECTORY.
    if(_exe_is_multi_config)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY              "${CMAKE_BINARY_DIR}/$<CONFIG>"
        )
    endif()
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /bigobj)
    endif()
endif()

if(UNIX)
    add_custom_target(validate-gui
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_vulkan_validation.sh"
                --binary "$<TARGET_FILE:${PROJECT_NAME}>"
        DEPENDS ${PROJECT_NAME}
        USES_TERMINAL
        COMMENT "Launching LichtFeld Studio with source-built Vulkan validation"
        VERBATIM
    )
endif()

if(LFS_FAST_COMPILE AND MSVC)
    function(lfs_enable_fast_cxx_compile target)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Od>
                $<$<COMPILE_LANGUAGE:CXX>:/Ob0>
            )
        endif()
    endfunction()

    # curated list of components to optimize for fast compilation (WIN32 only)
    # CUDA-heavy regressions stay on their target defaults.
    foreach(_lfs_fast_target IN ITEMS
        lfs_visualizer
        lfs_py
        lfs_core
        lfs_training
        lfs_io
        lfs_mcp
        Zep
        OpenMeshCore
    )
        lfs_enable_fast_cxx_compile(${_lfs_fast_target})
    endforeach()
endif()

# Set the default startup project in Visual Studio to LichtFeld-Studio
# (without this, VS defaults to ALL_BUILD when opening the solution)
if(MSVC)
    set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()

# =============================================================================
# TESTING (Optional)
# =============================================================================
if(BUILD_TESTS OR BUILD_UNICODE_TEST_ONLY)
    enable_testing()
    add_subdirectory(tests)
    if(BUILD_UNICODE_TEST_ONLY)
        message(STATUS "Unicode path test enabled. Build with 'ninja unicode_path_test' and run with 'ctest -R UnicodePathTest'")
    else()
        message(STATUS "Tests enabled. Build with 'ninja lichtfeld_tests' and run with 'ninja run_tests' or 'ctest -L fast'")
    endif()
endif()

# =============================================================================
# BUILD INFO & OPTIMIZATIONS
# =============================================================================
message(STATUS "===========================================")
message(STATUS "Build Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Vulkan Found: ${Vulkan_FOUND}")
message(STATUS "  Vulkan Include Dirs: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "  FreeType Found: ${FREETYPE_FOUND}")
message(STATUS "  WebP Found: ${WebP_FOUND}")
message(STATUS "  LibArchive Found: ${LibArchive_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  OpenImageIO Found: ${OpenImageIO_FOUND}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Unicode Test Only: ${BUILD_UNICODE_TEST_ONLY}")
message(STATUS "  Portable Build: ${BUILD_PORTABLE}")
if(BUILD_PORTABLE)
    message(STATUS "  -> PTX-only: ON (SM >= ${BUILD_CUDA_MIN_SM}, JIT at runtime)")
    message(STATUS "  -> Use: cmake --install build --prefix ./dist")
endif()
message(STATUS "===========================================")

# =============================================================================
# INSTALLATION (cmake --install build --prefix ./dist)
# =============================================================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
endif()

set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)

include(GNUInstallDirs)
include(InstallRequiredSystemLibraries)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS lfs_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS lfs_diagnostics
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS lfs_logger
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS lfs_event_bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS lfs_visualizer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS lfs_mcp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(_lfs_ort_runtime_libs)
    if(WIN32)
        set(_lfs_ort_install_dest "${CMAKE_INSTALL_BINDIR}")
    else()
        set(_lfs_ort_install_dest "${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(FILES ${_lfs_ort_runtime_libs} DESTINATION ${_lfs_ort_install_dest})
endif()
if(WIN32 AND BUILD_PORTABLE AND _lfs_cudnn_runtime_libs)
    install(FILES ${_lfs_cudnn_runtime_libs} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION ".")

set(INSTALL_RESOURCE_DIR "${CMAKE_INSTALL_DATADIR}/LichtFeld-Studio")
if(_LFS_VULKAN_RASTERIZER_SPV_DIR)
    install(DIRECTORY "${_LFS_VULKAN_RASTERIZER_SPV_DIR}/"
            DESTINATION "${INSTALL_RESOURCE_DIR}/shaders/vulkan_rasterizer"
            FILES_MATCHING
            PATTERN "*.spv")
endif()
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/icon" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/fonts" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/rendering/resources/assets/JetBrainsMono-Regular.ttf" DESTINATION "${INSTALL_RESOURCE_DIR}/assets/fonts")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/themes" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/environments" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-icon.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-logo.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-logo-dark.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-loading.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/lichtfeld-splash-loading-dark.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/core11-logo.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/core11-logo-dark.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/volinga-logo.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(FILES "${CMAKE_SOURCE_DIR}/src/visualizer/gui/assets/volinga-logo-dark.png" DESTINATION "${INSTALL_RESOURCE_DIR}/assets")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/rmlui/resources/" DESTINATION "${INSTALL_RESOURCE_DIR}/assets/rmlui")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/visualizer/gui/resources/locales" DESTINATION "${INSTALL_RESOURCE_DIR}")

# OpenUSD plugin resources (schema definitions + plugInfo.json files)
# On Windows configure_usd_plugins() looks for <exe_dir>/usd → install next to the exe (bin/usd).
# On Linux it looks for <exe_dir>/../lib/usd → install in lib/usd (unchanged).
if(WIN32)
    set(_USD_INSTALL_DEST "${CMAKE_INSTALL_BINDIR}/usd")
else()
    set(_USD_INSTALL_DEST "${CMAKE_INSTALL_LIBDIR}/usd")
endif()
if(EXISTS "${_USD_PLUGIN_SOURCE_DIR}")
    install(DIRECTORY "${_USD_PLUGIN_SOURCE_DIR}/" DESTINATION "${_USD_INSTALL_DEST}"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE)
endif()
if(EXISTS "${_USD_ROOT_INDEX}")
    install(FILES "${_USD_ROOT_INDEX}" DESTINATION "${_USD_INSTALL_DEST}")
endif()

# Plugin development documentation
set(DOC_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/doc/LichtFeld-Studio")
install(FILES
    "${CMAKE_SOURCE_DIR}/docs/PYTHON_API.md"
    "${CMAKE_SOURCE_DIR}/docs/plugin-system.md"
    "${CMAKE_SOURCE_DIR}/docs/plugin-use-cases.md"
    "${CMAKE_SOURCE_DIR}/docs/plugin-dev-workflow.md"
    DESTINATION "${DOC_INSTALL_DIR}"
    OPTIONAL
)

if(TARGET nvimgcodec)
    install(TARGETS nvimgcodec
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    if(TARGET nvjpeg_ext)
        install(TARGETS nvjpeg_ext
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/extensions"
            RUNTIME DESTINATION "extensions"
        )
    endif()
    if(TARGET nvjpeg2k_ext)
        install(TARGETS nvjpeg2k_ext
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/extensions"
            RUNTIME DESTINATION "extensions"
        )
        if(WITH_DYNAMIC_NVJPEG2K AND DEFINED _lfs_nvjpeg2k_runtime_file AND EXISTS "${_lfs_nvjpeg2k_runtime_file}")
            if(WIN32)
                install(FILES "${_lfs_nvjpeg2k_runtime_file}" DESTINATION ${CMAKE_INSTALL_BINDIR})
            elseif(UNIX)
                install(FILES "${_lfs_nvjpeg2k_runtime_file}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
                install(CODE "
                    execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                        \"${_lfs_nvjpeg2k_runtime_name}\"
                        \"libnvjpeg2k.so.0\"
                        WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
                    execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                        \"libnvjpeg2k.so.0\"
                        \"libnvjpeg2k.so\"
                        WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
                ")
            endif()
        endif()
    endif()
endif()

# Portable build: bundle runtime dependencies via GET_RUNTIME_DEPENDENCIES
if(BUILD_PORTABLE)
    # Exclusion patterns for system libraries
    # nvcuda.dll is provided by the NVIDIA GPU driver, not the CUDA Toolkit
    set(_PRE_EXCLUDE [[api-ms-.*]] [[ext-ms-.*]] [[API-MS-.*]] [[EXT-MS-.*]] [[nvcuda[.]dll]] [[libcuda[.]so.*]])
    set(_POST_EXCLUDE
            [[.*[Ss]ystem32.*]] [[.*[Ww]indows.*]]
            [[/lib/x86_64-linux-gnu/.*]] [[/lib64/ld-linux.*]] [[/usr/lib.*]]
            [[.*ld-linux.*]] [[.*libc[.]so[.].*]] [[.*libpthread.*]]
            [[.*libdl[.]so.*]] [[.*libm[.]so[.].*]] [[.*librt[.]so.*]]
            [[.*libstdc[+][+].*]] [[.*libgcc_s.*]]
            [[.*libnvimgcodec.*]] [[.*libnvjpeg_ext.*]])

    # Search directories for runtime dependencies (quoted for paths with spaces on Windows)
    set(_RUNTIME_SEARCH_DIRS
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/Release"
            "${CMAKE_BINARY_DIR}/lib"
            "${CMAKE_BINARY_DIR}/external/nvImageCodec"
            "${CUDAToolkit_BIN_DIR}"
            "${CUDAToolkit_LIBRARY_DIR}")
    if(WIN32 AND _lfs_cudnn_bin_dir)
        list(APPEND _RUNTIME_SEARCH_DIRS "${_lfs_cudnn_bin_dir}")
    endif()
    if(DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _RUNTIME_SEARCH_DIRS
                "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin"
                "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib"
                "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/bin"
                "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib")
    endif()

    set(_SEARCH_DIRS_STR "")
    foreach(_runtime_search_dir IN LISTS _RUNTIME_SEARCH_DIRS)
        string(APPEND _SEARCH_DIRS_STR " \"${_runtime_search_dir}\"")
    endforeach()

    # Helper: bundle runtime dependencies for a target.
    # On Windows DLLs must live beside the exe (bin/); on Linux they go in lib/.
    if(WIN32)
        set(_BUNDLE_DEST "${CMAKE_INSTALL_BINDIR}")
    else()
        set(_BUNDLE_DEST "${CMAKE_INSTALL_LIBDIR}")
    endif()
    function(bundle_runtime_deps TARGET_FILE IS_LIBRARY)
        if(IS_LIBRARY)
            set(_type "LIBRARIES")
        else()
            set(_type "EXECUTABLES")
        endif()
        install(CODE "
            file(GET_RUNTIME_DEPENDENCIES
                ${_type} \"${TARGET_FILE}\"
                RESOLVED_DEPENDENCIES_VAR _deps
                DIRECTORIES ${_SEARCH_DIRS_STR}
                PRE_EXCLUDE_REGEXES ${_PRE_EXCLUDE}
                POST_EXCLUDE_REGEXES ${_POST_EXCLUDE})
            foreach(_d IN LISTS _deps)
                file(INSTALL \"\${_d}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_BUNDLE_DEST}\" FOLLOW_SYMLINK_CHAIN)
            endforeach()
        " COMPONENT runtime)
    endfunction()

    bundle_runtime_deps("$<TARGET_FILE:${PROJECT_NAME}>" FALSE)
    if(TARGET nvjpeg_ext)
        bundle_runtime_deps("$<TARGET_FILE:nvjpeg_ext>" TRUE)
    endif()
    foreach(_lfs_ort_runtime_lib IN LISTS _lfs_ort_runtime_libs)
        bundle_runtime_deps("${_lfs_ort_runtime_lib}" TRUE)
    endforeach()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        configure_file(
                "${CMAKE_SOURCE_DIR}/cmake/templates/run_lichtfeld.sh.in"
                "${CMAKE_BINARY_DIR}/run_lichtfeld.sh"
                @ONLY)
        install(PROGRAMS "${CMAKE_BINARY_DIR}/run_lichtfeld.sh" DESTINATION ${CMAKE_INSTALL_BINDIR})
    elseif(WIN32)
        install(CODE "
            # Copy extensions/ from build to dist/extensions/
            # Handle both single-config (Ninja) and multi-config (VS) generators
            foreach(_cfg IN ITEMS \"\" \"Release\" \"Debug\" \"RelWithDebInfo\" \"MinSizeRel\")
                if(_cfg)
                    set(_ext_dir \"${CMAKE_BINARY_DIR}/\${_cfg}/extensions\")
                else()
                    set(_ext_dir \"${CMAKE_BINARY_DIR}/extensions\")
                endif()
                if(EXISTS \"\${_ext_dir}\")
                    # Install only DLLs from the build extensions directory to avoid non-binary files
                    file(GLOB _ext_dlls \"\${_ext_dir}/*.dll\")
                    if(_ext_dlls)
                        foreach(_e IN LISTS _ext_dlls)
                            file(INSTALL \"\${_e}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/extensions\")
                        endforeach()
                    endif()
                    break()
                endif()
            endforeach()
            # Copy vcpkg DLLs
            if(EXISTS \"${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin\")
                file(GLOB _vcpkg \"${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin/*.dll\")
                foreach(_v IN LISTS _vcpkg)
                    file(INSTALL \"\${_v}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
                endforeach()
            endif()
        " COMPONENT runtime)
    endif()
endif()

# Build statistics (parses .ninja_log, no Python required)
if(CMAKE_GENERATOR MATCHES "Ninja")
    # Manual target: ninja build_stats
    add_custom_target(build_stats
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/BuildStats.cmake
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Build time statistics by module"
    )

    # Auto-show after build: cmake -DSHOW_BUILD_STATS=ON ..
    option(SHOW_BUILD_STATS "Show build stats after every build" OFF)
    if(SHOW_BUILD_STATS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/BuildStats.cmake
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

# Clean build target: ninja clean_all (clears build + compiler cache)
if(CCACHE_PROGRAM)
    add_custom_target(clean_all
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
        COMMAND ${CCACHE_PROGRAM} --clear
        COMMENT "Cleaning build and compiler cache (ccache)"
    )
elseif(SCCACHE_PROGRAM)
    add_custom_target(clean_all
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
        COMMAND ${SCCACHE_PROGRAM} --zero-stats
        COMMENT "Cleaning build and resetting sccache stats"
    )
endif()
