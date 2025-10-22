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
project(LichtFeld-Studio LANGUAGES CUDA CXX C)

# Get git commit hash (short and full)
execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

execute_process(
        COMMAND git describe --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAGGED_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

# Fallback if not in a git repo
if(NOT GIT_COMMIT_HASH_SHORT)
    set(GIT_COMMIT_HASH_SHORT "unknown")
endif()

if(NOT GIT_TAGGED_VERSION)
    set(GIT_TAGGED_VERSION "unknown")
endif()

message(STATUS "Git commit (short): ${GIT_COMMIT_HASH_SHORT}")
message(STATUS "Git commit (full): ${GIT_TAGGED_VERSION}")

# Setup
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

option(ENABLE_CUDA_GL_INTEROP "Enable CUDA-OpenGL interoperability" ON)
option(BUILD_TESTS "Build tests" OFF)

# Build fat binaries for all modern SMs (>= minimum). When OFF (default), build for the native GPU only.
option(BUILD_CUDA_ALL_SM "Build CUDA fat binaries targeting all modern SMs (>= minimum SM)" OFF)

# Minimum SM (compute capability) to include when BUILD_CUDA_ALL_SM=ON.
# Examples: 75, 80, 86, 89, 90, 100...
# Only has effect if BUILD_CUDA_ALL_SM is ON.
set(BUILD_CUDA_ALL_MIN_SM "86" CACHE STRING "Minimum SM (compute capability) used when BUILD_CUDA_ALL_SM=ON Examples: 75, 80, 86, 89...")

# Set Torch_DIR based on platform and build type
if (WIN32)
    if(CMAKE_CONFIGURATION_TYPES AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
        # https://github.com/pytorch/pytorch/issues/155667
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
        message(STATUS "[${PROJECT_NAME}] User didn't specify -DCMAKE_BUILD_TYPE so use this path for searching libtorch -> ${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")

        if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/")
            message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/release/ on your disk")
        endif()

        if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/libtorch/")
            message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/release/libtorch)!")
        endif()
    else()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
            message(STATUS "[${PROJECT_NAME}] User specified -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} so let's search libtorch by this path -> ${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/")
                message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/release/ on your disk")
            endif()

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/release/libtorch")
                message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/release/libtorch)!")
            endif()
        elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
            message(STATUS "[${PROJECT_NAME}] User specified -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} so let's search libtorch by this path -> ${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/debug/")
                message(FATAL_ERROR "you don't have a such path ${PROJ_ROOT_DIR}/external/debug/ on your disk")
            endif()

            if (NOT EXISTS "${PROJ_ROOT_DIR}/external/debug/libtorch")
                message(FATAL_ERROR "you don't have an extracted pre-installed libtorch (${PROJ_ROOT_DIR}/external/debug/libtorch)!")
            endif()
        else()
            message(FATAL_ERROR "libtorch binaries only available for Debug and Release on Windows. Current build type: '${CMAKE_BUILD_TYPE}'")
        endif()
    endif()
else()
    set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Parallel build
include(ProcessorCount)
ProcessorCount(total_cores)
if(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2")
    set(ENV{MAKEFLAGS} "-j${used_cores}")
    message(STATUS "Building with ${used_cores} cores")
endif()

# CUDA setup
find_program(NVIDIA_SMI nvidia-smi)
if(NVIDIA_SMI)
    execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Detected GPU compute capability: ${DETECTED_COMPUTE_CAP}")
    set(LichtFeld-Studio_CUDA_ARCH "native")
else()
    message(WARNING "nvidia-smi not found! We are likely in CI, default to 8.6.")
    set(DETECTED_COMPUTE_CAP 8.6)
    set(LichtFeld-Studio_CUDA_ARCH "86")
endif()


# Detect supported SMs by querying 'ptxas --help' and filter by a minimum SM.
# We cannot use all or all-major since 5x and 6x versions are not supported
# in the CUDA code.
function(gs_detect_supported_sms OUT_VAR)
    set(options)
    set(oneValueArgs MIN_SM)
    cmake_parse_arguments(GS "${options}" "${oneValueArgs}" "" ${ARGN})

    # Use 'ptxas --help' to get the valid families
    set(_sms_list "")
    get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(PTXAS_EXECUTABLE NAMES ptxas
            HINTS "${_cuda_bin}" "${_cuda_bin}/.." "${CUDAToolkit_BIN_DIR}"
            PATHS ENV PATH)

    if(PTXAS_EXECUTABLE)
        execute_process(COMMAND "${PTXAS_EXECUTABLE}" --help
                OUTPUT_VARIABLE _ptxas_help
                ERROR_VARIABLE  _ptxas_err
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
        set(_help "${_ptxas_help}\n${_ptxas_err}")

        string(REGEX MATCHALL "sm_[0-9]+" _matches "${_help}")
        if(_matches)
            string(REPLACE "sm_" "" _nums "${_matches}")
            string(REPLACE " "  ";" _nums "${_nums}")
            separate_arguments(_nums)
            list(REMOVE_DUPLICATES _nums)
            list(SORT _nums COMPARE NATURAL)
            foreach(n IN LISTS _nums)
                if(GS_MIN_SM)
                    if(n GREATER_EQUAL ${GS_MIN_SM})
                        list(APPEND _sms_list ${n})
                    endif()
                else()
                    list(APPEND _sms_list ${n})
                endif()
            endforeach()
        endif()
    endif()

    if(NOT _sms_list)
        set(_sms_list 75 80 86 87 89 90 100 101 120)
        if(GS_MIN_SM)
            set(_filtered "")
            foreach(n IN LISTS _sms_list)
                if(n GREATER_EQUAL ${GS_MIN_SM})
                    list(APPEND _filtered ${n})
                endif()
            endforeach()
            set(_sms_list "${_filtered}")
        endif()
    endif()

    set(${OUT_VAR} "${_sms_list}" PARENT_SCOPE)
endfunction()

if(DEFINED TORCH_CUDA_ARCH_LIST)
    unset(TORCH_CUDA_ARCH_LIST CACHE)
endif()

if(BUILD_CUDA_ALL_SM)
    # Validate BUILD_CUDA_ALL_MIN_SM
    if(NOT BUILD_CUDA_ALL_MIN_SM MATCHES "^[0-9]+$")
        message(FATAL_ERROR "BUILD_CUDA_ALL_MIN_SM must be an integer like 70, 75, 86, 90.")
    endif()

    # Enforce lower bound 70
    if(BUILD_CUDA_ALL_MIN_SM LESS 70)
        message(FATAL_ERROR "BUILD_CUDA_ALL_MIN_SM must be >= 70 when BUILD_CUDA_ALL_SM=ON.")
    endif()

    # Fat binary for all the families
    gs_detect_supported_sms(CUDA_SMS MIN_SM ${BUILD_CUDA_ALL_MIN_SM})
    if(CUDA_SMS)
        # Build Torch list with decimals and add +PTX only to the highest SM
        list(SORT CUDA_SMS COMPARE NATURAL)
        list(GET CUDA_SMS -1 _max_sm)
        set(_torch_list "")
        foreach(sm IN LISTS CUDA_SMS)
            math(EXPR _maj "${sm}/10")
            math(EXPR _min "${sm}%10")
            set(_ver "${_maj}.${_min}")
            if(sm EQUAL ${_max_sm})
                list(APPEND _torch_list "${_ver}+PTX")
            else()
                list(APPEND _torch_list "${_ver}")
            endif()
        endforeach()
        list(JOIN _torch_list ";" TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}" CACHE STRING "Torch CUDA arch list" FORCE)

        # Per target CUDA architectures for CMake to be set after find_package(Torch)
        set(LichtFeld-Studio_CUDA_ARCH "${CUDA_SMS};${_max_sm}-virtual")
        message(STATUS "CUDA architectures (auto >=${BUILD_CUDA_ALL_MIN_SM}): ${LichtFeld-Studio_CUDA_ARCH}")
        message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
    else()
        message(FATAL_ERROR "Failed to detect any CUDA architecture >= ${BUILD_CUDA_ALL_MIN_SM}. Please check your CUDA toolkit installation.")
    endif()
else()
    # Native NVidia family only
    message(STATUS "CUDA architectures: ${LichtFeld-Studio_CUDA_ARCH} (native)")
    # Set Torch to the detected compute capability
    set(TORCH_CUDA_ARCH_LIST "${DETECTED_COMPUTE_CAP}" CACHE STRING "Torch CUDA arch list" FORCE)
    message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
endif()

enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)

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
find_package(Torch REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(args REQUIRED)
find_package(spdlog REQUIRED)
find_package(Freetype REQUIRED)
find_package(WebP REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenImageIO REQUIRED)

# TORCH_CUDA_ARCH_LIST is already defined by Torch
# use it here as the global default CUDA architecture list too
if(LichtFeld-Studio_CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
endif()

if(WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared") # when using INTEROP in windows static and dynamic symbols are mixed
endif()

# Check for CUDA-OpenGL interop capability (only if enabled)
if(ENABLE_CUDA_GL_INTEROP)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})
    check_cxx_source_compiles("
        #ifdef _WIN32
        #include <windows.h>
        #endif
        #include <cuda_runtime.h>
        #include <cuda_gl_interop.h>
        int main() {
            cudaGraphicsResource_t resource;
            return 0;
        }
    " CUDA_GL_INTEROP_FOUND)

    if(CUDA_GL_INTEROP_FOUND)
        message(STATUS "CUDA-OpenGL interop support: ENABLED")
        set(CUDA_GL_INTEROP_ENABLED 1)
    else()
        message(WARNING "CUDA-OpenGL interop support: DISABLED (headers not found)")
        set(CUDA_GL_INTEROP_ENABLED 0)
    endif()
else()
    message(STATUS "CUDA-OpenGL interop support: DISABLED (by user option)")
    set(CUDA_GL_INTEROP_FOUND FALSE)
    set(CUDA_GL_INTEROP_ENABLED 0)
endif()

# Create a configuration header
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
        @ONLY
)

# Project-wide definitions
add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
endif()

# =============================================================================
# Kernel libs
# =============================================================================
add_subdirectory(gsplat)
add_subdirectory(fastgs)

set_target_properties(gsplat_backend PROPERTIES CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}")
set_target_properties(fastgs_backend PROPERTIES CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}")

# =============================================================================
# SHARED CUDA KERNELS (only if needed)
# =============================================================================
set(KERNEL_SOURCES)
if(CUDA_GL_INTEROP_FOUND)
    list(APPEND KERNEL_SOURCES kernels/cuda_gl_interop.cu)
endif()

# Add SOG-related kernels
list(APPEND KERNEL_SOURCES
        kernels/morton_encoding.cu
        kernels/kmeans.cu
)

# Only create gaussian_kernels if there are kernels
if(KERNEL_SOURCES)
    add_library(gs_kernels STATIC ${KERNEL_SOURCES})

    set_target_properties(gs_kernels PROPERTIES
            CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_include_directories(gs_kernels
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPENGL_INCLUDE_DIRS}
            PRIVATE
    )

    target_link_libraries(gs_kernels
            PUBLIC
            CUDA::cudart
            CUDA::curand
            CUDA::cublas
            ${TORCH_LIBRARIES}
            glm::glm
            ${OPENGL_LIBRARIES}
            spdlog::spdlog
    )

    target_compile_options(gs_kernels PRIVATE
            # CUDA device code + MSVC host compiler flags (Windows only)
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo -Xcompiler=/Od -Xcompiler=/Z7>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG>

            # CUDA device code for non-Windows
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --ptxas-options=-v>
    )
else()
    # Create empty interface library if no kernels
    add_library(gs_kernels INTERFACE)
endif()

# =============================================================================
# LIBRARY MODULES
# =============================================================================
add_subdirectory(src/core_new)    # New core module (lfs::core namespace)
add_subdirectory(src/core)        # Foundation module
add_subdirectory(src/geometry)    # Geometry utilities
add_subdirectory(src/project)     # Project management
add_subdirectory(src/loader)      # Data loading
add_subdirectory(src/training)    # Training functionality
add_subdirectory(src/rendering)   # Rendering pipeline
add_subdirectory(src/visualizer)  # GUI and visualization

# =============================================================================
# MAIN EXECUTABLE
# =============================================================================
add_executable(${PROJECT_NAME} src/main.cpp resources/lichtfeld-studio.rc)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include

        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        gs_core
        gs_geometry
        gs_project
        gs_loader
        gs_training
        gs_rendering
        gs_visualizer
        gs_kernels
        gsplat_backend
        fastgs_backend

        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
        taywee::args
        OpenImageIO::OpenImageIO
)

# Platform-specific settings
if(WIN32)
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach()
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE GL GLU)

    find_path(TORCH_LIB_DIR libtorch_cpu.so
            PATHS "${Torch_DIR}/../../../lib"
            NO_DEFAULT_PATH)

    if(TORCH_LIB_DIR)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${TORCH_LIB_DIR}"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH_USE_LINK_PATH TRUE
        )
        message(STATUS "Torch library directory: ${TORCH_LIB_DIR}")
    else()
        message(WARNING "Could not find Torch library directory")
    endif()
endif()

# Build type configuration function
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

# Apply build configuration to all targets
configure_build_type(gs_core)
configure_build_type(gs_kernels)
configure_build_type(gsplat_backend)
configure_build_type(fastgs_backend)
configure_build_type(${PROJECT_NAME})

# =============================================================================
# TESTING (Optional)
# =============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests enabled. Build with 'ninja lichtfeld_tests' and run with 'ninja run_tests' or 'ctest'")
endif()

# =============================================================================
# BUILD INFO & OPTIMIZATIONS
# =============================================================================
message(STATUS "===========================================")
message(STATUS "Build Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Torch Version: ${Torch_VERSION}")
message(STATUS "  OpenGL Found: ${OPENGL_FOUND}")
message(STATUS "  FreeType Found: ${FREETYPE_FOUND}")
message(STATUS "  WebP Found: ${WebP_FOUND}")
message(STATUS "  LibArchive Found: ${LibArchive_FOUND}")
message(STATUS "  CUDA-GL Interop Option: ${ENABLE_CUDA_GL_INTEROP}")
message(STATUS "  CUDA-GL Interop Available: ${CUDA_GL_INTEROP_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  OpenImageIO Found: ${OpenImageIO_FOUND}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "===========================================")

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()