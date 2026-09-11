# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CMakeDependentOption)

set(NVIMGCODEC_PROJECT_NAME "nvimgcodec")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" NVIMGCODEC_VERSION
     REGEX "[0-9]+\\.[0-9]+\\.[0-9]+")

if (NVIMGCODEC_FLAVOR)
    set(NVIMGCODEC_FLAVOR "${NVIMGCODEC_FLAVOR}")  # could be nightly, weekly, etc if necessary
    set(NVIMGCODEC_FLAVOR_MINUS "-${NVIMGCODEC_FLAVOR}")  # -nightly, -weekly, etc
endif()

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "")

if (NOT ARCH)
    set(ARCH "x86_64")
endif()

if (NOT NVIMGCODEC_WHL_PLATFORM_NAME)
    if(UNIX)
        set(NVIMGCODEC_WHL_PLATFORM_NAME "manylinux_2_28_${ARCH}")
    else()
        set(NVIMGCODEC_WHL_PLATFORM_NAME "win_amd64")
    endif()
endif()

if(NOT BUILD_ID)
    set (BUILD_ID 0)
endif()

set(NVIMGCODEC_VERSION_WITH_BUILD "${NVIMGCODEC_VERSION}.${BUILD_ID}")
project(${NVIMGCODEC_PROJECT_NAME} VERSION ${NVIMGCODEC_VERSION} LANGUAGES CXX CUDA)
set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# Include custom cmake modules/scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

include(Utils)
parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)

add_definitions("-DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
add_definitions("-DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}")
add_definitions("-DCUDA_VERSION_PATCH=${CUDA_VERSION_PATCH}")

if(UNIX)
    set(NVIMGCODEC_SYSTEM_NAME "${CMAKE_SYSTEM_PROCESSOR}-linux")
else()
    set(NVIMGCODEC_SYSTEM_NAME "${CMAKE_SYSTEM_PROCESSOR}-win")
endif()

set(NVIMGCODEC_BUILD_SUFFIX "cuda${CUDA_VERSION_MAJOR}-${NVIMGCODEC_SYSTEM_NAME}")


# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_LIBRARY "Build library" ON)
log_option_value(BUILD_LIBRARY)
option(BUILD_TEST "Build tests" OFF)
log_option_value(BUILD_TEST)
option(BUILD_SAMPLES "Build samples" OFF)
log_option_value(BUILD_SAMPLES)
option(BUILD_CVCUDA_SAMPLES "Build CVCUDA samples" OFF)
log_option_value(BUILD_CVCUDA_SAMPLES)
option(BUILD_DOCS "Build documentation" OFF)
log_option_value(BUILD_DOCS)
option(BUILD_EXTENSIONS "Build extensions modules" ON)
log_option_value(BUILD_EXTENSIONS)

option(BUILD_NVJPEG_EXT "Build nvjpeg extensions module" ON)
option(BUILD_NVJPEG2K_EXT "Build nvjpeg2k extensions module" ON)
option(BUILD_NVTIFF_EXT "Build nvtiff extensions module" OFF)
option(BUILD_NVBMP_EXT "Build nvbmp extensions module" OFF)
option(BUILD_NVPNM_EXT "Build nvpnm extensions module" OFF)
option(BUILD_LIBJPEG_TURBO_EXT "Build libjpeg-turbo extensions module" OFF)
option(BUILD_LIBTIFF_EXT "Build libtiff extensions module" OFF)
option(BUILD_OPENCV_EXT "Build opencv extensions module" OFF)

if (NOT BUILD_EXTENSIONS)
    set(BUILD_NVJPEG_EXT OFF CACHE BOOL "Disable BUILD_NVJPEG_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL "Disable BUILD_NVJPEG2K_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_NVTIFF_EXT OFF CACHE BOOL "Disable BUILD_NVTIFF_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_NVBMP_EXT OFF CACHE BOOL "Disable BUILD_NVBMP_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_NVPNM_EXT OFF CACHE BOOL "Disable BUILD_NVPNM_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_LIBJPEG_TURBO_EXT OFF CACHE BOOL "Disable BUILD_LIBJPEG_TURBO_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_LIBTIFF_EXT OFF CACHE BOOL "Disable BUILD_LIBTIFF_EXT because BUILD_EXTENSIONS is disabled" FORCE)
    set(BUILD_OPENCV_EXT OFF CACHE BOOL "Disable BUILD_OPENCV_EXT because BUILD_EXTENSIONS is disabled" FORCE)
endif()
log_option_value(BUILD_NVJPEG_EXT)
log_option_value(BUILD_NVJPEG2K_EXT)
log_option_value(BUILD_NVTIFF_EXT)
log_option_value(BUILD_NVBMP_EXT)
log_option_value(BUILD_NVPNM_EXT)
log_option_value(BUILD_LIBJPEG_TURBO_EXT)
log_option_value(BUILD_LIBTIFF_EXT)
log_option_value(BUILD_OPENCV_EXT)

option(SKIP_NVTIFF_WITH_NVCOMP_TESTS "Skip nvTIFF extension tests that require nvCOMP" OFF)
if (NOT BUILD_NVTIFF_EXT)
    set(SKIP_NVTIFF_WITH_NVCOMP_TESTS ON CACHE BOOL "Enable SKIP_NVTIFF_WITH_NVCOMP_TESTS because BUILD_NVTIFF_EXT is disabled" FORCE)
endif()
log_option_value(SKIP_NVTIFF_WITH_NVCOMP_TESTS)
expose_via_preprocessor_macro(SKIP_NVTIFF_WITH_NVCOMP_TESTS)

option(BUILD_PYTHON "Build Python binding" OFF)
log_option_value(BUILD_PYTHON)

option(NVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR "Move nvimgcodec from install prefix to the python directory" ON)
option(BUILD_WHEEL "Build python wheel package" ON)
option(NVIMGCODEC_BUILD_DLPACK "Build dlpack from source. If disabled, it will use system installed dlpack" ON)
option(NVIMGCODEC_BUILD_PYBIND11 "Build pybind11 from source. If disabled, it will use system installed pybind11" ON)
if (NOT BUILD_PYTHON)
    set(NVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR OFF CACHE BOOL "Disable NVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR because BUILD_PYTHON is disabled" FORCE)
    set(BUILD_WHEEL OFF CACHE BOOL "Disable BUILD_WHEEL because BUILD_PYTHON is disabled" FORCE)
    set(NVIMGCODEC_BUILD_DLPACK OFF CACHE BOOL "Disable NVIMGCODEC_BUILD_DLPACK because BUILD_PYTHON is disabled" FORCE)
    set(NVIMGCODEC_BUILD_PYBIND11 OFF CACHE BOOL "Disable NVIMGCODEC_BUILD_PYBIND11 because BUILD_PYTHON is disabled" FORCE)
endif()
log_option_value(NVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR)
log_option_value(BUILD_WHEEL)
log_option_value(NVIMGCODEC_BUILD_DLPACK)
log_option_value(NVIMGCODEC_BUILD_PYBIND11)

option(WITH_DYNAMIC_LINK "Dynamically load CUDA libs (CUDA driver, nvjpeg, nvjpeg2k, nvtiff) at runtime via dlopen. If disabled, link with the system libraries" ON)
option(WITH_DYNAMIC_CUDA "Dynamically loads CUDA driver at runtime (via dlopen) vs. linking with the system libs (static or shared-object)" ON)
option(WITH_DYNAMIC_NVJPEG "Dynamically loads nvjpeg at runtime (via dlopen) vs. linking with the system libs (static or shared-object)" ON)
option(WITH_DYNAMIC_NVJPEG2K "Dynamically loads nvjpeg2k at runtime (via dlopen) vs. linking with the system libs (static or shared-object)" ON)
option(WITH_DYNAMIC_NVTIFF "Dynamically loads nvtiff at runtime (via dlopen) vs. linking with the system libs (static or shared-object)" ON)
option(WITH_SHARED_CUDA_LIBS "Link CUDA toolkit libraries (cudart, nvjpeg, nvjpeg2k, nvtiff) as shared rather than static when they are direct link dependencies. WITH_DYNAMIC_* still controls dlopen-based dispatch." OFF)

if (NOT WITH_DYNAMIC_LINK)
    set(WITH_DYNAMIC_CUDA OFF CACHE BOOL "Disable WITH_DYNAMIC_CUDA because WITH_DYNAMIC_LINK is disabled" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "Disable WITH_DYNAMIC_NVJPEG because WITH_DYNAMIC_LINK is disabled" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "Disable WITH_DYNAMIC_NVJPEG2K because WITH_DYNAMIC_LINK is disabled" FORCE)
    set(WITH_DYNAMIC_NVTIFF OFF CACHE BOOL "Disable WITH_DYNAMIC_NVTIFF because WITH_DYNAMIC_LINK is disabled" FORCE)
else()
    if (NOT BUILD_NVJPEG_EXT)
        set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "Disable WITH_DYNAMIC_NVJPEG because BUILD_NVJPEG_EXT is disabled" FORCE)
    endif()

    if (NOT BUILD_NVJPEG2K_EXT)
        set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "Disable WITH_DYNAMIC_NVJPEG2K because BUILD_NVJPEG2K_EXT is disabled" FORCE)
    endif()

    if (NOT BUILD_NVTIFF_EXT)
        set(WITH_DYNAMIC_NVTIFF OFF CACHE BOOL "Disable WITH_DYNAMIC_NVTIFF because BUILD_NVTIFF_EXT is disabled" FORCE)
    endif()
endif()

log_option_value(WITH_DYNAMIC_CUDA)
expose_via_preprocessor_macro(WITH_DYNAMIC_CUDA)

log_option_value(WITH_DYNAMIC_NVJPEG)
expose_via_preprocessor_macro(WITH_DYNAMIC_NVJPEG)

log_option_value(WITH_DYNAMIC_NVJPEG2K)
expose_via_preprocessor_macro(WITH_DYNAMIC_NVJPEG2K)

log_option_value(WITH_DYNAMIC_NVTIFF)
expose_via_preprocessor_macro(WITH_DYNAMIC_NVTIFF)

log_option_value(WITH_SHARED_CUDA_LIBS)
if (WITH_SHARED_CUDA_LIBS)
    set(_dynamic_link_ignored_shared_libs)
    if (WITH_DYNAMIC_CUDA)
        list(APPEND _dynamic_link_ignored_shared_libs "CUDA driver (WITH_DYNAMIC_CUDA)")
    endif()
    if (WITH_DYNAMIC_NVJPEG)
        list(APPEND _dynamic_link_ignored_shared_libs "nvjpeg (WITH_DYNAMIC_NVJPEG)")
    endif()
    if (WITH_DYNAMIC_NVJPEG2K)
        list(APPEND _dynamic_link_ignored_shared_libs "nvjpeg2k (WITH_DYNAMIC_NVJPEG2K)")
    endif()
    if (WITH_DYNAMIC_NVTIFF)
        list(APPEND _dynamic_link_ignored_shared_libs "nvtiff (WITH_DYNAMIC_NVTIFF)")
    endif()

    if (_dynamic_link_ignored_shared_libs)
        list(JOIN _dynamic_link_ignored_shared_libs ", " _dynamic_link_ignored_shared_libs_msg)
        message(WARNING
            "WITH_SHARED_CUDA_LIBS=ON only selects shared libraries for direct link dependencies. "
            "The following libraries are configured for dlopen via WITH_DYNAMIC_* and will ignore "
            "the shared/static link choice: ${_dynamic_link_ignored_shared_libs_msg}. "
            "Set the corresponding WITH_DYNAMIC_* option to OFF, or set WITH_DYNAMIC_LINK=OFF, "
            "if you want them linked as shared libraries.")
    endif()
endif()
if (WITH_SHARED_CUDA_LIBS)
    set(CUDA_CUDART_LIB CUDA::cudart)
    set(CUDA_NVJPEG_LIB CUDA::nvjpeg)
    set(NVJPEG2K_LIB_NAME nvjpeg2k)
    set(NVTIFF_LIB_NAME nvtiff)
else()
    set(CUDA_CUDART_LIB CUDA::cudart_static)
    set(CUDA_NVJPEG_LIB CUDA::nvjpeg_static)
    set(NVJPEG2K_LIB_NAME nvjpeg2k_static)
    set(NVTIFF_LIB_NAME nvtiff_static)
endif()

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYS_NAME)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    if(UNIX)
        set(CMAKE_INSTALL_LIBDIR "lib64")  # use lib64 instead of lib
    else()
        set(CMAKE_INSTALL_LIBDIR "lib")
    endif()
endif()
include(GNUInstallDirs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:\\Program Files\\NVIDIA nvImageCodec" CACHE PATH "where nvImageCodec will be installed" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "where nvImageCodec will be installed" FORCE)
    endif()
endif()

set(NVIMGCODEC_INSTALL_LIBDIR_DEFAULT "${CMAKE_INSTALL_LIBDIR}/libnvimgcodec/${CUDA_VERSION_MAJOR}")
if(WIN32)
    set(NVIMGCODEC_INSTALL_BINDIR_DEFAULT "${CMAKE_INSTALL_BINDIR}/${CUDA_VERSION_MAJOR}")
else()
    set(NVIMGCODEC_INSTALL_BINDIR_DEFAULT "${CMAKE_INSTALL_BINDIR}/nvimgcodec/${CUDA_VERSION_MAJOR}")
endif()
set(NVIMGCODEC_INSTALL_INCLUDEDIR_DEFAULT "${CMAKE_INSTALL_INCLUDEDIR}")
if(WIN32)
    # Runtime extension DLLs must be under bin for package-local discovery.
    # Import/static extension libraries stay in the CUDA-scoped library tree.
    set(NVIMGCODEC_INSTALL_EXTENSIONS_DIR_DEFAULT "${NVIMGCODEC_INSTALL_BINDIR_DEFAULT}/extensions")
    set(NVIMGCODEC_INSTALL_EXTENSION_LIBDIR_DEFAULT "${NVIMGCODEC_INSTALL_LIBDIR_DEFAULT}/extensions")
else()
    set(NVIMGCODEC_INSTALL_EXTENSIONS_DIR_DEFAULT "${NVIMGCODEC_INSTALL_LIBDIR_DEFAULT}/extensions")
    set(NVIMGCODEC_INSTALL_EXTENSION_LIBDIR_DEFAULT "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR_DEFAULT}")
endif()

set(NVIMGCODEC_INSTALL_LIBDIR "${NVIMGCODEC_INSTALL_LIBDIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec libraries")
set(NVIMGCODEC_INSTALL_BINDIR "${NVIMGCODEC_INSTALL_BINDIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec runtime binaries")
set(NVIMGCODEC_INSTALL_INCLUDEDIR "${NVIMGCODEC_INSTALL_INCLUDEDIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec public headers")
set(NVIMGCODEC_INSTALL_EXTENSIONS_DIR "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec extension modules")
set(NVIMGCODEC_INSTALL_EXTENSION_LIBDIR "${NVIMGCODEC_INSTALL_EXTENSION_LIBDIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec extension import/static libraries")
set(NVIMGCODEC_INSTALL_CMAKEDIR_DEFAULT "${NVIMGCODEC_INSTALL_LIBDIR}/cmake/nvimgcodec")
set(NVIMGCODEC_INSTALL_CMAKEDIR "${NVIMGCODEC_INSTALL_CMAKEDIR_DEFAULT}" CACHE PATH "Install directory for nvImageCodec CMake package files")

function(nvimgcodec_make_absolute_install_path output_var install_path)
    if(IS_ABSOLUTE "${install_path}")
        set(_install_path "${install_path}")
    else()
        set(_install_path "${CMAKE_INSTALL_PREFIX}/${install_path}")
    endif()
    file(TO_CMAKE_PATH "${_install_path}" _install_path)
    set(${output_var} "${_install_path}" PARENT_SCOPE)
endfunction()

nvimgcodec_make_absolute_install_path(NVIMGCODEC_FINAL_LIBDIR "${NVIMGCODEC_INSTALL_LIBDIR}")
nvimgcodec_make_absolute_install_path(NVIMGCODEC_FINAL_EXTENSIONS_DIR "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}")
set(NVIMGCODEC_DEFAULT_EXTENSIONS_PATH "${NVIMGCODEC_FINAL_EXTENSIONS_DIR}")
if(WIN32)
    string(REPLACE "\\" "/" NVIMGCODEC_DEFAULT_EXTENSIONS_PATH "${NVIMGCODEC_DEFAULT_EXTENSIONS_PATH}")
endif()
file(RELATIVE_PATH NVIMGCODEC_EXTENSIONS_RPATH_TO_LIBDIR "${NVIMGCODEC_FINAL_EXTENSIONS_DIR}" "${NVIMGCODEC_FINAL_LIBDIR}")
if(NVIMGCODEC_EXTENSIONS_RPATH_TO_LIBDIR STREQUAL "")
    set(NVIMGCODEC_EXTENSIONS_TO_LIBDIR_RPATH "$ORIGIN")
else()
    set(NVIMGCODEC_EXTENSIONS_TO_LIBDIR_RPATH "$ORIGIN/${NVIMGCODEC_EXTENSIONS_RPATH_TO_LIBDIR}")
endif()

include(cmake/CUDA_utils.cmake)

set(USED_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
message(STATUS "CUDA .cu files compiler: ${USED_CUDA_COMPILER}")

CUDA_find_supported_arch_values(CUDA_supported_archs ${USED_CUDA_COMPILER} ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED COMPARE NATURAL)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${USED_CUDA_COMPILER} ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}})"
      " is supported by ${USED_CUDA_COMPILER}, use one or more of: ${CUDA_supported_archs}")
endif()

CUDA_get_cmake_cuda_archs(CMAKE_CUDA_ARCHITECTURES ${CUDA_targeted_archs})
message(STATUS "Generated CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wno-deprecated-gpu-targets -Xfatbin -compress-all -t 0")

    include(third_party/generate_product_version)

    generate_product_version(ProductVersionFiles
        NAME "NVIDIA nvImageCodec library"
        VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR}
        VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR}
        VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH}
        VERSION_REVISION ${CMAKE_PROJECT_VERSION_TWEAK}
        COMPANY_NAME NVIDIA Corporation)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"-fvisibility=hidden -fvisibility-inlines-hidden -Wno-free-nonheap-object\" --Wno-deprecated-gpu-targets -Xfatbin -compress-all -t 0")

endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_BUILD_RPATH $ORIGIN)
list(APPEND CMAKE_INSTALL_RPATH $ORIGIN ${NVIMGCODEC_EXTENSIONS_TO_LIBDIR_RPATH})

if(UNIX)
    # CXX flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wl,--exclude-libs,ALL")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    message(STATUS "Added parallel build arguments to CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/test
    )

# Find all dependencies
include(Dependencies)

add_subdirectory(external)

if(BUILD_LIBRARY)
    add_subdirectory(src)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_PYTHON)
    if (NOT TARGET nvimgcodec_imgproc_static)
        add_subdirectory(src/imgproc)
    endif()
    add_subdirectory(python)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

set (NVIMGCODEC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
if(BUILD_EXTENSIONS)
    add_subdirectory(extensions)
endif()

if(UNIX)
add_custom_target(coverity COMMAND ../coverity.sh
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT
    "Running static analysis for nvImageCodec code with Coverity"
  VERBATIM
  )
endif()
