# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(Python COMPONENTS Interpreter)
set(PYTHONINTERP_FOUND ${Python_Interpreter_FOUND})
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

##################################################################
# Google C++ testing framework
##################################################################
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule")
  set(BUILD_GMOCK ON CACHE INTERNAL "Build gmock submodule")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/googletest/googletest/include)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/googletest/googlemock/include)
  set_target_properties(gtest gmock PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

function(CUDA_find_library out_path lib_name)
    find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                 PATH_SUFFIXES lib lib64)
endfunction()

find_package(CUDAToolkit REQUIRED)

set(CTK_SEARCH_PATHS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
)

include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_path(NVJPEG_INCLUDE
    NAMES nvjpeg.h
    PATHS
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
)
include_directories(SYSTEM ${NVJPEG_INCLUDE})

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/NVTX/c/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/dlpack/include)

if (BUILD_NVJPEG2K_EXT)
    if (WITH_DYNAMIC_NVJPEG2K)
        include(FetchContent)
        FetchContent_Declare(
            nvjpeg2k_headers
            URL      https://developer.download.nvidia.com/compute/nvjpeg2000/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-0.12.0.58_cuda12-archive.tar.xz
            URL_HASH SHA512=cbc872a74dc6e7dad62ef59933e719c775205a48e30cfbaccbefd3ec9d65feab52cd7078bf5d74b823c03c854fe257cebe38e6b013cf16ca62b1fb8f4f303b60
        )
        FetchContent_Populate(nvjpeg2k_headers)
        set(NVJPEG2K_SEARCH_PATHS "${nvjpeg2k_headers_SOURCE_DIR}/include")
    else()
        set(NVJPEG2K_SEARCH_PATHS ${CTK_SEARCH_PATHS})
        find_library(NVJPEG2K_LIBRARY ${NVJPEG2K_LIB_NAME} PATH_SUFFIXES lib lib64)
        if(NVJPEG2K_LIBRARY)
            message(STATUS "Found nvJPEG2k: ${NVJPEG2K_LIBRARY}")
        else()
            message(WARNING "nvJPEG2k library not found. Disabling nvJPEG2k and tests build.")
            set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVJPEG2K_EXT OFF)
        endif()
    endif()

    find_path(NVJPEG2K_INCLUDE NAMES nvjpeg2k.h HINTS ${NVJPEG2K_SEARCH_PATHS})

    if((NVJPEG2K_LIBRARY OR WITH_DYNAMIC_NVJPEG2K) AND NOT NVJPEG2K_INCLUDE)
        message(FATAL_ERROR
        "nvJPEG2k header file not found, please check your install"
        " or disable nvJPEG2k extension build with -DBUILD_NVJPEG2K_EXT=OFF")
    endif()
endif()

if (BUILD_NVJPEG2K_EXT)
    message(STATUS "Using NVJPEG2K_INCLUDE=${NVJPEG2K_INCLUDE}")
    include_directories(BEFORE SYSTEM ${NVJPEG2K_INCLUDE})
else()
    message(STATUS "nvJPEG2k extension build disabled")
endif()

if (BUILD_NVTIFF_EXT)
    if (WITH_DYNAMIC_NVTIFF)
        include(FetchContent)
        FetchContent_Declare(
           nvtiff_headers
           URL      https://developer.download.nvidia.com/compute/nvtiff/redist/libnvtiff/linux-x86_64/libnvtiff-linux-x86_64-0.9.0.86_cuda12-archive.tar.xz
           URL_HASH SHA512=3e57f823129ed60c7b4177b0cdfba32630272ba8610d8d57eabb4f4e8336fbabe3213f1b006f0cde4507fa19a2bb90cc91ae4ee0e253c0ccee6a6fddb7d3afde
        )
        FetchContent_Populate(nvtiff_headers)
        set(NVTIFF_SEARCH_PATHS "${nvtiff_headers_SOURCE_DIR}/include")
    else()
        set(NVTIFF_SEARCH_PATHS ${CTK_SEARCH_PATHS})
        find_library(NVTIFF_LIB ${NVTIFF_LIB_NAME} PATH_SUFFIXES lib lib64)
        if(NVTIFF_LIB)
            message(STATUS "Found nvTIFF: ${NVTIFF_LIB}")
        else()
            message(WARNING "nvTIFF library not found. Disabling nvTIFF extension and tests build.")
            set(BUILD_NVTIFF_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVTIFF_EXT OFF)
        endif()
    endif()

    find_path(NVTIFF_INCLUDE NAMES nvtiff.h HINTS ${NVTIFF_SEARCH_PATHS})

    if((NVTIFF_LIB OR WITH_DYNAMIC_NVTIFF) AND NOT NVTIFF_INCLUDE)
        message(FATAL_ERROR
        "nvTIFF header file not found, please check your install"
        " or disable nvTIFF extension build with -DBUILD_NVTIFF_EXT=OFF")
    endif()
endif()

if (BUILD_NVTIFF_EXT)
    message(STATUS "Using NVTIFF_INCLUDE=${NVTIFF_INCLUDE}")
    include_directories(BEFORE SYSTEM ${NVTIFF_INCLUDE})

    file(READ "${NVTIFF_INCLUDE}/nvtiff_version.h" _nvtiff_version_header)
    string(REGEX MATCH "#define[ \t]+NVTIFF_VER_MAJOR[ \t]+([0-9]+)" _nvtiff_major_match "${_nvtiff_version_header}")
    set(_nvtiff_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define[ \t]+NVTIFF_VER_MINOR[ \t]+([0-9]+)" _nvtiff_minor_match "${_nvtiff_version_header}")
    set(_nvtiff_minor "${CMAKE_MATCH_1}")
    if (NOT _nvtiff_major_match OR NOT _nvtiff_minor_match)
        message(FATAL_ERROR "Could not parse nvTIFF header version from ${NVTIFF_INCLUDE}/nvtiff_version.h")
    endif()
    if (_nvtiff_major EQUAL 0 AND _nvtiff_minor LESS 8)
        message(FATAL_ERROR "nvTIFF >= 0.8 is required. Found ${_nvtiff_major}.${_nvtiff_minor}.")
    endif()
else()
    message(STATUS "nvTIFF extension build disabled")
endif()

set(TIFF_LIBRARY_DEPS)

find_package(ZLIB)
if(NOT ZLIB_FOUND)
    message(STATUS "zlib not found - disabled")
else()
    message(STATUS "Using zlib at ${ZLIB_LIBRARIES}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZLIB_LIBRARIES})
endif()

find_package(ZSTD)
if(NOT DEFINED ZSTD_LIBRARY)
    message(FATAL_ERROR "zstd not found - disabled")
else()
    message(STATUS "Using zstd at ${ZSTD_LIBRARY}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZSTD_LIBRARY})
endif()

find_package(JPEG 62) # 1.5.3 version
if(NOT JPEG_FOUND)
    message(STATUS "libjpeg-turbo not found - disabled")
    set(BUILD_LIBJPEG_TURBO_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_LIBJPEG_TURBO_EXT OFF)
else()
    message(STATUS "Using libjpeg-turbo at ${JPEG_LIBRARIES}")
    include_directories(SYSTEM ${JPEG_INCLUDE_DIRS})
    list(APPEND TIFF_LIBRARY_DEPS ${JPEG_LIBRARIES})
endif()

# excludes the long-EOL libtiff 3.x series
find_package(TIFF 4.0)
if(NOT TIFF_FOUND)
    message(STATUS "libtiff not found - disabled")
    set(BUILD_LIBTIFF_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_LIBTIFF_EXT OFF)
else()
    # libtiff's exported static TIFF::tiff (picked up by FindTIFF on CMake >= 3.29)
    # links CMath::CMath, but its package config does not define it.
    if(NOT TARGET CMath::CMath)
        add_library(CMath::CMath INTERFACE IMPORTED)
    endif()
    message(STATUS "TIFF_INCLUDE_DIRS: ${TIFF_INCLUDE_DIRS}")
    message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")
    message(STATUS "libtiff dependencies: ${TIFF_LIBRARY_DEPS}")
    set_property(TARGET TIFF::TIFF APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${TIFF_LIBRARY_DEPS}")
endif()

if (NOT DEFINED OpenCV_VERSION AND (BUILD_OPENCV_EXT OR BUILD_TEST))
    if (WIN32)
        set(OpenCV_STATIC ON)
    endif()
    find_package(OpenCV 4.9 QUIET COMPONENTS core imgproc imgcodecs)

    if(NOT OpenCV_FOUND)
        message(STATUS "OpenCV not found - disabled")
        set(BUILD_OPENCV_EXT OFF CACHE BOOL INTERNAL)
        set(BUILD_OPENCV_EXT OFF)

        if (BUILD_TEST)
            message(WARNING "Native tests need OpenCV to run - Disabling tests")
            set(BUILD_TEST OFF CACHE BOOL INTERNAL)
            set(BUILD_TEST OFF)
        endif()
    else()
        message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"4.9\")")
        message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
        include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

        if(WIN32)
            set(NVIMGCODEC_OPENCV_RUNTIME_DLLS "")
            set(_opencv_runtime_search_dirs "")

            foreach(_opencv_dir_var IN ITEMS OpenCV_BIN_DIR OpenCV_RUNTIME_DIR)
                if(DEFINED ${_opencv_dir_var})
                    set(_opencv_dir_value "${${_opencv_dir_var}}")
                    if(_opencv_dir_value)
                        list(APPEND _opencv_runtime_search_dirs "${_opencv_dir_value}")
                    endif()
                endif()
            endforeach()

            if(DEFINED OpenCV_INSTALL_PATH)
                list(APPEND _opencv_runtime_search_dirs
                    "${OpenCV_INSTALL_PATH}/bin"
                    "${OpenCV_INSTALL_PATH}/x64/vc17/bin"
                    "${OpenCV_INSTALL_PATH}/x64/vc16/bin"
                    "${OpenCV_INSTALL_PATH}/x64/vc15/bin")
            endif()

            if(DEFINED OpenCV_DIR)
                list(APPEND _opencv_runtime_search_dirs
                    "${OpenCV_DIR}/../bin"
                    "${OpenCV_DIR}/../../bin"
                    "${OpenCV_DIR}/../../../bin")
            endif()

            foreach(_opencv_lib IN LISTS OpenCV_LIBRARIES)
                if(TARGET ${_opencv_lib})
                    foreach(_opencv_target_prop IN ITEMS
                        IMPORTED_LOCATION
                        IMPORTED_LOCATION_RELEASE
                        IMPORTED_LOCATION_RELWITHDEBINFO
                        IMPORTED_LOCATION_MINSIZEREL
                        IMPORTED_IMPLIB
                        IMPORTED_IMPLIB_RELEASE
                        IMPORTED_IMPLIB_RELWITHDEBINFO
                        IMPORTED_IMPLIB_MINSIZEREL)
                        get_target_property(_opencv_target_path ${_opencv_lib} ${_opencv_target_prop})
                        if(_opencv_target_path AND NOT _opencv_target_path MATCHES "NOTFOUND")
                            if(_opencv_target_path MATCHES "\\.dll$")
                                list(APPEND NVIMGCODEC_OPENCV_RUNTIME_DLLS "${_opencv_target_path}")
                            else()
                                get_filename_component(_opencv_target_dir "${_opencv_target_path}" DIRECTORY)
                                list(APPEND _opencv_runtime_search_dirs
                                    "${_opencv_target_dir}"
                                    "${_opencv_target_dir}/../bin"
                                    "${_opencv_target_dir}/../../bin")
                            endif()
                        endif()
                    endforeach()
                endif()
            endforeach()

            if(_opencv_runtime_search_dirs)
                list(REMOVE_DUPLICATES _opencv_runtime_search_dirs)
            endif()

            foreach(_opencv_runtime_dir IN LISTS _opencv_runtime_search_dirs)
                if(EXISTS "${_opencv_runtime_dir}")
                    file(GLOB _opencv_runtime_dlls CONFIGURE_DEPENDS "${_opencv_runtime_dir}/opencv_*.dll")
                    list(APPEND NVIMGCODEC_OPENCV_RUNTIME_DLLS ${_opencv_runtime_dlls})
                endif()
            endforeach()

            if(NVIMGCODEC_OPENCV_RUNTIME_DLLS)
                list(REMOVE_DUPLICATES NVIMGCODEC_OPENCV_RUNTIME_DLLS)
                message(STATUS "OpenCV runtime DLLs: ${NVIMGCODEC_OPENCV_RUNTIME_DLLS}")
            endif()
        endif()
    endif()
endif()

# #################################################################
# Boost preprocessor
# #################################################################
include_directories(${PROJECT_SOURCE_DIR}/external/boost/preprocessor/include)

set(NVIMGCODEC_COMMON_DEPENDENCIES "")
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES rt)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES pthread)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES m)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES dl)
