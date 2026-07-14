# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

set(LFS_ONNXRUNTIME_VERSION "1.23.2" CACHE STRING "ONNX Runtime prebuilt SDK version")
set(LFS_ONNXRUNTIME_ROOT "" CACHE PATH "Path to an unpacked ONNX Runtime SDK; empty downloads the pinned prebuilt SDK on supported platforms")
option(LFS_ONNXRUNTIME_USE_PREBUILT "Use pinned prebuilt ONNX Runtime GPU SDKs on supported platforms" ON)

function(_lfs_normalize_onnxruntime_root ROOT OUT_ROOT)
    set(_root "${ROOT}")
    if(NOT EXISTS "${_root}/include/onnxruntime_cxx_api.h")
        file(GLOB _nested_roots LIST_DIRECTORIES true "${_root}/onnxruntime-*")
        foreach(_nested_root IN LISTS _nested_roots)
            if(EXISTS "${_nested_root}/include/onnxruntime_cxx_api.h")
                set(_root "${_nested_root}")
                break()
            endif()
        endforeach()
    endif()
    set(${OUT_ROOT} "${_root}" PARENT_SCOPE)
endfunction()

function(_lfs_collect_onnxruntime_runtime_libs ROOT OUT_LIBS)
    if(WIN32)
        file(GLOB _runtime_libs
            "${ROOT}/lib/onnxruntime.dll"
            "${ROOT}/lib/onnxruntime_providers_cuda.dll"
            "${ROOT}/lib/onnxruntime_providers_shared.dll")
    else()
        file(GLOB _runtime_libs
            "${ROOT}/lib/libonnxruntime.so*"
            "${ROOT}/lib/libonnxruntime_providers_cuda.so*"
            "${ROOT}/lib/libonnxruntime_providers_shared.so*")
    endif()
    list(FILTER _runtime_libs INCLUDE REGEX "\\.(dll|dylib|so(\\.[0-9]+)*)$")
    list(REMOVE_DUPLICATES _runtime_libs)
    set(${OUT_LIBS} "${_runtime_libs}" PARENT_SCOPE)
endfunction()

function(_lfs_register_onnxruntime_root ROOT)
    _lfs_normalize_onnxruntime_root("${ROOT}" _root)
    set(_include_dir "${_root}/include")
    set(_lib_dir "${_root}/lib")

    if(NOT EXISTS "${_include_dir}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime SDK at '${ROOT}' does not contain include/onnxruntime_cxx_api.h")
    endif()

    if(WIN32)
        set(_runtime_library "${_lib_dir}/onnxruntime.dll")
        set(_import_library "${_lib_dir}/onnxruntime.lib")
        if(NOT EXISTS "${_runtime_library}" OR NOT EXISTS "${_import_library}")
            message(FATAL_ERROR "ONNX Runtime SDK at '${_root}' is missing onnxruntime.dll or onnxruntime.lib")
        endif()
    else()
        find_library(_runtime_library
            NAMES onnxruntime libonnxruntime
            PATHS "${_lib_dir}"
            NO_DEFAULT_PATH)
        if(NOT _runtime_library)
            message(FATAL_ERROR "ONNX Runtime SDK at '${_root}' is missing libonnxruntime")
        endif()
    endif()

    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_runtime_library}"
            INTERFACE_INCLUDE_DIRECTORIES "${_include_dir}")
        if(WIN32)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_IMPLIB "${_import_library}")
        endif()
    endif()

    _lfs_collect_onnxruntime_runtime_libs("${_root}" _runtime_libs)
    if(NOT _runtime_libs)
        message(FATAL_ERROR "ONNX Runtime SDK at '${_root}' does not contain CUDA/shared provider runtime libraries")
    endif()

    set(LFS_ONNXRUNTIME_INCLUDE_DIR "${_include_dir}" CACHE INTERNAL "ONNX Runtime include directory" FORCE)
    set(LFS_ONNXRUNTIME_TARGET "onnxruntime::onnxruntime" CACHE INTERNAL "ONNX Runtime CMake target" FORCE)
    set(LFS_ONNXRUNTIME_RUNTIME_LIBS "${_runtime_libs}" CACHE INTERNAL "ONNX Runtime runtime libraries to copy/bundle" FORCE)

    message(STATUS "ONNX Runtime: prebuilt SDK ${_root}")
endfunction()

function(_lfs_setup_onnxruntime_from_package)
    find_package(onnxruntime CONFIG REQUIRED)
    find_path(_include_dir
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime onnxruntime/core/session
        REQUIRED)

    if(TARGET onnxruntime::onnxruntime)
        set(_target onnxruntime::onnxruntime)
    elseif(TARGET onnxruntime)
        set(_target onnxruntime)
    else()
        find_library(_library NAMES onnxruntime REQUIRED)
        add_library(lfs_onnxruntime_external UNKNOWN IMPORTED GLOBAL)
        set_target_properties(lfs_onnxruntime_external PROPERTIES
            IMPORTED_LOCATION "${_library}"
            INTERFACE_INCLUDE_DIRECTORIES "${_include_dir}")
        set(_target lfs_onnxruntime_external)
    endif()

    set(_runtime_libs "")
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        file(GLOB _runtime_libs
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/*onnxruntime*"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/*onnxruntime*")
        list(FILTER _runtime_libs INCLUDE REGEX "\\.(dll|dylib|so(\\.[0-9]+)*)$")
        list(REMOVE_DUPLICATES _runtime_libs)
    endif()

    set(LFS_ONNXRUNTIME_INCLUDE_DIR "${_include_dir}" CACHE INTERNAL "ONNX Runtime include directory" FORCE)
    set(LFS_ONNXRUNTIME_TARGET "${_target}" CACHE INTERNAL "ONNX Runtime CMake target" FORCE)
    set(LFS_ONNXRUNTIME_RUNTIME_LIBS "${_runtime_libs}" CACHE INTERNAL "ONNX Runtime runtime libraries to copy/bundle" FORCE)
endfunction()

function(lfs_setup_onnxruntime)
    if(LFS_ONNXRUNTIME_ROOT)
        _lfs_register_onnxruntime_root("${LFS_ONNXRUNTIME_ROOT}")
        return()
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _processor)
    set(_is_x64 FALSE)
    if(_processor MATCHES "^(amd64|x86_64|x64)$")
        set(_is_x64 TRUE)
    endif()

    if(LFS_ONNXRUNTIME_USE_PREBUILT AND _is_x64)
        if(WIN32)
            set(_asset "onnxruntime-win-x64-gpu-${LFS_ONNXRUNTIME_VERSION}.zip")
            set(_sha256 "e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(_asset "onnxruntime-linux-x64-gpu-${LFS_ONNXRUNTIME_VERSION}.tgz")
            set(_sha256 "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53")
        endif()

        if(DEFINED _asset)
            set(_download_url
                "https://github.com/microsoft/onnxruntime/releases/download/v${LFS_ONNXRUNTIME_VERSION}/${_asset}")
            set(_cached_archive "${LFS_DOWNLOAD_CACHE_DIR}/${_asset}")
            set(_cache_lock "${LFS_DOWNLOAD_CACHE_DIR}/.onnxruntime.lock")
            file(LOCK "${_cache_lock}" GUARD FUNCTION TIMEOUT 600)

            if(EXISTS "${_cached_archive}")
                file(SHA256 "${_cached_archive}" _cached_sha256)
                if(NOT "${_cached_sha256}" STREQUAL "${_sha256}")
                    message(WARNING "Removing invalid cached ONNX Runtime archive: ${_cached_archive}")
                    file(REMOVE "${_cached_archive}")
                endif()
            endif()

            if(NOT EXISTS "${_cached_archive}")
                set(_partial_archive "${_cached_archive}.part")
                file(REMOVE "${_partial_archive}")
                message(STATUS "ONNX Runtime: downloading ${_asset}")
                file(DOWNLOAD
                    "${_download_url}"
                    "${_partial_archive}"
                    EXPECTED_HASH "SHA256=${_sha256}"
                    SHOW_PROGRESS
                    STATUS _download_status
                    TLS_VERIFY ON)
                list(GET _download_status 0 _download_code)
                list(GET _download_status 1 _download_message)
                if(NOT _download_code EQUAL 0)
                    file(REMOVE "${_partial_archive}")
                    message(FATAL_ERROR "ONNX Runtime download failed: ${_download_message}")
                endif()
                file(RENAME "${_partial_archive}" "${_cached_archive}")
            else()
                message(STATUS "ONNX Runtime: using cached archive ${_cached_archive}")
            endif()
            file(LOCK "${_cache_lock}" RELEASE)

            include(FetchContent)
            FetchContent_Declare(lfs_onnxruntime_prebuilt
                URL "${_cached_archive}"
                URL_HASH "SHA256=${_sha256}"
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
            FetchContent_MakeAvailable(lfs_onnxruntime_prebuilt)
            _lfs_register_onnxruntime_root("${lfs_onnxruntime_prebuilt_SOURCE_DIR}")
            return()
        endif()
    endif()

    _lfs_setup_onnxruntime_from_package()
endfunction()

lfs_setup_onnxruntime()
