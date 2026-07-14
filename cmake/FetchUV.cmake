# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# FetchUV.cmake
# Downloads the uv Python package manager for bundling with portable builds.
# uv is a fast, Rust-based Python package manager from Astral.

set(UV_VERSION "0.10.2" CACHE STRING "UV version to download")

# Platform-specific download URL and binary name
if(WIN32)
    set(UV_BINARY_NAME "uv.exe")
    set(UV_ARCHIVE_EXT ".zip")
    set(UV_PLATFORM "x86_64-pc-windows-msvc")
    if(UV_VERSION STREQUAL "0.10.2")
        set(UV_ARCHIVE_SHA256 "493ebbe0e06128d6ee4905e1ed5e2a433fb0f7cfc08b0eaca9fab4ca76778ae1")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(UV_BINARY_NAME "uv")
    set(UV_ARCHIVE_EXT ".tar.gz")
    set(UV_PLATFORM "${CMAKE_HOST_SYSTEM_PROCESSOR}-unknown-linux-gnu")
    if(UV_VERSION STREQUAL "0.10.2")
        if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(UV_ARCHIVE_SHA256 "6aa4576c31f791c0b9d4739e256d07358d45e7535695287fec03cf6839e25512")
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(UV_ARCHIVE_SHA256 "4998f545234d52fc6f1280827d392f00a9278295050d59c53a776546dbf0124d")
        endif()
    endif()
else()
    message(WARNING "FetchUV: Unsupported platform, uv will not be bundled")
    return()
endif()

set(UV_DOWNLOAD_URL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_PLATFORM}${UV_ARCHIVE_EXT}")
set(UV_WORK_DIR "${CMAKE_BINARY_DIR}/_deps/uv")
set(UV_ARCHIVE_DIR "${LFS_DOWNLOAD_CACHE_DIR}")
set(UV_ARCHIVE_PATH "${UV_ARCHIVE_DIR}/uv-${UV_VERSION}-${UV_PLATFORM}${UV_ARCHIVE_EXT}")
set(UV_EXTRACT_DIR "${UV_WORK_DIR}/extract")
# Windows zip extracts flat, Linux tar.gz extracts to subdirectory
if(WIN32)
    set(UV_BINARY_PATH "${UV_EXTRACT_DIR}/${UV_BINARY_NAME}")
else()
    set(UV_BINARY_PATH "${UV_EXTRACT_DIR}/uv-${UV_PLATFORM}/${UV_BINARY_NAME}")
endif()

# Function to download and extract uv
function(fetch_uv)
    # Check if already downloaded
    if(EXISTS "${UV_BINARY_PATH}")
        message(STATUS "FetchUV: uv ${UV_VERSION} already available")
        return()
    endif()

    file(MAKE_DIRECTORY "${UV_ARCHIVE_DIR}")
    file(LOCK "${UV_ARCHIVE_DIR}/.uv.lock" GUARD FUNCTION TIMEOUT 600)

    if(EXISTS "${UV_ARCHIVE_PATH}" AND DEFINED UV_ARCHIVE_SHA256)
        file(SHA256 "${UV_ARCHIVE_PATH}" _cached_sha256)
        if(NOT "${_cached_sha256}" STREQUAL "${UV_ARCHIVE_SHA256}")
            message(WARNING "FetchUV: Removing invalid cached archive ${UV_ARCHIVE_PATH}")
            file(REMOVE "${UV_ARCHIVE_PATH}")
        endif()
    endif()

    if(EXISTS "${UV_ARCHIVE_PATH}")
        message(STATUS "FetchUV: Using cached archive ${UV_ARCHIVE_PATH}")
    else()
        message(STATUS "FetchUV: Downloading uv ${UV_VERSION} for ${UV_PLATFORM}")
        message(STATUS "FetchUV: URL: ${UV_DOWNLOAD_URL}")
        set(_partial_archive "${UV_ARCHIVE_PATH}.part")
        file(REMOVE "${_partial_archive}")
        set(_hash_arguments)
        if(DEFINED UV_ARCHIVE_SHA256)
            list(APPEND _hash_arguments EXPECTED_HASH "SHA256=${UV_ARCHIVE_SHA256}")
        endif()
        file(DOWNLOAD
            "${UV_DOWNLOAD_URL}"
            "${_partial_archive}"
            SHOW_PROGRESS
            STATUS _download_status
            TLS_VERIFY ON
            ${_hash_arguments}
        )

        list(GET _download_status 0 _status_code)
        list(GET _download_status 1 _status_string)

        if(NOT _status_code EQUAL 0)
            file(REMOVE "${_partial_archive}")
            message(WARNING "FetchUV: Download failed: ${_status_string}")
            message(WARNING "FetchUV: uv will not be bundled. Users can install it manually.")
            return()
        endif()
        file(RENAME "${_partial_archive}" "${UV_ARCHIVE_PATH}")
    endif()

    message(STATUS "FetchUV: Extracting...")

    # Extract the archive
    file(MAKE_DIRECTORY "${UV_EXTRACT_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${UV_ARCHIVE_PATH}"
        WORKING_DIRECTORY "${UV_EXTRACT_DIR}"
        RESULT_VARIABLE _extract_result
    )

    if(NOT _extract_result EQUAL 0)
        message(WARNING "FetchUV: Extraction failed")
        return()
    endif()

    if(EXISTS "${UV_BINARY_PATH}")
        message(STATUS "FetchUV: uv ${UV_VERSION} ready at ${UV_BINARY_PATH}")
    else()
        message(WARNING "FetchUV: Binary not found after extraction at ${UV_BINARY_PATH}")
    endif()
endfunction()

# Copy uv to build/bin/ for development builds
function(copy_uv_to_build)
    if(NOT EXISTS "${UV_BINARY_PATH}")
        return()
    endif()

    set(_DEST_DIR "${CMAKE_BINARY_DIR}/bin")
    if(EXISTS "${_DEST_DIR}/${UV_BINARY_NAME}")
        return()
    endif()

    file(MAKE_DIRECTORY "${_DEST_DIR}")
    file(COPY "${UV_BINARY_PATH}" DESTINATION "${_DEST_DIR}")
    message(STATUS "FetchUV: Copied uv to ${_DEST_DIR}/")
endfunction()

# Function to install uv to the bin directory
function(install_uv)
    # First try the downloaded binary
    if(EXISTS "${UV_BINARY_PATH}")
        install(PROGRAMS "${UV_BINARY_PATH}"
            DESTINATION bin
            COMPONENT runtime
        )
        message(STATUS "FetchUV: Will install uv to bin/")
        return()
    endif()

    # Fall back to system uv if available
    find_program(SYSTEM_UV_PATH uv)
    if(SYSTEM_UV_PATH)
        message(STATUS "FetchUV: Using system uv at ${SYSTEM_UV_PATH}")
        install(PROGRAMS "${SYSTEM_UV_PATH}"
            DESTINATION bin
            COMPONENT runtime
        )
    else()
        message(STATUS "FetchUV: uv not found - package management will be unavailable in portable build")
    endif()
endfunction()
