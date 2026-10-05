# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Only the isolated helper has a 27.0 deployment target. The host application's
# CMAKE_OSX_DEPLOYMENT_TARGET stays 26.0; it never links a private framework.
include(GNUInstallDirs)
find_program(LFS_REFRAME_XCRUN xcrun REQUIRED)
execute_process(COMMAND ${LFS_REFRAME_XCRUN} --sdk macosx --show-sdk-version
    OUTPUT_VARIABLE _reframe_sdk OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _reframe_sdk_result)
if(NOT _reframe_sdk_result EQUAL 0 OR _reframe_sdk VERSION_LESS "27.0")
    message(FATAL_ERROR "Apple Reframe requires a macOS 27+ SDK. Disable LFS_ENABLE_APPLE_REFRAME to build with older SDKs.")
endif()
set(_reframe_source "${CMAKE_CURRENT_LIST_DIR}")
set(_reframe_build "${CMAKE_BINARY_DIR}/apple-reframe")
set(_reframe_helper "${CMAKE_BINARY_DIR}/bin/lfs-reframe")
add_custom_command(OUTPUT "${_reframe_helper}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_reframe_build}/cache" "${CMAKE_BINARY_DIR}/bin"
    COMMAND ${LFS_REFRAME_XCRUN} --sdk macosx clang -O2 -target arm64-apple-macos27.0
        -fobjc-arc -fmodules "-fmodules-cache-path=${_reframe_build}/cache"
        -c "${_reframe_source}/ModelResolver.m" -o "${_reframe_build}/ModelResolver.o"
    COMMAND ${LFS_REFRAME_XCRUN} --sdk macosx swiftc -O -parse-as-library
        -target arm64-apple-macos27.0 -module-cache-path "${_reframe_build}/cache"
        -I "${_reframe_source}/Stubs" -L "${_reframe_source}/Stubs" -lAlchemistBase
        -import-objc-header "${_reframe_source}/ModelResolver.h"
        "${_reframe_source}/ReframeHelper.swift" "${_reframe_build}/ModelResolver.o"
        -o "${_reframe_helper}"
    DEPENDS "${_reframe_source}/ModelResolver.m" "${_reframe_source}/ModelResolver.h"
        "${_reframe_source}/ReframeHelper.swift"
        "${_reframe_source}/Stubs/AlchemistBase.swiftinterface"
        "${_reframe_source}/Stubs/libAlchemistBase.tbd"
    VERBATIM)
add_custom_target(lfs_reframe_helper DEPENDS "${_reframe_helper}")
install(PROGRAMS "${_reframe_helper}" DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES "${_reframe_source}/NOTICE" DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/apple-reframe")
