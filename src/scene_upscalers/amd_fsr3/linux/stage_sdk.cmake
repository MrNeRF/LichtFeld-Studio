# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: MIT

if(NOT FFX_SDK_SOURCE_DIR OR NOT FFX_SDK_DEST_DIR)
    message(FATAL_ERROR "FFX_SDK_SOURCE_DIR and FFX_SDK_DEST_DIR are required")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${FFX_SDK_DEST_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${FFX_SDK_SOURCE_DIR}" "${FFX_SDK_DEST_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
