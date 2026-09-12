# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: MIT

if(NOT FFX_SDK_STAGE_DIR OR NOT FFX_PATCH_DIR)
    message(FATAL_ERROR "FFX_SDK_STAGE_DIR and FFX_PATCH_DIR are required")
endif()

find_program(FFX_PATCH_EXECUTABLE NAMES patch REQUIRED)
if(NOT FFX_PATCH_EXECUTABLE)
    message(FATAL_ERROR
        "AMD FSR 3.1 Linux SDK patching requires the 'patch' executable, "
        "but it was not found in PATH")
endif()

# Do not use git apply: the staged SDK can be inside LichtFeld's Git worktree.
foreach(patch_name IN ITEMS
        0001-shader-compiler-linux-pch.patch
        0002-vk-backend-effect-context-alignment.patch)
    execute_process(
        COMMAND "${FFX_PATCH_EXECUTABLE}" --forward --dry-run --batch -p1
                -i "${FFX_PATCH_DIR}/${patch_name}"
        WORKING_DIRECTORY "${FFX_SDK_STAGE_DIR}"
        RESULT_VARIABLE patch_probe_result
        OUTPUT_VARIABLE patch_probe_output
        ERROR_VARIABLE patch_probe_error)
    if(patch_probe_result EQUAL 0)
        execute_process(
            COMMAND "${FFX_PATCH_EXECUTABLE}" --forward -p1
                    -i "${FFX_PATCH_DIR}/${patch_name}"
            WORKING_DIRECTORY "${FFX_SDK_STAGE_DIR}"
            COMMAND_ERROR_IS_FATAL ANY)
    elseif(patch_probe_output MATCHES "[Rr]eversed|previously applied")
        message(STATUS "Skipping already-applied patch ${patch_name}")
    else()
        message(FATAL_ERROR
            "Failed to apply ${patch_name}: ${patch_probe_output}"
            "${patch_probe_error}")
    endif()
endforeach()
