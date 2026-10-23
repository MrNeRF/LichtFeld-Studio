# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/WindowsSystemLibraryExclusions.cmake")
lfs_windows_system_library_regex(exclude "C:\\Windows\\")
foreach(path IN ITEMS "C:/Windows/System32/kernel32.dll" "c:/WINDOWS/SysWOW64/ole32.dll"
        "C:/Windows/WinSxS/amd64/component.dll"
        [[C:\Windows\system32/kernel32.dll]] [[c:\WINDOWS\SysWOW64\ole32.dll]])
    if(NOT path MATCHES "${exclude}")
        message(FATAL_ERROR "System library retained: ${path}")
    endif()
endforeach()
foreach(path IN ITEMS "D:/build-windows/bin/amdhip64.dll" "D:/deps-windows/x64-windows/bin/zstd.dll"
        "D:/System32-backup/lib.dll" "D:/Windows/System32/custom.dll"
        "C:/WindowsApps/vendor.dll" "C:/Windows.old/System32/kernel32.dll")
    if(path MATCHES "${exclude}")
        message(FATAL_ERROR "Application dependency incorrectly excluded: ${path}")
    endif()
endforeach()
lfs_windows_system_library_regex(escaped "D:/Win.dows+(system)")
if(NOT "d:/WIN.DOWS+(SYSTEM)/System32/test.dll" MATCHES "${escaped}" OR
        "D:/WinXdowssystem/System32/test.dll" MATCHES "${escaped}")
    message(FATAL_ERROR "Regex metacharacters in system root were not escaped")
endif()
message(STATUS "PASS Windows system-library packaging exclusions")
